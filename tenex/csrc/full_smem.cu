#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level reduction ──────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── Full-SMEM Transfer Entropy kernel ────────────────────────────────────────
//
// Grid : (n_pairs,)   — one block per variable pair
// Block: (BLOCK,)     — power of 2, >= max(L*K, 2*B^2), <= 1024
//
// bin_arrs (n_vars, T, K) is read directly using pairs[] indices.
// For typical datasets (~6 MB), bin_arrs fits in GPU L2 cache; adjacent
// blocks sharing target/source variables pay L2 latency rather than global-memory latency.
//
// Shared memory layout (dynamic):
//   [0                 .. B3*4)                 : int32  cnt3[B3]
//   [B3*4              .. B3*4 + B2*4)          : float  cnt2a[B2]   (xt1,xt)
//   [B3*4 + B2*4       .. B3*4 + 2*B2*4)       : float  cnt2b[B2]   (xt,yt)
//   [B3*4 + 2*B2*4     .. B3*4 + 2*B2*4 + B*4) : float  cnt1[B]     (xt)
//   [...               .. + WARPS*4)            : float  warp_buf[WARPS]
//
template<typename BinType>
__global__ void te_smem_kernel(
    const BinType*  __restrict__ bin_arrs,  // (n_vars * T * K,) int8/int16/int32, row-major
    const int64_t* __restrict__ pairs,     // (n_pairs * 2,)     int64, row-major
    float*         __restrict__ te_ptr,    // (n_pairs,)         float32 output
    int T, int K, int tau, int L,           // L = T - tau
    int B, int B2, int B3
) {
    extern __shared__ char smem[];

    int32_t* cnt3     = (int32_t*)smem;
    float*   cnt2a    = (float*)(cnt3 + B3);
    float*   cnt2b    = cnt2a + B2;
    float*   cnt1_sm  = cnt2b + B2;
    float*   warp_buf = cnt1_sm + B;

    const int pair_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;
    const int N       = L * K;
    const int TK      = T * K;

    // Gene indices for this pair
    int target_var = (int)pairs[pair_id * 2    ];
    int source_var = (int)pairs[pair_id * 2 + 1];

    // Base pointers into bin_arrs (BinType: int8/int16/int32 — widening to int is free)
    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // ── Phase 0: clear cnt3 ─────────────────────────────────────────────────
    for (int b = tid; b < B3; b += BLOCK)
        cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram directly from bin_arrs ─────────────────────
    // K=1 fast path eliminates integer division (common case for scRNA-seq)
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = __ldg(&target_base[tau + i]);
            int xtv  = __ldg(&target_base[i]);
            int ytv  = __ldg(&source_base[i]);
            atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = __ldg(&target_base[(tau + t) * K + c]);
            int xtv  = __ldg(&target_base[t        * K + c]);
            int ytv  = __ldg(&source_base[t        * K + c]);
            atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: marginals — grid-stride loops work for any BLOCK size ────────
    // BUG FIX: original used if(tid<B2)/if(tid>=B2&&tid<2*B2) which fails when
    // 2*B^2 > block_size (e.g. B=27, block_size=1024: 2*729=1458 > 1024).
    for (int j = tid; j < B; j += BLOCK) {     // cnt1[j]
        float s = 0.0f;
        for (int i = 0; i < B; i++)
            for (int k = 0; k < B; k++)
                s += (float)cnt3[i * B2 + j * B + k];
        cnt1_sm[j] = s;
    }
    for (int t = tid; t < B2; t += BLOCK) {    // cnt2a[i,j] = Σ_k cnt3[i,j,k]
        int i = t / B, j = t % B;
        float s = 0.0f;
        for (int k = 0; k < B; k++)
            s += (float)cnt3[i * B2 + j * B + k];
        cnt2a[t] = s;
    }
    for (int t = tid; t < B2; t += BLOCK) {    // cnt2b[j,k] = Σ_i cnt3[i,j,k]
        int j = t / B, k = t % B;
        float s = 0.0f;
        for (int i = 0; i < B; i++)
            s += (float)cnt3[i * B2 + j * B + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula ─────────────────────────────────────────────────
    float te_local = 0.0f;
    for (int b = tid; b < B3; b += BLOCK) {
        int i = b / B2;
        int j = (b % B2) / B;
        int k = b % B;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * B + j];
        float c2b = cnt2b[j * B + k];
        float c1  = cnt1_sm[j];
        float denom = c2a * c2b;

        if (c3 > 0.0f && denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: block reduction ────────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = te_local;
    __syncthreads();

    // All lanes participate; inactive lanes contribute 0.0f
    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        te_ptr[pair_id] = te_local / (float)N;
}


// ── Opt-in shared memory query ────────────────────────────────────────────────
// Returns the maximum configurable dynamic shared memory per block for the
// current device.  On RTX A5000 (GA102 Ampere) this is 99 KB; on A100 ~163 KB;
// The default limit (48 KB) is raised automatically by te_smem_launch before
// each kernel call, so callers only need this value for the fallback decision.
int64_t get_smem_optin() {
    int dev = 0, val = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)val;
}

// ── Pair-free kernel: compute (tgt, src) from linear pair_id ─────────────────
//
// Eliminates the need for an explicit pairs tensor.
// pair_id = tgt * (n_vars - 1) + src_local
// src = src_local + (src_local >= tgt ? 1 : 0)
//
// Grid : (n_pairs_local,)  — one block per variable pair
// pair_offset allows multi-GPU splitting by index range.
//
template<typename BinType>
__global__ void te_smem_kernel_pairfree(
    const BinType*  __restrict__ bin_arrs,  // (n_vars * T * K,) int8/int16/int32, row-major
    float*         __restrict__ te_ptr,    // (n_pairs_local,)   float32 output
    int T, int K, int tau, int L,
    int B, int B2, int B3,
    int n_vars,
    int64_t pair_offset                    // starting global pair index
) {
    extern __shared__ char smem[];

    int32_t* cnt3     = (int32_t*)smem;
    float*   cnt2a    = (float*)(cnt3 + B3);
    float*   cnt2b    = cnt2a + B2;
    float*   cnt1_sm  = cnt2b + B2;
    float*   warp_buf = cnt1_sm + B;

    const int local_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int BLOCK    = blockDim.x;
    const int WARPS    = BLOCK / 32;
    const int N        = L * K;
    const int TK       = T * K;

    // Compute (target_var, source_var) from linear index
    const int64_t global_pair_id = (int64_t)local_id + pair_offset;
    const int nm1 = n_vars - 1;
    int target_var = (int)(global_pair_id / nm1);
    int source_var = (int)(global_pair_id % nm1);
    if (source_var >= target_var) source_var++;

    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // ── Phase 0: clear cnt3
    for (int b = tid; b < B3; b += BLOCK)
        cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram directly from bin_arrs
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = (int)__ldg(&target_base[tau + i]);
            int xtv  = (int)__ldg(&target_base[i]);
            int ytv  = (int)__ldg(&source_base[i]);
            atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = (int)__ldg(&target_base[(tau + t) * K + c]);
            int xtv  = (int)__ldg(&target_base[t        * K + c]);
            int ytv  = (int)__ldg(&source_base[t        * K + c]);
            atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: marginals
    for (int j = tid; j < B; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < B; i++)
            for (int k = 0; k < B; k++)
                s += (float)cnt3[i * B2 + j * B + k];
        cnt1_sm[j] = s;
    }
    for (int t = tid; t < B2; t += BLOCK) {
        int i = t / B, j = t % B;
        float s = 0.0f;
        for (int k = 0; k < B; k++)
            s += (float)cnt3[i * B2 + j * B + k];
        cnt2a[t] = s;
    }
    for (int t = tid; t < B2; t += BLOCK) {
        int j = t / B, k = t % B;
        float s = 0.0f;
        for (int i = 0; i < B; i++)
            s += (float)cnt3[i * B2 + j * B + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula
    float te_local = 0.0f;
    for (int b = tid; b < B3; b += BLOCK) {
        int i = b / B2;
        int j = (b % B2) / B;
        int k = b % B;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * B + j];
        float c2b = cnt2b[j * B + k];
        float c1  = cnt1_sm[j];
        float denom = c2a * c2b;

        if (c3 > 0.0f && denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: block reduction
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        te_ptr[local_id] = te_local / (float)N;
}


// ── Dtype dispatch helpers ────────────────────────────────────────────────────
template<typename BinType>
void _launch_smem(
    torch::Tensor& bin_arrs, torch::Tensor& pairs, torch::Tensor& te,
    int T, int K, int tau, int L, int B, int B2, int B3, int block_size, size_t smem
) {
    cudaFuncSetAttribute(
        te_smem_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    int n_pairs = (int)pairs.size(0);
    te_smem_kernel<BinType><<<n_pairs, block_size, smem>>>(
        bin_arrs.data_ptr<BinType>(),
        pairs.data_ptr<int64_t>(),
        te.data_ptr<float>(),
        T, K, tau, L, B, B2, B3
    );
}

template<typename BinType>
void _launch_smem_pairfree(
    torch::Tensor& bin_arrs, torch::Tensor& te,
    int T, int K, int tau, int L, int B, int B2, int B3,
    int block_size, int n_vars, int64_t pair_offset, int64_t n_pairs_local, size_t smem
) {
    cudaFuncSetAttribute(
        te_smem_kernel_pairfree<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    te_smem_kernel_pairfree<BinType><<<(int)n_pairs_local, block_size, smem>>>(
        bin_arrs.data_ptr<BinType>(),
        te.data_ptr<float>(),
        T, K, tau, L, B, B2, B3,
        n_vars, pair_offset
    );
}

// ── Python-facing launcher ────────────────────────────────────────────────────
torch::Tensor te_smem_launch(
    torch::Tensor bin_arrs,  // (n_vars, T, K) int8/int16/int32 CUDA
    torch::Tensor pairs,     // (n_pairs, 2)    int64 CUDA
    int T, int K, int tau,
    int B, int B2, int B3,
    int block_size
) {
    int64_t n_pairs64 = pairs.size(0);
    TORCH_CHECK(n_pairs64 <= (int64_t)INT_MAX,
        "n_pairs (", n_pairs64, ") exceeds INT_MAX grid limit; split into smaller batches");
    auto te = torch::empty({n_pairs64}, pairs.options().dtype(torch::kFloat32));

    int L     = T - tau;
    int warps = block_size / 32;
    size_t smem = (size_t)B3 * 4
                + (size_t)B2 * 4 * 2
                + (size_t)B  * 4
                + (size_t)warps * 4;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_smem<int8_t>(bin_arrs, pairs, te, T, K, tau, L, B, B2, B3, block_size, smem);
    } else if (dtype == torch::kInt16) {
        _launch_smem<int16_t>(bin_arrs, pairs, te, T, K, tau, L, B, B2, B3, block_size, smem);
    } else {
        _launch_smem<int32_t>(bin_arrs, pairs, te, T, K, tau, L, B, B2, B3, block_size, smem);
    }
    return te;
}

// ── Pair-free launcher ──────────────────────────────────────────────────────
torch::Tensor te_smem_launch_pairfree(
    torch::Tensor bin_arrs,  // (n_vars, T, K) int8/int16/int32 CUDA
    int T, int K, int tau,
    int B, int B2, int B3,
    int block_size,
    int n_vars,
    int64_t pair_offset,
    int64_t n_pairs_local
) {
    TORCH_CHECK(n_pairs_local <= (int64_t)INT_MAX,
        "n_pairs_local (", n_pairs_local, ") exceeds INT_MAX grid limit; "
        "use multi-GPU or call with smaller chunks");

    auto te = torch::empty({n_pairs_local},
                           bin_arrs.options().dtype(torch::kFloat32));
    int L     = T - tau;
    int warps = block_size / 32;
    size_t smem = (size_t)B3 * 4
                + (size_t)B2 * 4 * 2
                + (size_t)B  * 4
                + (size_t)warps * 4;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_smem_pairfree<int8_t>(bin_arrs, te, T, K, tau, L, B, B2, B3, block_size, n_vars, pair_offset, n_pairs_local, smem);
    } else if (dtype == torch::kInt16) {
        _launch_smem_pairfree<int16_t>(bin_arrs, te, T, K, tau, L, B, B2, B3, block_size, n_vars, pair_offset, n_pairs_local, smem);
    } else {
        _launch_smem_pairfree<int32_t>(bin_arrs, te, T, K, tau, L, B, B2, B3, block_size, n_vars, pair_offset, n_pairs_local, smem);
    }
    return te;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("te_smem_launch", &te_smem_launch,
          "Full-SMEM Transfer Entropy kernel (direct bin_arrs access, cnt3 fully in SMEM)");
    m.def("te_smem_launch_pairfree", &te_smem_launch_pairfree,
          "Full-SMEM TE kernel — pair-free mode (no explicit pairs tensor)");
    m.def("get_smem_optin", &get_smem_optin,
          "Max configurable dynamic shared memory per block (opt-in) for current device");
}
