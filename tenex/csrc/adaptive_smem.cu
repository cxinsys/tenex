#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level reduction ───────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── Adaptive-SMEM kernel: per-var asymmetric B (xt1,xt ~ b_t; yt ~ b_s) ────────
//
// Grid : (n_pairs,) — one block per variable pair
// Block: (BLOCK,)   — power of 2, >= 128, <= 1024
//
// SMEM layout (dynamic, sizes in bytes):
//   [0             .. b_t2_s*4)                    : int32  cnt3[b_t*b_t*b_s]
//   [b_t2_s*4        .. b_t2_s*4 + b_t2*4)            : float  cnt2a[b_t*b_t]   (xt1,xt)
//   [b_t2_s*4+b_t2*4  .. b_t2_s*4 + b_t2*4 + b_tb_s*4)  : float  cnt2b[b_t*b_s]   (xt,yt)
//   [.. + b_tb_s*4   .. + b_t*4)                    : float  cnt1[b_t]        (xt)
//   [.. + b_t*4     .. + WARPS*4)                 : float  warp_buf[WARPS]
//
// The kernel is launched in SMEM-sorted batches (Python side).
// All pairs in one batch satisfy: (b_t*b_t*b_s + b_t*b_t + b_t*b_s + b_t)*4 <= smem_budget.
//
template<typename BinType>
__global__ void te_adaptive_smem_kernel(
    const BinType*  __restrict__ bin_arrs,    // (n_vars * T * K,) int8/int16/int32, row-major
    const int64_t* __restrict__ pairs,        // (n_pairs * 2,)     int64
    const int32_t* __restrict__ n_bins_arr,  // (n_vars,)          int32
    float*         __restrict__ te_ptr,       // (n_pairs,)          float32 output
    int T, int K, int tau, int L              // L = T - tau
) {
    extern __shared__ char smem_buf[];

    const int pair_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;
    const int N       = L * K;
    const int TK      = T * K;

    // Per-pair bin counts (read from n_bins_arr)
    int target_var = (int)pairs[pair_id * 2    ];
    int source_var = (int)pairs[pair_id * 2 + 1];
    int b_t  = n_bins_arr[target_var];   // bins for xt1, xt
    int b_s  = n_bins_arr[source_var];   // bins for yt
    int b_t2 = b_t * b_t;
    int b_t2_s = b_t2 * b_s;

    // SMEM pointers — offsets depend on this pair's b_t, b_s
    int32_t* cnt3    = (int32_t*)smem_buf;
    float*   cnt2a   = (float*)(cnt3 + b_t2_s);
    float*   cnt2b   = cnt2a + b_t2;
    float*   cnt1_sm = cnt2b + b_t * b_s;
    float*   warp_buf = cnt1_sm + b_t;

    // Base pointers into bin_arrs (BinType: int8/int16/int32 — widening to int is free)
    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // ── Phase 0: clear cnt3 ────────────────────────────────────────────────
    for (int b = tid; b < b_t2_s; b += BLOCK)
        cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram (K==1 fast path) ──────────────────────────
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = __ldg(&target_base[tau + i]);
            int xtv  = __ldg(&target_base[i]);
            int ytv  = __ldg(&source_base[i]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = __ldg(&target_base[(tau + t) * K + c]);
            int xtv  = __ldg(&target_base[t        * K + c]);
            int ytv  = __ldg(&source_base[t        * K + c]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: marginals — loop-based, works for any BLOCK vs b_t2 ───────

    // cnt1[j] = Σ_{i,k} cnt3[i, j, k]
    for (int j = tid; j < b_t; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            for (int k = 0; k < b_s; k++)
                s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt1_sm[j] = s;
    }

    // cnt2a[i, j] = Σ_k cnt3[i, j, k]
    for (int t = tid; t < b_t2; t += BLOCK) {
        int i = t / b_t, j = t % b_t;
        float s = 0.0f;
        for (int k = 0; k < b_s; k++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2a[t] = s;
    }

    // cnt2b[j, k] = Σ_i cnt3[i, j, k]
    for (int t = tid; t < b_t * b_s; t += BLOCK) {
        int j = t / b_s, k = t % b_s;
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula ────────────────────────────────────────────────
    float te_local = 0.0f;
    for (int b = tid; b < b_t2_s; b += BLOCK) {
        int i  = b / (b_t * b_s);
        int jk = b % (b_t * b_s);
        int j  = jk / b_s;
        int k  = jk % b_s;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * b_t + j];
        float c2b = cnt2b[j * b_s + k];
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

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        te_ptr[pair_id] = te_local / (float)N;
}


// ── Pair-free kernel: compute (tgt, src) from linear pair_id ─────────────────
//
// Eliminates the need for an explicit pairs tensor.
// pair_id = tgt * (n_vars - 1) + src_local
// src = src_local + (src_local >= tgt ? 1 : 0)
//
template<typename BinType>
__global__ void te_adaptive_smem_kernel_pairfree(
    const BinType*  __restrict__ bin_arrs,    // (n_vars * T * K,) row-major
    const int32_t* __restrict__ n_bins_arr,  // (n_vars,) int32
    float*         __restrict__ te_ptr,       // (n_pairs_local,) float32 output
    int T, int K, int tau, int L,
    int n_vars,
    int64_t pair_offset
) {
    extern __shared__ char smem_buf[];

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

    int b_t  = n_bins_arr[target_var];
    int b_s  = n_bins_arr[source_var];
    int b_t2 = b_t * b_t;
    int b_t2_s = b_t2 * b_s;

    int32_t* cnt3    = (int32_t*)smem_buf;
    float*   cnt2a   = (float*)(cnt3 + b_t2_s);
    float*   cnt2b   = cnt2a + b_t2;
    float*   cnt1_sm = cnt2b + b_t * b_s;
    float*   warp_buf = cnt1_sm + b_t;

    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // ── Phase 0: clear cnt3
    for (int b = tid; b < b_t2_s; b += BLOCK) cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = (int)__ldg(&target_base[tau + i]);
            int xtv  = (int)__ldg(&target_base[i]);
            int ytv  = (int)__ldg(&source_base[i]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = (int)__ldg(&target_base[(tau + t) * K + c]);
            int xtv  = (int)__ldg(&target_base[t        * K + c]);
            int ytv  = (int)__ldg(&source_base[t        * K + c]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: marginals
    for (int j = tid; j < b_t; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            for (int k = 0; k < b_s; k++)
                s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt1_sm[j] = s;
    }
    for (int t = tid; t < b_t2; t += BLOCK) {
        int i = t / b_t, j = t % b_t;
        float s = 0.0f;
        for (int k = 0; k < b_s; k++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2a[t] = s;
    }
    for (int t = tid; t < b_t * b_s; t += BLOCK) {
        int j = t / b_s, k = t % b_s;
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula
    float te_local = 0.0f;
    for (int b = tid; b < b_t2_s; b += BLOCK) {
        int i  = b / (b_t * b_s);
        int jk = b % (b_t * b_s);
        int j  = jk / b_s;
        int k  = jk % b_s;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * b_t + j];
        float c2b = cnt2b[j * b_s + k];
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


// ── Opt-in shared memory query ─────────────────────────────────────────────────
int64_t get_smem_optin() {
    int dev = 0, val = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)val;
}


// ── Python-facing launcher ─────────────────────────────────────────────────────
// Dispatch by bin_arrs dtype: int8 (4× memory savings), int16 (2×), or int32.
template<typename BinType>
void _launch_kernel(
    torch::Tensor& bin_arrs, torch::Tensor& pairs, torch::Tensor& n_bins_arr,
    torch::Tensor& te, int T, int K, int tau, int L, int block_size, int smem_bytes
) {
    cudaFuncSetAttribute(
        te_adaptive_smem_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    int n_pairs = pairs.size(0);
    te_adaptive_smem_kernel<BinType><<<n_pairs, block_size, smem_bytes>>>(
        bin_arrs.data_ptr<BinType>(),
        pairs.data_ptr<int64_t>(),
        n_bins_arr.data_ptr<int32_t>(),
        te.data_ptr<float>(),
        T, K, tau, L
    );
}

torch::Tensor te_adaptive_smem_launch(
    torch::Tensor bin_arrs,    // (n_vars, T, K) int8/int16/int32 CUDA
    torch::Tensor pairs,       // (n_pairs, 2)    int64 CUDA
    torch::Tensor n_bins_arr,  // (n_vars,)      int32 CUDA
    int T, int K, int tau,
    int block_size,
    int smem_bytes             // max SMEM for this batch (pre-computed Python side)
) {
    int n_pairs = pairs.size(0);
    auto te = torch::empty({n_pairs}, pairs.options().dtype(torch::kFloat32));
    int L = T - tau;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_kernel<int8_t>(bin_arrs, pairs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes);
    } else if (dtype == torch::kInt16) {
        _launch_kernel<int16_t>(bin_arrs, pairs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes);
    } else {
        _launch_kernel<int32_t>(bin_arrs, pairs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes);
    }
    return te;
}

// ── Pair-free launcher ──────────────────────────────────────────────────────
template<typename BinType>
void _launch_kernel_pairfree(
    torch::Tensor& bin_arrs, torch::Tensor& n_bins_arr, torch::Tensor& te,
    int T, int K, int tau, int L, int block_size, int smem_bytes,
    int n_vars, int64_t pair_offset, int64_t n_pairs_local
) {
    cudaFuncSetAttribute(
        te_adaptive_smem_kernel_pairfree<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    te_adaptive_smem_kernel_pairfree<BinType><<<(int)n_pairs_local, block_size, smem_bytes>>>(
        bin_arrs.data_ptr<BinType>(),
        n_bins_arr.data_ptr<int32_t>(),
        te.data_ptr<float>(),
        T, K, tau, L,
        n_vars, pair_offset
    );
}

torch::Tensor te_adaptive_smem_launch_pairfree(
    torch::Tensor bin_arrs,    // (n_vars, T, K) int8/int16/int32 CUDA
    torch::Tensor n_bins_arr,  // (n_vars,) int32 CUDA
    int T, int K, int tau,
    int block_size,
    int smem_bytes,
    int n_vars,
    int64_t pair_offset,
    int64_t n_pairs_local
) {
    TORCH_CHECK(n_pairs_local <= (int64_t)INT_MAX,
        "n_pairs_local (", n_pairs_local, ") exceeds INT_MAX grid limit");
    auto te = torch::empty({n_pairs_local},
                           bin_arrs.options().dtype(torch::kFloat32));
    int L = T - tau;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_kernel_pairfree<int8_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_vars, pair_offset, n_pairs_local);
    } else if (dtype == torch::kInt16) {
        _launch_kernel_pairfree<int16_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_vars, pair_offset, n_pairs_local);
    } else {
        _launch_kernel_pairfree<int32_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_vars, pair_offset, n_pairs_local);
    }
    return te;
}

// ── Cross pair-free kernel: TE between two disjoint variable blocks ───────────────
//
// Grid : (n_pairs_local,) — one block per (target, source) pair
// bin_arrs layout: [tgt_genes (n_tgt rows) | src_genes (n_src rows)]
// pair_id = tgt_local * n_src + src_local
// target reads from row tgt_local, source reads from row (n_tgt + src_local)
//
template<typename BinType>
__global__ void te_adaptive_smem_kernel_cross(
    const BinType*  __restrict__ bin_arrs,    // ((n_tgt + n_src) * T * K,) row-major
    const int32_t* __restrict__ n_bins_arr,  // (n_tgt + n_src,) int32
    float*         __restrict__ te_ptr,       // (n_pairs_local,) float32 output
    int T, int K, int tau, int L,
    int n_tgt, int n_src,
    int64_t pair_offset
) {
    extern __shared__ char smem_buf[];

    const int local_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int BLOCK    = blockDim.x;
    const int WARPS    = BLOCK / 32;
    const int N        = L * K;
    const int TK       = T * K;

    // Compute (target, source) from linear index within this block combination
    const int64_t global_pair_id = (int64_t)local_id + pair_offset;
    int target_local = (int)(global_pair_id / n_src);
    int source_local = (int)(global_pair_id % n_src);
    // target is in [0, n_tgt), source is in [n_tgt, n_tgt + n_src)
    int target_var = target_local;
    int source_var = n_tgt + source_local;

    int b_t  = n_bins_arr[target_var];
    int b_s  = n_bins_arr[source_var];
    int b_t2 = b_t * b_t;
    int b_t2_s = b_t2 * b_s;

    int32_t* cnt3    = (int32_t*)smem_buf;
    float*   cnt2a   = (float*)(cnt3 + b_t2_s);
    float*   cnt2b   = cnt2a + b_t2;
    float*   cnt1_sm = cnt2b + b_t * b_s;
    float*   warp_buf = cnt1_sm + b_t;

    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // ── Phase 0: clear cnt3
    for (int b = tid; b < b_t2_s; b += BLOCK) cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = (int)__ldg(&target_base[tau + i]);
            int xtv  = (int)__ldg(&target_base[i]);
            int ytv  = (int)__ldg(&source_base[i]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = (int)__ldg(&target_base[(tau + t) * K + c]);
            int xtv  = (int)__ldg(&target_base[t        * K + c]);
            int ytv  = (int)__ldg(&source_base[t        * K + c]);
            atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
        }
    }
    __syncthreads();

    // ── Phase 2: marginals
    for (int j = tid; j < b_t; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            for (int k = 0; k < b_s; k++)
                s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt1_sm[j] = s;
    }
    for (int t = tid; t < b_t2; t += BLOCK) {
        int i = t / b_t, j = t % b_t;
        float s = 0.0f;
        for (int k = 0; k < b_s; k++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2a[t] = s;
    }
    for (int t = tid; t < b_t * b_s; t += BLOCK) {
        int j = t / b_s, k = t % b_s;
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            s += (float)cnt3[i * b_t * b_s + j * b_s + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula
    float te_local = 0.0f;
    for (int b = tid; b < b_t2_s; b += BLOCK) {
        int i  = b / (b_t * b_s);
        int jk = b % (b_t * b_s);
        int j  = jk / b_s;
        int k  = jk % b_s;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * b_t + j];
        float c2b = cnt2b[j * b_s + k];
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


// ── Cross pair-free launcher ──────────────────────────────────────────────
template<typename BinType>
void _launch_kernel_cross(
    torch::Tensor& bin_arrs, torch::Tensor& n_bins_arr, torch::Tensor& te,
    int T, int K, int tau, int L, int block_size, int smem_bytes,
    int n_tgt, int n_src, int64_t pair_offset, int64_t n_pairs_local
) {
    cudaFuncSetAttribute(
        te_adaptive_smem_kernel_cross<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );
    te_adaptive_smem_kernel_cross<BinType><<<(int)n_pairs_local, block_size, smem_bytes>>>(
        bin_arrs.data_ptr<BinType>(),
        n_bins_arr.data_ptr<int32_t>(),
        te.data_ptr<float>(),
        T, K, tau, L,
        n_tgt, n_src, pair_offset
    );
}

torch::Tensor te_adaptive_smem_launch_cross(
    torch::Tensor bin_arrs,    // ((n_tgt + n_src) * T * K,) int8/int16/int32 CUDA
    torch::Tensor n_bins_arr,  // (n_tgt + n_src,) int32 CUDA
    int T, int K, int tau,
    int block_size,
    int smem_bytes,
    int n_tgt, int n_src,
    int64_t pair_offset,
    int64_t n_pairs_local
) {
    TORCH_CHECK(n_pairs_local <= (int64_t)INT_MAX,
        "n_pairs_local (", n_pairs_local, ") exceeds INT_MAX grid limit");
    auto te = torch::empty({n_pairs_local},
                           bin_arrs.options().dtype(torch::kFloat32));
    int L = T - tau;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_kernel_cross<int8_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_tgt, n_src, pair_offset, n_pairs_local);
    } else if (dtype == torch::kInt16) {
        _launch_kernel_cross<int16_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_tgt, n_src, pair_offset, n_pairs_local);
    } else {
        _launch_kernel_cross<int32_t>(bin_arrs, n_bins_arr, te, T, K, tau, L, block_size, smem_bytes, n_tgt, n_src, pair_offset, n_pairs_local);
    }
    return te;
}

// ── In-place bin coarsening kernel ───────────────────────────────────────────
// Zero extra memory: reads and writes bin_arr in-place.
// new_bin = old_bin * limit / n_per_var[var]
// Skips variables where n_per_var[var] <= limit (no coarsening needed).
// Grid: (n_vars,), Block: (256,)
__global__ void coarsen_bins_kernel(
    int8_t* __restrict__ bin_arr,        // (n_vars, TK) flattened
    const int32_t* __restrict__ n_per_var, // (n_vars,)
    int TK,                               // T * K
    int limit                             // smem_bin_limit
) {
    const int var = blockIdx.x;
    const int ng = n_per_var[var];
    if (ng <= limit) return;

    int8_t* row = bin_arr + (int64_t)var * (int64_t)TK;
    for (int i = threadIdx.x; i < TK; i += blockDim.x) {
        int old_bin = (int)row[i];
        row[i] = (int8_t)((old_bin * limit) / ng);
    }
}

void coarsen_bins_cuda(
    torch::Tensor bin_arr,       // (n_vars, T, K) int8 CUDA, modified in-place
    torch::Tensor n_per_var,    // (n_vars,) int32 CUDA
    int smem_bin_limit
) {
    TORCH_CHECK(bin_arr.scalar_type() == torch::kInt8,
        "coarsen_bins_cuda requires int8 bin_arr");
    TORCH_CHECK(bin_arr.is_cuda(), "bin_arr must be on CUDA");

    int n_vars = bin_arr.size(0);
    int TK = 1;
    for (int d = 1; d < bin_arr.dim(); d++) TK *= bin_arr.size(d);

    coarsen_bins_kernel<<<n_vars, 256>>>(
        bin_arr.data_ptr<int8_t>(),
        n_per_var.data_ptr<int32_t>(),
        TK, smem_bin_limit
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("te_adaptive_smem_launch", &te_adaptive_smem_launch,
          "Adaptive-SMEM TE kernel: per-var asymmetric B (n_i^2 * n_j SMEM)");
    m.def("te_adaptive_smem_launch_pairfree", &te_adaptive_smem_launch_pairfree,
          "Adaptive-SMEM TE kernel — pair-free mode (no explicit pairs tensor)");
    m.def("te_adaptive_smem_launch_cross", &te_adaptive_smem_launch_cross,
          "Adaptive-SMEM TE kernel — cross pair-free mode (two disjoint variable blocks)");
    m.def("get_smem_optin", &get_smem_optin,
          "Max configurable dynamic shared memory per block for current device");
    m.def("coarsen_bins_cuda", &coarsen_bins_cuda,
          "In-place bin coarsening (zero extra memory)");
}
