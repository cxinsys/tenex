#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level sum reduction ─────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── Global-memory kernel ─────────────────────────────────────────────────────
//
// bin_arrs stored as uint8 (bins 0..B-1, B <= 127 guaranteed by remap_bins).
// Each block handles exactly one overflow pair.
// cnt3 in global memory (per-pair region via offset table); only marginals in SMEM.
//
// SMEM layout (dynamic, per-block):
//   float cnt2a[b_t*b_t]   at offset 0
//   float cnt2b[b_t*b_s]   after cnt2a
//   float cnt1[b_t]        after cnt2b
//   float warp_buf[WARPS] after cnt1
//
__global__ void te_gmem_kernel(
    const uint8_t* __restrict__ bin_arrs_u8, // (n_vars, T*K) uint8, row-major
    const int64_t* __restrict__ pairs,         // (n_overflow, 2) int64
    const int32_t* __restrict__ n_bins_arr,   // (n_vars,) int32
    const int64_t* __restrict__ cnt3_offsets,  // (n_overflow+1,) int64 prefix sum
    int32_t*       __restrict__ cnt3_gmem,     // (total_cnt3_cells,) int32, pre-zeroed
    float*         __restrict__ te_out,         // (n_overflow,) float32
    int T, int K, int tau, int L, int N
) {
    extern __shared__ char smem_buf[];

    const int pair_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;
    const int TK      = T * K;

    const int target_var = (int)pairs[pair_id * 2    ];
    const int source_var = (int)pairs[pair_id * 2 + 1];
    const int b_t   = n_bins_arr[target_var];
    const int b_s   = n_bins_arr[source_var];
    const int b_t2  = b_t * b_t;
    const int b_t2_s = b_t2 * b_s;
    const int b_tb_s = b_t * b_s;

    // SMEM pointers (marginals only, cnt3 is in global memory)
    float* cnt2a    = (float*)smem_buf;
    float* cnt2b    = cnt2a + b_t2;
    float* cnt1_sm  = cnt2b + b_tb_s;
    float* warp_buf = cnt1_sm + b_t;

    // Global-memory cnt3 region for this pair (pre-zeroed by caller)
    int32_t* cnt3 = cnt3_gmem + cnt3_offsets[pair_id];

    // Per-pair bin_arrs base pointers (uint8)
    const uint8_t* target_base = bin_arrs_u8 + (int64_t)target_var * TK;
    const uint8_t* source_base = bin_arrs_u8 + (int64_t)source_var * TK;

    // ── Phase 0: clear SMEM marginals ─────────────────────────────────────────
    for (int b = tid; b < b_t2;  b += BLOCK) cnt2a[b]   = 0.0f;
    for (int b = tid; b < b_tb_s; b += BLOCK) cnt2b[b]   = 0.0f;
    for (int b = tid; b < b_t;   b += BLOCK) cnt1_sm[b] = 0.0f;
    __syncthreads();

    // ── Phase 1: build cnt3 in global memory via atomicAdd ──────────────────
    // Each block operates on its own cnt3 region → no inter-block conflicts.
    // L2 cache (~6 MB) can hold per-pair cnt3 (40-250 KB) for nearby pairs.
    if (K == 1) {
        for (int i = tid; i < L; i += BLOCK) {
            int xt1v = (int)__ldg(target_base + tau + i);
            int xtv  = (int)__ldg(target_base + i);
            int ytv  = (int)__ldg(source_base + i);
            atomicAdd(&cnt3[xt1v * b_tb_s + xtv * b_s + ytv], 1);
        }
    } else {
        for (int i = tid; i < N; i += BLOCK) {
            int t = i / K, c = i % K;
            int xt1v = (int)__ldg(target_base + (tau + t) * K + c);
            int xtv  = (int)__ldg(target_base + t * K + c);
            int ytv  = (int)__ldg(source_base + t * K + c);
            atomicAdd(&cnt3[xt1v * b_tb_s + xtv * b_s + ytv], 1);
        }
    }
    // __syncthreads() ensures all atomicAdds within this block are visible
    // to subsequent reads by threads in this same block.
    __syncthreads();

    // ── Phase 2: compute marginals from global-memory cnt3 ──────────────────
    // Each thread computes a subset of marginal cells.

    // cnt1[j] = sum_i sum_k cnt3[i,j,k]
    for (int j = tid; j < b_t; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            for (int k = 0; k < b_s; k++)
                s += (float)cnt3[i * b_tb_s + j * b_s + k];
        cnt1_sm[j] = s;
    }

    // cnt2a[i,j] = sum_k cnt3[i,j,k]
    for (int t = tid; t < b_t2; t += BLOCK) {
        int i = t / b_t, j = t % b_t;
        float s = 0.0f;
        for (int k = 0; k < b_s; k++)
            s += (float)cnt3[i * b_tb_s + j * b_s + k];
        cnt2a[t] = s;
    }

    // cnt2b[j,k] = sum_i cnt3[i,j,k]
    for (int t = tid; t < b_tb_s; t += BLOCK) {
        int j = t / b_s, k = t % b_s;
        float s = 0.0f;
        for (int i = 0; i < b_t; i++)
            s += (float)cnt3[i * b_tb_s + j * b_s + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula ───────────────────────────────────────────────────
    // Re-read cnt3 from global memory (sequential access → L2 friendly);
    // marginals from SMEM (fast).
    float te_local = 0.0f;
    for (int b = tid; b < b_t2_s; b += BLOCK) {
        int i  = b / b_tb_s;
        int jk = b % b_tb_s;
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

    // ── Phase 4: block reduction ──────────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);
    if (tid == 0)
        te_out[pair_id] = te_local / (float)N;
}


// ── Query SMEM opt-in limit ───────────────────────────────────────────────────
int64_t get_smem_optin() {
    int dev = 0, val = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)val;
}


// ── Compute per-pair cnt3 prefix-sum offsets (GPU) ────────────────────────────
torch::Tensor compute_cnt3_offsets(
    torch::Tensor pairs,        // (n_overflow, 2) int64 CUDA
    torch::Tensor n_bins_arr    // (n_vars,) int32 CUDA
) {
    int n = (int)pairs.size(0);
    if (n == 0) {
        return torch::zeros({1}, pairs.options().dtype(torch::kInt64));
    }
    // sizes[i] = b_t[i]^2 * b_s[i]
    auto target_bins = n_bins_arr.index({pairs.select(1, 0)}).to(torch::kInt64);
    auto source_bins = n_bins_arr.index({pairs.select(1, 1)}).to(torch::kInt64);
    auto sizes = target_bins * target_bins * source_bins;  // (n,)

    // Build prefix sum: offsets[0]=0, offsets[i+1] = offsets[i] + sizes[i]
    auto offsets = torch::empty({n + 1}, pairs.options().dtype(torch::kInt64));
    offsets[0] = 0;
    offsets.slice(0, 1, n + 1).copy_(torch::cumsum(sizes, 0));
    return offsets;
}


// ── Python-facing launcher ─────────────────────────────────────────────────────
torch::Tensor te_gmem_launch(
    torch::Tensor bin_arrs_u8,   // (n_vars, T*K) uint8 CUDA
    torch::Tensor pairs,          // (n_overflow, 2) int64 CUDA
    torch::Tensor n_bins_arr,    // (n_vars,) int32 CUDA
    torch::Tensor cnt3_offsets,  // (n_overflow+1,) int64 CUDA
    int64_t total_cnt3,           // total cnt3 cells (= cnt3_offsets[-1])
    int T, int K, int tau,
    int block_size,
    int smem_bytes                // pre-computed SMEM size for marginals
) {
    int n_overflow = (int)pairs.size(0);
    if (n_overflow == 0) {
        return torch::empty({0}, pairs.options().dtype(torch::kFloat32));
    }

    int L = T - tau;
    int N = L * K;

    // Allocate and zero global-memory cnt3 buffer
    auto cnt3_gmem = torch::zeros({total_cnt3},
                                  pairs.options().dtype(torch::kInt32));
    auto te = torch::empty({n_overflow},
                            pairs.options().dtype(torch::kFloat32));

    cudaFuncSetAttribute(
        te_gmem_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes
    );

    te_gmem_kernel<<<n_overflow, block_size, smem_bytes>>>(
        bin_arrs_u8.data_ptr<uint8_t>(),
        pairs.data_ptr<int64_t>(),
        n_bins_arr.data_ptr<int32_t>(),
        cnt3_offsets.data_ptr<int64_t>(),
        cnt3_gmem.data_ptr<int32_t>(),
        te.data_ptr<float>(),
        T, K, tau, L, N
    );
    return te;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("te_gmem_launch", &te_gmem_launch,
          "Global-memory kernel: cnt3 in device DRAM (per-pair offset table), SMEM marginals, uint8 bin_arrs");
    m.def("compute_cnt3_offsets", &compute_cnt3_offsets,
          "Compute prefix-sum offsets for per-pair global-memory cnt3 allocation");
    m.def("get_smem_optin", &get_smem_optin,
          "Max configurable dynamic SMEM per block for current device");
}
