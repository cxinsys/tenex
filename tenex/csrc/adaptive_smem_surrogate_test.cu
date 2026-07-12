#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level reduction ──────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum_adaptive_surr(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── Fused Adaptive-SMEM surrogate-test kernel ────────────────────────────────
//
// Grid : (n_pairs_local,)   — one block per (target, source) variable pair.
// Block: (BLOCK,)            — same sizing as te_adaptive_smem_kernel_pairfree.
//
// For each pair (target_var, source_var) this block iterates over
// ``n_surrogates`` block-shuffled copies of the target and source variables
// (mirroring ``_block_shuffle_gpu`` / ``_random_shuffle_gpu``), computes the
// TE formula fully inside shared memory, and accumulates
//   sum_te   [i, j] = Σ_k te_k
//   sum_sq_te[i, j] = Σ_k te_k²
//   count_ge [i, j] = #{k : te_k >= te_obs}
// entirely on the GPU. Outputs are pre-zeroed by the caller; because one
// block uniquely owns each (i, j) slot no atomics are required for the final
// write. This mirrors ``full_smem_surrogate_test_kernel`` but uses the
// per-pair asymmetric bin layout from ``te_adaptive_smem_kernel_pairfree``.
//
// Shared memory layout (dynamic, sizes in bytes; b_t = n_per_var[target_var],
// b_s = n_per_var[source_var]):
//   [0                         .. b_t2_s*4)                   : int32 cnt3[b_t*b_t*b_s]
//   [b_t2_s*4                    .. b_t2_s*4 + b_t2*4)           : float cnt2a[b_t*b_t]
//   [b_t2_s*4 + b_t2*4            .. b_t2_s*4 + b_t2*4 + b_tb_s*4)  : float cnt2b[b_t*b_s]
//   [...                       .. + b_t*4)                    : float cnt1_sm[b_t]
//   [...                       .. + WARPS*4)                 : float warp_buf[WARPS]
//
template<typename BinType>
__global__ void adaptive_smem_surrogate_test_kernel(
    const BinType*  __restrict__ bin_arrs,      // (n_vars * T * K,) row-major
    const int32_t*  __restrict__ n_per_var,     // (n_vars,) int32
    const int32_t*  __restrict__ block_perm,    // (n_surrogates, n_vars, n_blocks)
    const float*    __restrict__ observed_te,   // (n_vars * n_vars,) row-major
    float*          __restrict__ sum_te,        // (n_vars * n_vars,) inout
    float*          __restrict__ sum_sq_te,     // (n_vars * n_vars,) inout
    int32_t*        __restrict__ count_ge,      // (n_vars * n_vars,) inout
    int T, int K, int tau, int L,
    int block_length, int n_blocks,
    int n_surrogates, int n_vars,
    int64_t pair_offset
) {
    extern __shared__ char smem_buf[];

    const int local_id = blockIdx.x;
    const int tid      = threadIdx.x;
    const int BLOCK    = blockDim.x;
    const int WARPS    = BLOCK / 32;
    const int N        = L * K;
    const int TK       = T * K;

    // Map linear pair id → (target_var, source_var) (same scheme as pair-free).
    const int64_t global_pair_id = (int64_t)local_id + pair_offset;
    const int nm1 = n_vars - 1;
    int target_var = (int)(global_pair_id / nm1);
    int source_var = (int)(global_pair_id % nm1);
    if (source_var >= target_var) source_var++;

    // Per-pair bin counts (read from n_per_var)
    int b_t   = n_per_var[target_var];
    int b_s   = n_per_var[source_var];
    int b_t2  = b_t * b_t;
    int b_t2_s = b_t2 * b_s;
    int b_tb_s = b_t * b_s;

    // SMEM pointers — offsets depend on this pair's b_t, b_s
    int32_t* cnt3     = (int32_t*)smem_buf;
    float*   cnt2a    = (float*)(cnt3 + b_t2_s);
    float*   cnt2b    = cnt2a + b_t2;
    float*   cnt1_sm  = cnt2b + b_tb_s;
    float*   warp_buf = cnt1_sm + b_t;

    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // Observed TE value for this pair (read once). Matrix convention:
    //   M[i, j] = TE(i -> j)   → source along rows, target along columns.
    // So this pair's slot is [source_var, target_var].
    const int64_t pair_slot = (int64_t)source_var * n_vars + target_var;
    float te_obs = observed_te[pair_slot];

    // Cover length of the "head" region the permutation touches.
    const int head_len = n_blocks * block_length;

    // Thread-0 registers: surrogate accumulators.
    float   loc_sum    = 0.0f;
    float   loc_sum_sq = 0.0f;
    int32_t loc_count  = 0;

    for (int k = 0; k < n_surrogates; ++k) {
        // Per-surrogate base pointers into block_perm: each variable owns its
        // own block permutation (identical semantics to the existing
        // _block_shuffle_gpu / _random_shuffle_gpu Python paths).
        //   block_perm[k, target_var, :]  — drives ``xt1v`` / ``xtv``
        //   block_perm[k, source_var, :]  — drives ``ytv``
        const int32_t* perm_tgt = block_perm
            + ((int64_t)k * n_vars + (int64_t)target_var) * n_blocks;
        const int32_t* perm_src = block_perm
            + ((int64_t)k * n_vars + (int64_t)source_var) * n_blocks;

        // ── Phase 0: clear cnt3 ───────────────────────────────────────────
        for (int b = tid; b < b_t2_s; b += BLOCK)
            cnt3[b] = 0;
        __syncthreads();

        // ── Phase 1: build histogram with PERMUTED target + source reads ──
        if (K == 1) {
            for (int i = tid; i < L; i += BLOCK) {
                // tau + i indirection (target "next" sample). tau + i may fall
                // into the head or the tail depending on block alignment.
                int xt1_t;
                int it = tau + i;
                if (it < head_len) {
                    int bi = it / block_length;
                    int of = it - bi * block_length;
                    xt1_t = perm_tgt[bi] * block_length + of;
                } else {
                    xt1_t = it;
                }

                int xt_t, y_t;
                if (i < head_len) {
                    int bi = i / block_length;
                    int of = i - bi * block_length;
                    xt_t = perm_tgt[bi] * block_length + of;
                    y_t  = perm_src[bi] * block_length + of;
                } else {
                    xt_t = i;
                    y_t  = i;
                }

                int xt1v = (int)__ldg(&target_base[xt1_t]);
                int xtv  = (int)__ldg(&target_base[xt_t]);
                int ytv  = (int)__ldg(&source_base[y_t]);
                atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
            }
        } else {
            for (int i = tid; i < N; i += BLOCK) {
                int t = i / K, c = i % K;

                int xt1_t;
                int it = tau + t;
                if (it < head_len) {
                    int bi = it / block_length;
                    int of = it - bi * block_length;
                    xt1_t = perm_tgt[bi] * block_length + of;
                } else {
                    xt1_t = it;
                }

                int xt_t, y_t;
                if (t < head_len) {
                    int bi = t / block_length;
                    int of = t - bi * block_length;
                    xt_t = perm_tgt[bi] * block_length + of;
                    y_t  = perm_src[bi] * block_length + of;
                } else {
                    xt_t = t;
                    y_t  = t;
                }

                int xt1v = (int)__ldg(&target_base[xt1_t * K + c]);
                int xtv  = (int)__ldg(&target_base[xt_t  * K + c]);
                int ytv  = (int)__ldg(&source_base[y_t   * K + c]);
                atomicAdd(&cnt3[xt1v * b_t * b_s + xtv * b_s + ytv], 1);
            }
        }
        __syncthreads();

        // ── Phase 2: marginals — copy verbatim from pair-free Adaptive-SMEM
        // cnt1[j] = Σ_{i,k} cnt3[i, j, k]
        for (int j = tid; j < b_t; j += BLOCK) {
            float s = 0.0f;
            for (int i = 0; i < b_t; i++)
                for (int kk = 0; kk < b_s; kk++)
                    s += (float)cnt3[i * b_t * b_s + j * b_s + kk];
            cnt1_sm[j] = s;
        }
        // cnt2a[i, j] = Σ_k cnt3[i, j, k]
        for (int t = tid; t < b_t2; t += BLOCK) {
            int i = t / b_t, j = t % b_t;
            float s = 0.0f;
            for (int kk = 0; kk < b_s; kk++)
                s += (float)cnt3[i * b_t * b_s + j * b_s + kk];
            cnt2a[t] = s;
        }
        // cnt2b[j, k] = Σ_i cnt3[i, j, k]
        for (int t = tid; t < b_tb_s; t += BLOCK) {
            int j = t / b_s, kk = t % b_s;
            float s = 0.0f;
            for (int i = 0; i < b_t; i++)
                s += (float)cnt3[i * b_t * b_s + j * b_s + kk];
            cnt2b[t] = s;
        }
        __syncthreads();

        // ── Phase 3: TE formula — copy verbatim
        float te_local = 0.0f;
        for (int b = tid; b < b_t2_s; b += BLOCK) {
            int i  = b / (b_t * b_s);
            int jk = b % (b_t * b_s);
            int j  = jk / b_s;
            int kk = jk % b_s;

            float c3  = (float)cnt3[b];
            float c2a = cnt2a[i * b_t + j];
            float c2b = cnt2b[j * b_s + kk];
            float c1  = cnt1_sm[j];
            float denom = c2a * c2b;

            if (c3 > 0.0f && denom > 0.0f)
                te_local += c3 * log2f(c3 * c1 / denom);
        }

        // ── Phase 4: warp + block reduction
        te_local = warp_reduce_sum_adaptive_surr(te_local);
        if (tid % 32 == 0)
            warp_buf[tid / 32] = te_local;
        __syncthreads();

        te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
        te_local = warp_reduce_sum_adaptive_surr(te_local);

        // Thread 0 updates accumulators (same normalization as base kernel:
        // divide by N = L * K).
        if (tid == 0) {
            float te_k = te_local / (float)N;
            loc_sum    += te_k;
            loc_sum_sq += te_k * te_k;
            if (te_k >= te_obs) loc_count += 1;
        }

        // Ensure all threads complete Phase 4 before the next iteration
        // overwrites cnt3/cnt2a/cnt2b/cnt1_sm/warp_buf.
        __syncthreads();
    }

    if (tid == 0) {
        sum_te   [pair_slot] = loc_sum;
        sum_sq_te[pair_slot] = loc_sum_sq;
        count_ge [pair_slot] = loc_count;
    }
}


// ── Opt-in shared memory query ────────────────────────────────────────────────
int64_t get_smem_optin() {
    int dev = 0, val = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)val;
}

// ── Dtype dispatch helper ─────────────────────────────────────────────────────
template<typename BinType>
void _launch_adaptive_smem_surrogate_test(
    torch::Tensor& bin_arrs,
    torch::Tensor& n_per_var,
    torch::Tensor& block_perm,
    torch::Tensor& observed_te,
    torch::Tensor& sum_te,
    torch::Tensor& sum_sq_te,
    torch::Tensor& count_ge,
    int T, int K, int tau, int L,
    int block_length, int n_blocks,
    int n_surrogates, int n_vars,
    int64_t pair_offset, int64_t n_pairs_local,
    int block_size, size_t smem
) {
    cudaError_t attr_err = cudaFuncSetAttribute(
        adaptive_smem_surrogate_test_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    TORCH_CHECK(attr_err == cudaSuccess,
        "cudaFuncSetAttribute for adaptive_smem_surrogate_test_kernel failed: ",
        cudaGetErrorString(attr_err));
    adaptive_smem_surrogate_test_kernel<BinType><<<(int)n_pairs_local, block_size, smem>>>(
        bin_arrs.data_ptr<BinType>(),
        n_per_var.data_ptr<int32_t>(),
        block_perm.data_ptr<int32_t>(),
        observed_te.data_ptr<float>(),
        sum_te.data_ptr<float>(),
        sum_sq_te.data_ptr<float>(),
        count_ge.data_ptr<int32_t>(),
        T, K, tau, L,
        block_length, n_blocks,
        n_surrogates, n_vars, pair_offset
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ── Python-facing launcher ────────────────────────────────────────────────────
void adaptive_smem_surrogate_test_launch(
    torch::Tensor bin_arrs,      // (n_vars, T, K) int{8,16,32} CUDA
    torch::Tensor n_per_var,     // (n_vars,) int32 CUDA
    torch::Tensor block_perm,    // (n_surrogates, n_vars, n_blocks) int32 CUDA
    torch::Tensor observed_te,   // (n_vars, n_vars) float32 CUDA
    torch::Tensor sum_te,        // (n_vars, n_vars) float32 CUDA, pre-zeroed
    torch::Tensor sum_sq_te,     // (n_vars, n_vars) float32 CUDA, pre-zeroed
    torch::Tensor count_ge,      // (n_vars, n_vars) int32 CUDA, pre-zeroed
    int T, int K, int tau,
    int block_length, int n_surrogates,
    int block_size,
    int smem_bytes,              // max SMEM for this pair range (pre-computed Python side)
    int n_vars,
    int64_t pair_offset,
    int64_t n_pairs_local
) {
    TORCH_CHECK(n_pairs_local <= (int64_t)INT_MAX,
        "n_pairs_local (", n_pairs_local, ") exceeds INT_MAX grid limit; "
        "use multi-GPU or call with smaller chunks");
    // Device & contiguity
    TORCH_CHECK(bin_arrs.is_cuda() && n_per_var.is_cuda() &&
                block_perm.is_cuda() && observed_te.is_cuda() &&
                sum_te.is_cuda() && sum_sq_te.is_cuda() && count_ge.is_cuda(),
        "all tensors must be on CUDA");
    auto dev = bin_arrs.device();
    TORCH_CHECK(n_per_var.device() == dev && block_perm.device() == dev &&
                observed_te.device() == dev && sum_te.device() == dev &&
                sum_sq_te.device() == dev && count_ge.device() == dev,
        "all tensors must be on the same CUDA device as bin_arrs");
    TORCH_CHECK(bin_arrs.is_contiguous() && n_per_var.is_contiguous() &&
                block_perm.is_contiguous() && observed_te.is_contiguous() &&
                sum_te.is_contiguous() && sum_sq_te.is_contiguous() &&
                count_ge.is_contiguous(),
        "all tensors must be contiguous");
    // Rank & shape
    TORCH_CHECK(bin_arrs.dim() == 3,
        "bin_arrs must have shape (n_vars, T, K)");
    TORCH_CHECK(n_per_var.dim() == 1 && n_per_var.size(0) == n_vars,
        "n_per_var must have shape (n_vars,)");
    TORCH_CHECK(block_perm.dim() == 3,
        "block_perm must have shape (n_surrogates, n_vars, n_blocks)");
    TORCH_CHECK(observed_te.dim() == 2 && observed_te.size(0) == n_vars &&
                observed_te.size(1) == n_vars,
        "observed_te must have shape (n_vars, n_vars)");
    TORCH_CHECK(sum_te.dim() == 2 && sum_te.size(0) == n_vars &&
                sum_te.size(1) == n_vars,
        "sum_te must have shape (n_vars, n_vars)");
    TORCH_CHECK(sum_sq_te.dim() == 2 && sum_sq_te.size(0) == n_vars &&
                sum_sq_te.size(1) == n_vars,
        "sum_sq_te must have shape (n_vars, n_vars)");
    TORCH_CHECK(count_ge.dim() == 2 && count_ge.size(0) == n_vars &&
                count_ge.size(1) == n_vars,
        "count_ge must have shape (n_vars, n_vars)");
    // Dtypes
    TORCH_CHECK(n_per_var.scalar_type() == torch::kInt32,
        "n_per_var must be int32");
    TORCH_CHECK(block_perm.scalar_type() == torch::kInt32,
        "block_perm must be int32");
    TORCH_CHECK(observed_te.scalar_type() == torch::kFloat32,
        "observed_te must be float32");
    TORCH_CHECK(sum_te.scalar_type() == torch::kFloat32,
        "sum_te must be float32");
    TORCH_CHECK(sum_sq_te.scalar_type() == torch::kFloat32,
        "sum_sq_te must be float32");
    TORCH_CHECK(count_ge.scalar_type() == torch::kInt32,
        "count_ge must be int32");

    const int n_blocks = (int)block_perm.size(2);
    int L     = T - tau;
    size_t smem = (size_t)smem_bytes;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_adaptive_smem_surrogate_test<int8_t>(
            bin_arrs, n_per_var, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    } else if (dtype == torch::kInt16) {
        _launch_adaptive_smem_surrogate_test<int16_t>(
            bin_arrs, n_per_var, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    } else {
        _launch_adaptive_smem_surrogate_test<int32_t>(
            bin_arrs, n_per_var, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_smem_surrogate_test_launch", &adaptive_smem_surrogate_test_launch,
          "Fused Adaptive-SMEM surrogate-test kernel (per-pair accumulators on GPU)");
    m.def("get_smem_optin", &get_smem_optin,
          "Max configurable dynamic shared memory per block (opt-in) for current device");
}
