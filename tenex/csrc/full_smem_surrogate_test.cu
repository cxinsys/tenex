#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level reduction ──────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum_surr(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── Fused surrogate-test kernel ──────────────────────────────────────────────
//
// Grid : (n_pairs_local,)   — one block per (target, source) variable pair.
// Block: (BLOCK,)            — same sizing as te_smem_kernel_pairfree.
//
// For each pair (target_var, source_var) this block iterates over
// ``n_surrogates`` block-shuffled copies of the source variable, computes
// the TE formula fully inside shared memory, and accumulates
//   sum_te[i,j]    = Σ_k te_k
//   sum_sq_te[i,j] = Σ_k te_k²
//   count_ge[i,j]  = #{k : te_k >= te_obs}
// entirely on the GPU. Outputs are pre-zeroed by the caller; because one
// block uniquely owns each (i,j) slot no atomics are required for the final
// write.
//
// Shared memory layout mirrors the pair-free Full-SMEM kernel exactly:
//   [0             .. B3*4)               : int32  cnt3[B3]
//   [B3*4          .. B3*4 + B2*4)        : float  cnt2a[B2]
//   [B3*4 + B2*4   .. B3*4 + 2*B2*4)      : float  cnt2b[B2]
//   [...           .. + B*4)              : float  cnt1_sm[B]
//   [...           .. + WARPS*4)          : float  warp_buf[WARPS]
//
template<typename BinType>
__global__ void full_smem_surrogate_test_kernel(
    const BinType*  __restrict__ bin_arrs,      // (n_vars * T * K,) row-major
    const int32_t*  __restrict__ block_perm,    // (n_surrogates, n_vars, n_blocks)
    const float*    __restrict__ observed_te,   // (n_vars * n_vars,) row-major
    float*          __restrict__ sum_te,        // (n_vars * n_vars,) inout
    float*          __restrict__ sum_sq_te,     // (n_vars * n_vars,) inout
    int32_t*        __restrict__ count_ge,      // (n_vars * n_vars,) inout
    int T, int K, int tau, int L,
    int B, int B2, int B3,
    int block_length, int n_blocks,
    int n_surrogates, int n_vars,
    int64_t pair_offset
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

    // Map linear pair id → (target_var, source_var) (same scheme as pair-free).
    const int64_t global_pair_id = (int64_t)local_id + pair_offset;
    const int nm1 = n_vars - 1;
    int target_var = (int)(global_pair_id / nm1);
    int source_var = (int)(global_pair_id % nm1);
    if (source_var >= target_var) source_var++;

    const BinType* target_base = bin_arrs + (int64_t)target_var * TK;
    const BinType* source_base = bin_arrs + (int64_t)source_var * TK;

    // Observed TE value for this pair (read once). The surrogate-test
    // accumulators follow the matrix convention ``M[i, j] = TE(i -> j)``
    // used throughout TENEX (see ``_pairfree_to_matrix`` /
    // ``_assemble_result_pairfree``): source along rows, target along
    // columns. So for this pair (target_var, source_var) the slot is
    // ``[source_var, target_var]``.
    const int64_t pair_slot = (int64_t)source_var * n_vars + target_var;
    float te_obs = observed_te[pair_slot];

    // Cover length of the "head" region that the permutation touches.
    const int head_len = n_blocks * block_length;

    // Thread-0 registers: surrogate accumulators.
    float   loc_sum    = 0.0f;
    float   loc_sum_sq = 0.0f;
    int32_t loc_count  = 0;

    for (int k = 0; k < n_surrogates; ++k) {
        // Per-surrogate base pointers into block_perm: each variable owns
        // its own block permutation (identical semantics to the existing
        // _block_shuffle_gpu / _random_shuffle_gpu Python paths, which
        // shuffle every variable's time axis independently before calling
        // the pair-free kernel).
        //   block_perm[k, target_var, :]   — drives ``xt1v`` / ``xtv``
        //   block_perm[k, source_var, :]   — drives ``ytv``
        const int32_t* perm_tgt = block_perm
            + ((int64_t)k * n_vars + (int64_t)target_var) * n_blocks;
        const int32_t* perm_src = block_perm
            + ((int64_t)k * n_vars + (int64_t)source_var) * n_blocks;

        // ── Phase 0: clear cnt3 ───────────────────────────────────────────
        for (int b = tid; b < B3; b += BLOCK)
            cnt3[b] = 0;
        __syncthreads();

        // ── Phase 1: build histogram with PERMUTED target + source reads ──
        // The original pair-free kernel reads from ``bin_shuf`` at time
        // indices ``i`` / ``tau + i``; the fused kernel reproduces those
        // reads by indirecting through the per-variable block permutation.
        if (K == 1) {
            for (int i = tid; i < L; i += BLOCK) {
                // tau + i indirection (target "next" sample). tau + i may fall
                // in the head or tail depending on block alignment.
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
                atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
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
                atomicAdd(&cnt3[xt1v * B2 + xtv * B + ytv], 1);
            }
        }
        __syncthreads();

        // ── Phase 2: marginals ───────────────────────────────────────────
        for (int j = tid; j < B; j += BLOCK) {
            float s = 0.0f;
            for (int i = 0; i < B; i++)
                for (int kk = 0; kk < B; kk++)
                    s += (float)cnt3[i * B2 + j * B + kk];
            cnt1_sm[j] = s;
        }
        for (int t = tid; t < B2; t += BLOCK) {
            int i = t / B, j = t % B;
            float s = 0.0f;
            for (int kk = 0; kk < B; kk++)
                s += (float)cnt3[i * B2 + j * B + kk];
            cnt2a[t] = s;
        }
        for (int t = tid; t < B2; t += BLOCK) {
            int j = t / B, kk = t % B;
            float s = 0.0f;
            for (int i = 0; i < B; i++)
                s += (float)cnt3[i * B2 + j * B + kk];
            cnt2b[t] = s;
        }
        __syncthreads();

        // ── Phase 3: TE formula ──────────────────────────────────────────
        float te_local = 0.0f;
        for (int b = tid; b < B3; b += BLOCK) {
            int i = b / B2;
            int j = (b % B2) / B;
            int kk = b % B;

            float c3  = (float)cnt3[b];
            float c2a = cnt2a[i * B + j];
            float c2b = cnt2b[j * B + kk];
            float c1  = cnt1_sm[j];
            float denom = c2a * c2b;

            if (c3 > 0.0f && denom > 0.0f)
                te_local += c3 * log2f(c3 * c1 / denom);
        }

        // ── Phase 4: warp + block reduction ──────────────────────────────
        te_local = warp_reduce_sum_surr(te_local);
        if (tid % 32 == 0)
            warp_buf[tid / 32] = te_local;
        __syncthreads();

        te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
        te_local = warp_reduce_sum_surr(te_local);

        // Thread 0 updates accumulators.
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
void _launch_full_smem_surrogate_test(
    torch::Tensor& bin_arrs,
    torch::Tensor& block_perm,
    torch::Tensor& observed_te,
    torch::Tensor& sum_te,
    torch::Tensor& sum_sq_te,
    torch::Tensor& count_ge,
    int T, int K, int tau, int L,
    int B, int B2, int B3,
    int block_length, int n_blocks,
    int n_surrogates, int n_vars,
    int64_t pair_offset, int64_t n_pairs_local,
    int block_size, size_t smem
) {
    cudaError_t attr_err = cudaFuncSetAttribute(
        full_smem_surrogate_test_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)smem
    );
    TORCH_CHECK(attr_err == cudaSuccess,
        "cudaFuncSetAttribute for full_smem_surrogate_test_kernel failed: ",
        cudaGetErrorString(attr_err));
    full_smem_surrogate_test_kernel<BinType><<<(int)n_pairs_local, block_size, smem>>>(
        bin_arrs.data_ptr<BinType>(),
        block_perm.data_ptr<int32_t>(),
        observed_te.data_ptr<float>(),
        sum_te.data_ptr<float>(),
        sum_sq_te.data_ptr<float>(),
        count_ge.data_ptr<int32_t>(),
        T, K, tau, L, B, B2, B3,
        block_length, n_blocks,
        n_surrogates, n_vars, pair_offset
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ── Python-facing launcher ────────────────────────────────────────────────────
void full_smem_surrogate_test_launch(
    torch::Tensor bin_arrs,      // (n_vars, T, K) int{8,16,32} CUDA
    torch::Tensor block_perm,    // (n_surrogates, n_vars, n_blocks) int32 CUDA
    torch::Tensor observed_te,   // (n_vars, n_vars) float32 CUDA
    torch::Tensor sum_te,        // (n_vars, n_vars) float32 CUDA, pre-zeroed
    torch::Tensor sum_sq_te,     // (n_vars, n_vars) float32 CUDA, pre-zeroed
    torch::Tensor count_ge,      // (n_vars, n_vars) int32 CUDA, pre-zeroed
    int T, int K, int tau,
    int B, int B2, int B3,
    int block_length, int n_surrogates,
    int block_size,
    int n_vars,
    int64_t pair_offset,
    int64_t n_pairs_local
) {
    TORCH_CHECK(n_pairs_local <= (int64_t)INT_MAX,
        "n_pairs_local (", n_pairs_local, ") exceeds INT_MAX grid limit; "
        "use multi-GPU or call with smaller chunks");
    // Device & contiguity — all tensors must live on the same CUDA device
    // and be contiguous (the kernel assumes row-major stride arithmetic).
    TORCH_CHECK(bin_arrs.is_cuda() && block_perm.is_cuda() &&
                observed_te.is_cuda() && sum_te.is_cuda() &&
                sum_sq_te.is_cuda() && count_ge.is_cuda(),
        "all tensors must be on CUDA");
    auto dev = bin_arrs.device();
    TORCH_CHECK(block_perm.device() == dev && observed_te.device() == dev &&
                sum_te.device() == dev && sum_sq_te.device() == dev &&
                count_ge.device() == dev,
        "all tensors must be on the same CUDA device as bin_arrs");
    TORCH_CHECK(bin_arrs.is_contiguous() && block_perm.is_contiguous() &&
                observed_te.is_contiguous() && sum_te.is_contiguous() &&
                sum_sq_te.is_contiguous() && count_ge.is_contiguous(),
        "all tensors must be contiguous");
    // Rank checks
    TORCH_CHECK(bin_arrs.dim() == 3,
        "bin_arrs must have shape (n_vars, T, K)");
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
    int warps = block_size / 32;
    size_t smem = (size_t)B3 * 4
                + (size_t)B2 * 4 * 2
                + (size_t)B  * 4
                + (size_t)warps * 4;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8) {
        _launch_full_smem_surrogate_test<int8_t>(
            bin_arrs, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, B, B2, B3,
            block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    } else if (dtype == torch::kInt16) {
        _launch_full_smem_surrogate_test<int16_t>(
            bin_arrs, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, B, B2, B3,
            block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    } else {
        _launch_full_smem_surrogate_test<int32_t>(
            bin_arrs, block_perm, observed_te,
            sum_te, sum_sq_te, count_ge,
            T, K, tau, L, B, B2, B3,
            block_length, n_blocks,
            n_surrogates, n_vars, pair_offset, n_pairs_local,
            block_size, smem);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("full_smem_surrogate_test_launch", &full_smem_surrogate_test_launch,
          "Fused Full-SMEM surrogate-test kernel (per-pair accumulators on GPU)");
    m.def("get_smem_optin", &get_smem_optin,
          "Max configurable dynamic shared memory per block (opt-in) for current device");
}
