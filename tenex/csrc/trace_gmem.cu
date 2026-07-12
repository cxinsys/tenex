/*
 * POINT Global-Memory kernel — OutTE/InTE for arbitrary connected-set sizes.
 *
 * Unlike the SMEM variant (trace_smem.cu), this kernel stores the 3D histogram
 * in global memory, removing the SMEM size constraint. This allows handling
 * any connected-set size d (B^d can be large), at the cost of slower atomicAdd
 * (~80× slower than SMEM). However, since POINT computes only n tasks (one per
 * variable) rather than n² pairs, the slower atomics are acceptable.
 *
 * Fixes vs trace_smem.cu:
 *   1. No SMEM overflow — histogram in global memory (up to VRAM limit)
 *   2. Uses int32 bin_arrs (no uint8/int8 sign issues)
 *   3. Uses standard log2f (no --use_fast_math precision loss)
 *
 * OutTE(X) = H[Y(t)/Y_] - H[Y(t)/X_, Y_]
 *   cnt3[Y_future_enc, Y_past_enc, X_past] — dims: By × By × Bx
 *
 * InTE(X) = H[X(t)/X_] - H[X(t)/X_, Y_]
 *   cnt3[X_future, X_past, Y_past_enc] — dims: Bx × Bx × By
 *
 * Task descriptor layout (same as trace_smem.cu):
 *   [0]: var_idx
 *   [1]: n_connected (d)
 *   [2]: Bx (bins for the univariate variable)
 *   [3]: By_enc (product of connected variable bins = Π b_j)
 *   [4..4+d): connected variable indices
 *   [4+d..4+2d): cumulative radix multipliers
 *   [4+2d]: K
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}


// ── OutTE kernel ────────────────────────────────────────────────────────────

__global__ void trace_outte_gmem_kernel(
    const int32_t* __restrict__ bin_arrs,      // (n_vars * T * K) int32
    const int32_t* __restrict__ task_desc,
    const int32_t* __restrict__ task_offsets,   // (n_tasks,)
    const int64_t* __restrict__ cnt3_offsets,   // (n_tasks+1,) prefix sum of hist sizes
    int32_t*       __restrict__ cnt3_gmem,      // global cnt3 buffer (pre-zeroed)
    float*         __restrict__ outte_ptr,      // (n_tasks,)
    int T, int tau
) {
    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    extern __shared__ char smem_buf[];
    float* warp_buf = (float*)smem_buf;

    const int32_t* desc = task_desc + task_offsets[task_id];
    int var_idx   = desc[0];
    int n_targets = desc[1];
    int Bx        = desc[2];
    int By        = desc[3];
    int K         = desc[4 + 2 * n_targets];
    int TK        = T * K;
    int L         = T - tau;
    int N         = L * K;

    if (n_targets == 0 || By == 0) {
        if (tid == 0) outte_ptr[task_id] = 0.0f;
        return;
    }

    int hist_size = By * By * Bx;
    int32_t* cnt3 = cnt3_gmem + cnt3_offsets[task_id];

    // ── Phase 1: build histogram in global memory ───────────────────────
    for (int i = tid; i < N; i += BLOCK) {
        int t = (K == 1) ? i : i / K;
        int c = (K == 1) ? 0 : i % K;

        int y_future_enc = 0;
        int y_past_enc = 0;
        for (int d = 0; d < n_targets; d++) {
            int tj = desc[4 + d];
            int radix = desc[4 + n_targets + d];
            y_future_enc += bin_arrs[(int64_t)tj * TK + (tau + t) * K + c] * radix;
            y_past_enc   += bin_arrs[(int64_t)tj * TK + t * K + c] * radix;
        }
        int x_past = bin_arrs[(int64_t)var_idx * TK + t * K + c];

        atomicAdd(&cnt3[y_future_enc * By * Bx + y_past_enc * Bx + x_past], 1);
    }
    __syncthreads();

    // ── Phase 2-3: marginals + TE in one pass (to avoid large marginal arrays) ──
    // For each (a, b, c) in cnt3, accumulate TE term directly.
    // cnt1[b] = Σ_{a,c} cnt3[a,b,c]  — computed on the fly per b
    // cnt2a[a,b] = Σ_c cnt3[a,b,c]   — computed on the fly per (a,b)
    // cnt2b[b,c] = Σ_a cnt3[a,b,c]   — computed on the fly per (b,c)
    //
    // To avoid O(By² + By*Bx) shared memory for marginals (which may also be large),
    // we compute marginals in registers per thread's assigned bins.

    float te_local = 0.0f;

    for (int flat = tid; flat < hist_size; flat += BLOCK) {
        int a = flat / (By * Bx);
        int rem = flat % (By * Bx);
        int b = rem / Bx;
        int c = rem % Bx;

        float c3 = (float)cnt3[flat];
        if (c3 == 0.0f) continue;

        // cnt2a[a,b] = Σ_c cnt3[a,b,c]
        float c2a = 0.0f;
        for (int cc = 0; cc < Bx; cc++)
            c2a += (float)cnt3[a * By * Bx + b * Bx + cc];

        // cnt2b[b,c] = Σ_a cnt3[a,b,c]
        float c2b = 0.0f;
        for (int aa = 0; aa < By; aa++)
            c2b += (float)cnt3[aa * By * Bx + b * Bx + c];

        // cnt1[b] = Σ_{a,c} cnt3[a,b,c]
        float c1 = 0.0f;
        for (int aa = 0; aa < By; aa++)
            for (int cc = 0; cc < Bx; cc++)
                c1 += (float)cnt3[aa * By * Bx + b * Bx + cc];

        float denom = c2a * c2b;
        if (denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: block reduction ────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        outte_ptr[task_id] = te_local / (float)N;
}


// ── InTE kernel ─────────────────────────────────────────────────────────────

__global__ void trace_inte_gmem_kernel(
    const int32_t* __restrict__ bin_arrs,
    const int32_t* __restrict__ task_desc,
    const int32_t* __restrict__ task_offsets,
    const int64_t* __restrict__ cnt3_offsets,
    int32_t*       __restrict__ cnt3_gmem,
    float*         __restrict__ inte_ptr,
    int T, int tau
) {
    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    extern __shared__ char smem_buf[];
    float* warp_buf = (float*)smem_buf;

    const int32_t* desc = task_desc + task_offsets[task_id];
    int var_idx   = desc[0];
    int n_sources = desc[1];
    int Bx        = desc[2];
    int By        = desc[3];
    int K         = desc[4 + 2 * n_sources];
    int TK        = T * K;
    int L         = T - tau;
    int N         = L * K;

    if (n_sources == 0 || By == 0) {
        if (tid == 0) inte_ptr[task_id] = 0.0f;
        return;
    }

    int hist_size = Bx * Bx * By;
    int32_t* cnt3 = cnt3_gmem + cnt3_offsets[task_id];

    // ── Phase 1: build histogram ────────────────────────────────────────
    for (int i = tid; i < N; i += BLOCK) {
        int t = (K == 1) ? i : i / K;
        int c = (K == 1) ? 0 : i % K;

        int x_future = bin_arrs[(int64_t)var_idx * TK + (tau + t) * K + c];
        int x_past   = bin_arrs[(int64_t)var_idx * TK + t * K + c];

        int y_past_enc = 0;
        for (int d = 0; d < n_sources; d++) {
            int sj = desc[4 + d];
            int radix = desc[4 + n_sources + d];
            y_past_enc += bin_arrs[(int64_t)sj * TK + t * K + c] * radix;
        }

        atomicAdd(&cnt3[x_future * Bx * By + x_past * By + y_past_enc], 1);
    }
    __syncthreads();

    // ── Phase 2-3: marginals + TE in one pass ───────────────────────────
    float te_local = 0.0f;

    for (int flat = tid; flat < hist_size; flat += BLOCK) {
        int a = flat / (Bx * By);
        int rem = flat % (Bx * By);
        int b = rem / By;
        int c_idx = rem % By;

        float c3 = (float)cnt3[flat];
        if (c3 == 0.0f) continue;

        float c2a = 0.0f;
        for (int cc = 0; cc < By; cc++)
            c2a += (float)cnt3[a * Bx * By + b * By + cc];

        float c2b = 0.0f;
        for (int aa = 0; aa < Bx; aa++)
            c2b += (float)cnt3[aa * Bx * By + b * By + c_idx];

        float c1 = 0.0f;
        for (int aa = 0; aa < Bx; aa++)
            for (int cc = 0; cc < By; cc++)
                c1 += (float)cnt3[aa * Bx * By + b * By + cc];

        float denom = c2a * c2b;
        if (denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: block reduction ────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0)
        warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        inte_ptr[task_id] = te_local / (float)N;
}


// ── Launchers ───────────────────────────────────────────────────────────────

torch::Tensor trace_outte_gmem_launch(
    torch::Tensor bin_arrs,      // (n_vars * T * K) int32 CUDA
    torch::Tensor task_desc,     // packed int32 CUDA
    torch::Tensor task_offsets,  // (n_tasks,) int32 CUDA
    torch::Tensor cnt3_offsets,  // (n_tasks+1,) int64 CUDA
    torch::Tensor cnt3_gmem,    // (total_cells,) int32 CUDA, pre-zeroed
    int T, int tau, int block_size
) {
    int n_tasks = task_offsets.size(0);
    auto outte = torch::zeros({n_tasks}, bin_arrs.options().dtype(torch::kFloat32));
    int smem = (block_size / 32) * 4;  // warp_buf only

    trace_outte_gmem_kernel<<<n_tasks, block_size, smem>>>(
        bin_arrs.data_ptr<int32_t>(),
        task_desc.data_ptr<int32_t>(),
        task_offsets.data_ptr<int32_t>(),
        cnt3_offsets.data_ptr<int64_t>(),
        cnt3_gmem.data_ptr<int32_t>(),
        outte.data_ptr<float>(),
        T, tau);
    return outte;
}

torch::Tensor trace_inte_gmem_launch(
    torch::Tensor bin_arrs,
    torch::Tensor task_desc,
    torch::Tensor task_offsets,
    torch::Tensor cnt3_offsets,
    torch::Tensor cnt3_gmem,
    int T, int tau, int block_size
) {
    int n_tasks = task_offsets.size(0);
    auto inte = torch::zeros({n_tasks}, bin_arrs.options().dtype(torch::kFloat32));
    int smem = (block_size / 32) * 4;

    trace_inte_gmem_kernel<<<n_tasks, block_size, smem>>>(
        bin_arrs.data_ptr<int32_t>(),
        task_desc.data_ptr<int32_t>(),
        task_offsets.data_ptr<int32_t>(),
        cnt3_offsets.data_ptr<int64_t>(),
        cnt3_gmem.data_ptr<int32_t>(),
        inte.data_ptr<float>(),
        T, tau);
    return inte;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trace_outte_gmem_launch", &trace_outte_gmem_launch);
    m.def("trace_inte_gmem_launch", &trace_inte_gmem_launch);
}
