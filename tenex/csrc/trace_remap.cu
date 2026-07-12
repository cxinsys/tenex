/*
 * POINT Remap kernel — OutTE/InTE with multivariate state remapping.
 *
 * Key insight: for d connected variables with B bins each, the theoretical
 * state space is B^d (exponential). But with L time points, at most L unique
 * multivariate states are observed. By remapping observed states to dense
 * indices [0, U), the histogram size becomes U² × Bx where U ≤ L.
 *
 * This makes histogram size independent of d (connected-set size), bounded
 * only by L (time series length). All tasks can run on GPU regardless of d.
 *
 * Workflow per task:
 *   1. Python pre-computes remapped arrays: Y_future_remap[L], Y_past_remap[L]
 *      with values in [0, U_future) and [0, U_past) respectively.
 *   2. Kernel builds cnt3[U_future, U_past, Bx] in global memory.
 *   3. Kernel computes TE from cnt3 (same Phase 2-4 as Adaptive-SMEM).
 *
 * This approach mirrors TENEX's per-var remap_bins for pairwise TE,
 * applied to multivariate states.
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


// ── OutTE kernel with remapped multivariate states ──────────────────────────
//
// One block per task (variable).
// Inputs per task:
//   y_future_remap[L]  — remapped multivariate future, values in [0, Uf)
//   y_past_remap[L]    — remapped multivariate past,   values in [0, Up)
//   x_past[L]          — univariate past of var X,     values in [0, Bx)
//   Uf, Up, Bx         — dimensions
//
// cnt3[y_future, y_past, x_past] in global memory, size = Uf * Up * Bx
//
__global__ void trace_outte_remap_kernel(
    const int32_t* __restrict__ y_future_all,  // (total_L,) concatenated
    const int32_t* __restrict__ y_past_all,    // (total_L,)
    const int32_t* __restrict__ x_past_all,    // (total_L,)
    const int32_t* __restrict__ task_meta,     // (n_tasks, 5): [L, Uf, Up, Bx, data_offset]
    const int64_t* __restrict__ cnt3_offsets,  // (n_tasks+1,) prefix sum
    int32_t*       __restrict__ cnt3_gmem,     // pre-zeroed global cnt3
    float*         __restrict__ outte_ptr      // (n_tasks,) output
) {
    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    extern __shared__ char smem[];
    float* warp_buf = (float*)smem;

    int L           = task_meta[task_id * 5 + 0];
    int Uf          = task_meta[task_id * 5 + 1];
    int Up          = task_meta[task_id * 5 + 2];
    int Bx          = task_meta[task_id * 5 + 3];
    int data_offset = task_meta[task_id * 5 + 4];

    if (L == 0 || Uf == 0 || Up == 0) {
        if (tid == 0) outte_ptr[task_id] = 0.0f;
        return;
    }

    int hist_size = Uf * Up * Bx;
    int32_t* cnt3 = cnt3_gmem + cnt3_offsets[task_id];

    const int32_t* yf = y_future_all + data_offset;
    const int32_t* yp = y_past_all   + data_offset;
    const int32_t* xp = x_past_all   + data_offset;

    // ── Phase 1: build histogram ────────────────────────────────────────
    for (int t = tid; t < L; t += BLOCK) {
        int a = yf[t];  // Y_future remapped
        int b = yp[t];  // Y_past remapped
        int c = xp[t];  // X_past raw
        atomicAdd(&cnt3[a * Up * Bx + b * Bx + c], 1);
    }
    __syncthreads();

    // ── Phase 2-3: marginals + TE in one pass ───────────────────────────
    float te_local = 0.0f;

    for (int flat = tid; flat < hist_size; flat += BLOCK) {
        int a   = flat / (Up * Bx);
        int rem = flat % (Up * Bx);
        int b   = rem / Bx;
        int c   = rem % Bx;

        float c3 = (float)cnt3[flat];
        if (c3 == 0.0f) continue;

        // cnt2a[a, b] = Σ_c cnt3[a, b, c]
        float c2a = 0.0f;
        for (int cc = 0; cc < Bx; cc++)
            c2a += (float)cnt3[a * Up * Bx + b * Bx + cc];

        // cnt2b[b, c] = Σ_a cnt3[a, b, c]
        float c2b = 0.0f;
        for (int aa = 0; aa < Uf; aa++)
            c2b += (float)cnt3[aa * Up * Bx + b * Bx + c];

        // cnt1[b] = Σ_{a,c} cnt3[a, b, c]
        float c1 = 0.0f;
        for (int aa = 0; aa < Uf; aa++)
            for (int cc = 0; cc < Bx; cc++)
                c1 += (float)cnt3[aa * Up * Bx + b * Bx + cc];

        float denom = c2a * c2b;
        if (denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: reduction ──────────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0) warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        outte_ptr[task_id] = te_local / (float)L;
}


// ── InTE kernel with remapped multivariate states ───────────────────────────
//
// cnt3[X_future, X_past, Y_past_remap] — size = Bx * Bx * Up
//
__global__ void trace_inte_remap_kernel(
    const int32_t* __restrict__ x_future_all,  // (total_L,)
    const int32_t* __restrict__ x_past_all,    // (total_L,)
    const int32_t* __restrict__ y_past_all,    // (total_L,) remapped
    const int32_t* __restrict__ task_meta,     // (n_tasks, 4): [L, Bx, Up, data_offset]
    const int64_t* __restrict__ cnt3_offsets,
    int32_t*       __restrict__ cnt3_gmem,
    float*         __restrict__ inte_ptr
) {
    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    extern __shared__ char smem[];
    float* warp_buf = (float*)smem;

    int L           = task_meta[task_id * 4 + 0];
    int Bx          = task_meta[task_id * 4 + 1];
    int Up          = task_meta[task_id * 4 + 2];
    int data_offset = task_meta[task_id * 4 + 3];

    if (L == 0 || Up == 0) {
        if (tid == 0) inte_ptr[task_id] = 0.0f;
        return;
    }

    int hist_size = Bx * Bx * Up;
    int32_t* cnt3 = cnt3_gmem + cnt3_offsets[task_id];

    const int32_t* xf = x_future_all + data_offset;
    const int32_t* xp = x_past_all   + data_offset;
    const int32_t* yp = y_past_all   + data_offset;

    // ── Phase 1: build histogram ────────────────────────────────────────
    for (int t = tid; t < L; t += BLOCK) {
        int a = xf[t];
        int b = xp[t];
        int c = yp[t];
        atomicAdd(&cnt3[a * Bx * Up + b * Up + c], 1);
    }
    __syncthreads();

    // ── Phase 2-3: marginals + TE ───────────────────────────────────────
    float te_local = 0.0f;

    for (int flat = tid; flat < hist_size; flat += BLOCK) {
        int a   = flat / (Bx * Up);
        int rem = flat % (Bx * Up);
        int b   = rem / Up;
        int c   = rem % Up;

        float c3 = (float)cnt3[flat];
        if (c3 == 0.0f) continue;

        float c2a = 0.0f;
        for (int cc = 0; cc < Up; cc++)
            c2a += (float)cnt3[a * Bx * Up + b * Up + cc];

        float c2b = 0.0f;
        for (int aa = 0; aa < Bx; aa++)
            c2b += (float)cnt3[aa * Bx * Up + b * Up + c];

        float c1 = 0.0f;
        for (int aa = 0; aa < Bx; aa++)
            for (int cc = 0; cc < Up; cc++)
                c1 += (float)cnt3[aa * Bx * Up + b * Up + cc];

        float denom = c2a * c2b;
        if (denom > 0.0f)
            te_local += c3 * log2f(c3 * c1 / denom);
    }

    // ── Phase 4: reduction ──────────────────────────────────────────────
    te_local = warp_reduce_sum(te_local);
    if (tid % 32 == 0) warp_buf[tid / 32] = te_local;
    __syncthreads();

    te_local = (tid < WARPS) ? warp_buf[tid] : 0.0f;
    te_local = warp_reduce_sum(te_local);

    if (tid == 0)
        inte_ptr[task_id] = te_local / (float)L;
}


// ── Launchers ───────────────────────────────────────────────────────────────

torch::Tensor trace_outte_remap_launch(
    torch::Tensor y_future_all,
    torch::Tensor y_past_all,
    torch::Tensor x_past_all,
    torch::Tensor task_meta,
    torch::Tensor cnt3_offsets,
    torch::Tensor cnt3_gmem,
    int block_size
) {
    int n_tasks = task_meta.size(0);
    auto outte = torch::zeros({n_tasks}, cnt3_gmem.options().dtype(torch::kFloat32));
    int smem = (block_size / 32) * 4;

    trace_outte_remap_kernel<<<n_tasks, block_size, smem>>>(
        y_future_all.data_ptr<int32_t>(),
        y_past_all.data_ptr<int32_t>(),
        x_past_all.data_ptr<int32_t>(),
        task_meta.data_ptr<int32_t>(),
        cnt3_offsets.data_ptr<int64_t>(),
        cnt3_gmem.data_ptr<int32_t>(),
        outte.data_ptr<float>());
    return outte;
}

torch::Tensor trace_inte_remap_launch(
    torch::Tensor x_future_all,
    torch::Tensor x_past_all,
    torch::Tensor y_past_all,
    torch::Tensor task_meta,
    torch::Tensor cnt3_offsets,
    torch::Tensor cnt3_gmem,
    int block_size
) {
    int n_tasks = task_meta.size(0);
    auto inte = torch::zeros({n_tasks}, cnt3_gmem.options().dtype(torch::kFloat32));
    int smem = (block_size / 32) * 4;

    trace_inte_remap_kernel<<<n_tasks, block_size, smem>>>(
        x_future_all.data_ptr<int32_t>(),
        x_past_all.data_ptr<int32_t>(),
        y_past_all.data_ptr<int32_t>(),
        task_meta.data_ptr<int32_t>(),
        cnt3_offsets.data_ptr<int64_t>(),
        cnt3_gmem.data_ptr<int32_t>(),
        inte.data_ptr<float>());
    return inte;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trace_outte_remap_launch", &trace_outte_remap_launch);
    m.def("trace_inte_remap_launch", &trace_inte_remap_launch);
}
