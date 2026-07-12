/*
 * POINT SMEM kernel — Pruned Outgoing and INcoming Transfer entropy.
 *
 * Based on the Adaptive-SMEM kernel, adapted for multivariate TE:
 *   OutTE(X) = H[Y(t)/Y_] - H[Y(t)/X_, Y_]
 *   InTE(X)  = H[X(t)/X_] - H[X(t)/X_, Y_]
 *
 * where Y = connected variables (small set after pruning).
 *
 * Key idea: encode multivariate Y as a single integer via mixed-radix encoding,
 * then use the same 3D histogram approach as Adaptive-SMEM.
 *
 * For OutTE(X) with connected targets {j1, j2, ..., jd}:
 *   - "target" = multivariate Y (encoded), bins = product of per-var bins
 *   - "source" = univariate X
 *   - cnt3[Y_future_enc, Y_past_enc, X_past] in SMEM
 *   - TE = (1/N) Σ cnt3 * log2(cnt3 * cnt1 / (cnt2a * cnt2b))
 *
 * For InTE(X) with connected sources {j1, j2, ..., jd}:
 *   - "target" = univariate X
 *   - "source" = multivariate Y (encoded)
 *   - cnt3[X_future, X_past, Y_past_enc] in SMEM
 *   - Same TE formula
 *
 * Grid:  (n_tasks,)  — one block per OutTE or InTE computation
 * Block: (BLOCK,)    — power of 2, 128-1024
 */

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math.h>

// ── Warp-level reduction ────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v,  8);
    v += __shfl_down_sync(0xffffffff, v,  4);
    v += __shfl_down_sync(0xffffffff, v,  2);
    v += __shfl_down_sync(0xffffffff, v,  1);
    return v;
}

// ── OutTE kernel: one variable → its connected targets ──────────────────────
//
// task_desc layout per task (variable i):
//   [0]: var_idx (i)
//   [1]: n_targets (d)
//   [2]: Bx (bins for variable i)
//   [3]: By_enc (product of target bins = Πj b_j)
//   [4..4+d): target variable indices
//   [4+d..4+2d): cumulative radix multipliers for encoding
//
// cnt3 dims: [Y_future_enc, Y_past_enc, X_past] = By_enc * By_enc * Bx
//
template<typename BinType>
__global__ void trace_outte_kernel(
    const BinType*  __restrict__ bin_arrs,   // (n_vars * T,) row-major (K=1)
    const int32_t* __restrict__ task_desc,   // packed task descriptors
    const int32_t* __restrict__ task_offsets, // (n_tasks,) offset into task_desc
    float*         __restrict__ outte_ptr,   // (n_tasks,) output
    int T, int tau, int L
) {
    extern __shared__ char smem_buf[];

    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    // Read task descriptor
    const int32_t* desc = task_desc + task_offsets[task_id];
    int var_idx   = desc[0];
    int n_targets = desc[1];
    int Bx        = desc[2];
    int By_enc    = desc[3];
    // target indices: desc[4 .. 4+n_targets)
    // radix mults:    desc[4+n_targets .. 4+2*n_targets)

    if (n_targets == 0 || By_enc == 0) {
        if (tid == 0) outte_ptr[task_id] = 0.0f;
        return;
    }

    int By2 = By_enc * By_enc;
    int hist_size = By2 * Bx;  // cnt3 size

    // SMEM layout: cnt3[hist_size] + cnt2a[By2] + cnt2b[By_enc*Bx] + cnt1[By_enc] + warp_buf[WARPS]
    int32_t* cnt3    = (int32_t*)smem_buf;
    float*   cnt2a   = (float*)(cnt3 + hist_size);
    float*   cnt2b   = cnt2a + By2;
    float*   cnt1_sm = cnt2b + By_enc * Bx;
    float*   warp_buf = cnt1_sm + By_enc;

    // ── Phase 0: clear histogram ────────────────────────────────────────
    for (int b = tid; b < hist_size; b += BLOCK)
        cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram ────────────────────────────────────────
    // K is read from desc[4 + 2*n_targets]
    int K = desc[4 + 2 * n_targets];
    int TK = T * K;
    int N = L * K;

    for (int i = tid; i < N; i += BLOCK) {
        int t = (K == 1) ? i : i / K;
        int c = (K == 1) ? 0 : i % K;

        // Encode Y_future = Σ bin_arrs[target_j, (t+tau)*K+c] * radix_j
        int y_future_enc = 0;
        int y_past_enc = 0;
        for (int d = 0; d < n_targets; d++) {
            int tj = desc[4 + d];
            int radix = desc[4 + n_targets + d];
            y_future_enc += (int)__ldg(&bin_arrs[(int64_t)tj * TK + (tau + t) * K + c]) * radix;
            y_past_enc   += (int)__ldg(&bin_arrs[(int64_t)tj * TK + t * K + c]) * radix;
        }
        int x_past = (int)__ldg(&bin_arrs[(int64_t)var_idx * TK + t * K + c]);

        // cnt3[y_future_enc, y_past_enc, x_past]
        atomicAdd(&cnt3[y_future_enc * By_enc * Bx + y_past_enc * Bx + x_past], 1);
    }
    __syncthreads();

    // ── Phase 2: marginals ──────────────────────────────────────────────
    // cnt1[y_past_enc] = Σ_{y_future, x} cnt3[y_future, y_past, x]
    for (int j = tid; j < By_enc; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < By_enc; i++)
            for (int k = 0; k < Bx; k++)
                s += (float)cnt3[i * By_enc * Bx + j * Bx + k];
        cnt1_sm[j] = s;
    }
    // cnt2a[y_future, y_past] = Σ_x cnt3[y_future, y_past, x]
    for (int t = tid; t < By2; t += BLOCK) {
        int i = t / By_enc, j = t % By_enc;
        float s = 0.0f;
        for (int k = 0; k < Bx; k++)
            s += (float)cnt3[i * By_enc * Bx + j * Bx + k];
        cnt2a[t] = s;
    }
    // cnt2b[y_past, x] = Σ_{y_future} cnt3[y_future, y_past, x]
    for (int t = tid; t < By_enc * Bx; t += BLOCK) {
        int j = t / Bx, k = t % Bx;
        float s = 0.0f;
        for (int i = 0; i < By_enc; i++)
            s += (float)cnt3[i * By_enc * Bx + j * Bx + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula ─────────────────────────────────────────────
    float te_local = 0.0f;
    for (int b = tid; b < hist_size; b += BLOCK) {
        int i  = b / (By_enc * Bx);
        int jk = b % (By_enc * Bx);
        int j  = jk / Bx;
        int k  = jk % Bx;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * By_enc + j];
        float c2b = cnt2b[j * Bx + k];
        float c1  = cnt1_sm[j];
        float denom = c2a * c2b;

        if (c3 > 0.0f && denom > 0.0f)
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


// ── InTE kernel: connected sources → one variable ───────────────────────────
//
// InTE(X) = H[X(t)/X_] - H[X(t)/X_, Y_]
// cnt3[X_future, X_past, Y_past_enc] = Bx * Bx * By_enc
//
template<typename BinType>
__global__ void trace_inte_kernel(
    const BinType*  __restrict__ bin_arrs,
    const int32_t* __restrict__ task_desc,
    const int32_t* __restrict__ task_offsets,
    float*         __restrict__ inte_ptr,
    int T, int tau, int L
) {
    extern __shared__ char smem_buf[];

    const int task_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int BLOCK   = blockDim.x;
    const int WARPS   = BLOCK / 32;

    const int32_t* desc = task_desc + task_offsets[task_id];
    int var_idx   = desc[0];
    int n_sources = desc[1];
    int Bx        = desc[2];
    int By_enc    = desc[3];

    if (n_sources == 0 || By_enc == 0) {
        if (tid == 0) inte_ptr[task_id] = 0.0f;
        return;
    }

    int Bx2 = Bx * Bx;
    int hist_size = Bx2 * By_enc;

    int32_t* cnt3    = (int32_t*)smem_buf;
    float*   cnt2a   = (float*)(cnt3 + hist_size);
    float*   cnt2b   = cnt2a + Bx2;
    float*   cnt1_sm = cnt2b + Bx * By_enc;
    float*   warp_buf = cnt1_sm + Bx;

    // ── Phase 0: clear ──────────────────────────────────────────────────
    for (int b = tid; b < hist_size; b += BLOCK)
        cnt3[b] = 0;
    __syncthreads();

    // ── Phase 1: build histogram ────────────────────────────────────────
    int K = desc[4 + 2 * n_sources];
    int TK = T * K;
    int N = L * K;

    for (int i = tid; i < N; i += BLOCK) {
        int t = (K == 1) ? i : i / K;
        int c = (K == 1) ? 0 : i % K;

        int x_future = (int)__ldg(&bin_arrs[(int64_t)var_idx * TK + (tau + t) * K + c]);
        int x_past   = (int)__ldg(&bin_arrs[(int64_t)var_idx * TK + t * K + c]);

        int y_past_enc = 0;
        for (int d = 0; d < n_sources; d++) {
            int sj = desc[4 + d];
            int radix = desc[4 + n_sources + d];
            y_past_enc += (int)__ldg(&bin_arrs[(int64_t)sj * TK + t * K + c]) * radix;
        }

        atomicAdd(&cnt3[x_future * Bx * By_enc + x_past * By_enc + y_past_enc], 1);
    }
    __syncthreads();

    // ── Phase 2: marginals ──────────────────────────────────────────────
    for (int j = tid; j < Bx; j += BLOCK) {
        float s = 0.0f;
        for (int i = 0; i < Bx; i++)
            for (int k = 0; k < By_enc; k++)
                s += (float)cnt3[i * Bx * By_enc + j * By_enc + k];
        cnt1_sm[j] = s;
    }
    for (int t = tid; t < Bx2; t += BLOCK) {
        int i = t / Bx, j = t % Bx;
        float s = 0.0f;
        for (int k = 0; k < By_enc; k++)
            s += (float)cnt3[i * Bx * By_enc + j * By_enc + k];
        cnt2a[t] = s;
    }
    for (int t = tid; t < Bx * By_enc; t += BLOCK) {
        int j = t / By_enc, k = t % By_enc;
        float s = 0.0f;
        for (int i = 0; i < Bx; i++)
            s += (float)cnt3[i * Bx * By_enc + j * By_enc + k];
        cnt2b[t] = s;
    }
    __syncthreads();

    // ── Phase 3: TE formula ─────────────────────────────────────────────
    float te_local = 0.0f;
    for (int b = tid; b < hist_size; b += BLOCK) {
        int i  = b / (Bx * By_enc);
        int jk = b % (Bx * By_enc);
        int j  = jk / By_enc;
        int k  = jk % By_enc;

        float c3  = (float)cnt3[b];
        float c2a = cnt2a[i * Bx + j];
        float c2b = cnt2b[j * By_enc + k];
        float c1  = cnt1_sm[j];
        float denom = c2a * c2b;

        if (c3 > 0.0f && denom > 0.0f)
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


// ── Python-facing launchers ─────────────────────────────────────────────────

template<typename BinType>
void _launch_outte(
    torch::Tensor& bin_arrs, torch::Tensor& task_desc, torch::Tensor& task_offsets,
    torch::Tensor& outte, int T, int tau, int L, int block_size, int smem_bytes, int n_tasks
) {
    cudaFuncSetAttribute(
        trace_outte_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    trace_outte_kernel<BinType><<<n_tasks, block_size, smem_bytes>>>(
        bin_arrs.data_ptr<BinType>(),
        task_desc.data_ptr<int32_t>(),
        task_offsets.data_ptr<int32_t>(),
        outte.data_ptr<float>(),
        T, tau, L);
}

template<typename BinType>
void _launch_inte(
    torch::Tensor& bin_arrs, torch::Tensor& task_desc, torch::Tensor& task_offsets,
    torch::Tensor& inte, int T, int tau, int L, int block_size, int smem_bytes, int n_tasks
) {
    cudaFuncSetAttribute(
        trace_inte_kernel<BinType>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    trace_inte_kernel<BinType><<<n_tasks, block_size, smem_bytes>>>(
        bin_arrs.data_ptr<BinType>(),
        task_desc.data_ptr<int32_t>(),
        task_offsets.data_ptr<int32_t>(),
        inte.data_ptr<float>(),
        T, tau, L);
}

torch::Tensor trace_outte_launch(
    torch::Tensor bin_arrs,
    torch::Tensor task_desc,
    torch::Tensor task_offsets,
    int T, int tau, int block_size, int smem_bytes
) {
    int n_tasks = task_offsets.size(0);
    auto outte = torch::zeros({n_tasks}, bin_arrs.options().dtype(torch::kFloat32));
    int L = T - tau;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8)
        _launch_outte<int8_t>(bin_arrs, task_desc, task_offsets, outte, T, tau, L, block_size, smem_bytes, n_tasks);
    else if (dtype == torch::kInt16)
        _launch_outte<int16_t>(bin_arrs, task_desc, task_offsets, outte, T, tau, L, block_size, smem_bytes, n_tasks);
    else
        _launch_outte<int32_t>(bin_arrs, task_desc, task_offsets, outte, T, tau, L, block_size, smem_bytes, n_tasks);
    return outte;
}

torch::Tensor trace_inte_launch(
    torch::Tensor bin_arrs,
    torch::Tensor task_desc,
    torch::Tensor task_offsets,
    int T, int tau, int block_size, int smem_bytes
) {
    int n_tasks = task_offsets.size(0);
    auto inte = torch::zeros({n_tasks}, bin_arrs.options().dtype(torch::kFloat32));
    int L = T - tau;

    auto dtype = bin_arrs.scalar_type();
    if (dtype == torch::kInt8)
        _launch_inte<int8_t>(bin_arrs, task_desc, task_offsets, inte, T, tau, L, block_size, smem_bytes, n_tasks);
    else if (dtype == torch::kInt16)
        _launch_inte<int16_t>(bin_arrs, task_desc, task_offsets, inte, T, tau, L, block_size, smem_bytes, n_tasks);
    else
        _launch_inte<int32_t>(bin_arrs, task_desc, task_offsets, inte, T, tau, L, block_size, smem_bytes, n_tasks);
    return inte;
}

// ── SMEM query ──────────────────────────────────────────────────────────────
int64_t trace_get_smem_optin() {
    int dev = 0, val = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return (int64_t)val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trace_outte_launch", &trace_outte_launch,
          "POINT OutTE kernel (multivariate target, SMEM histogram)");
    m.def("trace_inte_launch", &trace_inte_launch,
          "POINT InTE kernel (multivariate source, SMEM histogram)");
    m.def("trace_get_smem_optin", &trace_get_smem_optin,
          "Max SMEM per block (opt-in)");
}
