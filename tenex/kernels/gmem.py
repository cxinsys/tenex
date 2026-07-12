"""
Global-memory kernel for Transfer Entropy — fallback for pairs exceeding SMEM budget.

Problems solved vs Adaptive-SMEM kernel (kernels/adaptive_smem.py)
---------------------------------------------------------
1. Overflow scatter_add bottleneck:
   - compute_te_scatter_add uses b_max³ tensors (584 MB for CeNGEN)
   - PyTorch TE formula on 584 MB tensors = ~5 GB intermediate global-memory traffic
   - GMem kernel: dedicated CUDA kernel, per-pair global-memory cnt3 via prefix-sum offset table
     → marginals only in SMEM (≤ 22 KB, always fits 48 KB default SMEM)
     → global-memory traffic: ~5 GB → ~200 MB (~25× reduction)

2. bin_arrs int32 → int8 (optional, for b_max ≤ 127):
   - 4× smaller bin_arrs memory (9 GB → 2.3 GB for CeNGEN full set)
   - 4× less global-memory bandwidth during Phase 1 (histogram building)
   - Better L2 cache utilization

Kernel design
-------------
Grid:  (n_overflow,) — one block per overflow pair
Block: (BLOCK,)      — power of 2, 128–1024

SMEM layout (marginals only — cnt3 stays in global memory):
  [0             .. b_t²×4)       : float cnt2a[b_t*b_t]
  [b_t²×4         .. +b_tb_s×4)    : float cnt2b[b_t*b_s]
  [+b_tb_s×4       .. +b_t×4)      : float cnt1[b_t]
  [+b_t×4         .. +WARPS×4)   : float warp_buf[WARPS]

Max SMEM (b_t=b_s=53): 53²×8 + 53×4 + 32×4 = 22,472 + 212 + 128 = 22,812 B < 48 KB ✓

Phase 1: atomicAdd to global-memory cnt3 (per-pair region)
Phase 2: sequential read global-memory cnt3 → marginals in SMEM
Phase 3: sequential read global-memory cnt3 + SMEM marginals → TE formula → warp reduction
"""


import threading
from typing import Optional

import numpy as np
import torch

# ── CUDA source ────────────────────────────────────────────────────────────────

_CUDA_SRC_GMEM = r"""
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
"""

# ── Python module cache ────────────────────────────────────────────────────────

_te_gmem_module = None
_te_gmem_lock = threading.Lock()

def _load_module():
    global _te_gmem_module
    if _te_gmem_module is not None:
        return _te_gmem_module

    with _te_gmem_lock:
        if _te_gmem_module is not None:
            return _te_gmem_module

        try:
            import tenex._ext.te_gmem as _mod
            _te_gmem_module = _mod
            return _te_gmem_module
        except ImportError:
            pass

        import torch.utils.cpp_extension as cpp_ext
        _te_gmem_module = cpp_ext.load_inline(
            name='te_gmem',
            cpp_sources='',
            cuda_sources=_CUDA_SRC_GMEM,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _te_gmem_module


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_te_gmem(
    bin_arrs:    torch.Tensor,   # (n_vars, T, K) int32 or uint8 CUDA
    n_bins_arr:  torch.Tensor,   # (n_vars,) int32 CUDA
    pairs:       torch.Tensor,   # (n_overflow, 2) int64 CUDA
    tau:          int = 1,
    block_size:  Optional[int] = None,
) -> np.ndarray:
    """
    Compute TE for overflow pairs using the global-memory kernel.

    Replaces the scatter_add fallback for pairs that exceed the Adaptive-SMEM
    budget. Uses per-pair global-memory cnt3 with SMEM-only marginals, dramatically
    reducing intermediate tensor allocation (~25× memory traffic reduction).

    bin_arrs can be int32 (auto-converted) or uint8 (preferred, 4× smaller).
    All bin values must be in [0, 127] (guaranteed by remap_bins for b_max ≤ 127).
    """
    if pairs.shape[0] == 0:
        return np.empty(0, dtype=np.float32)

    mod = _load_module()

    T = bin_arrs.shape[1]
    K = bin_arrs.shape[2]
    L = T - tau
    N = L * K

    # Ensure C-contiguous layout — CUDA kernel assumes row-major (var, time) strides.
    # bin_arrs from remap_bins/discretize may be Fortran-ordered (stride=(1, N_genes, ...))
    # which would cause raw pointer arithmetic to read from wrong memory locations.
    bin_arrs = bin_arrs.contiguous()

    # Convert bin_arrs to uint8 if needed
    if bin_arrs.dtype == torch.int32:
        _max_val = int(bin_arrs.max().item())
        if _max_val >= 256:
            raise ValueError(
                f"GMem kernel requires bin values < 256 for uint8 packing, "
                f"got max={_max_val}. Use a smaller kp or fewer bins."
            )
        bin_arrs_u8 = bin_arrs.to(torch.uint8).view(bin_arrs.shape[0], -1)
    elif bin_arrs.dtype == torch.uint8:
        bin_arrs_u8 = bin_arrs.view(bin_arrs.shape[0], -1)
    elif bin_arrs.dtype == torch.int8:
        bin_arrs_u8 = bin_arrs.view(torch.uint8).view(bin_arrs.shape[0], -1)
    elif bin_arrs.dtype == torch.int16:
        _max_val = int(bin_arrs.max().item())
        if _max_val >= 256:
            raise ValueError(
                f"GMem kernel requires bin values < 256 for uint8 packing, "
                f"got max={_max_val}. Use a smaller kp or fewer bins."
            )
        bin_arrs_u8 = bin_arrs.to(torch.uint8).view(bin_arrs.shape[0], -1)
    else:
        raise ValueError(f"bin_arrs dtype must be int32/int16/int8/uint8, got {bin_arrs.dtype}")

    # Auto block size
    if block_size is None:
        from tenex.kernels import _next_pow2
        b_max = int(n_bins_arr.max().item())
        block_size = max(128, min(1024, _next_pow2(max(N, 2 * b_max * b_max))))

    # SMEM for marginals (pre-computed on Python side)
    b_max = int(n_bins_arr.max().item())
    WARPS = block_size // 32
    smem_bytes = (b_max * b_max * 4   # cnt2a (b_t*b_t floats)
                + b_max * b_max * 4   # cnt2b (b_t*b_s floats, overestimate with b_t)
                + b_max * 4           # cnt1
                + WARPS * 4)          # warp_buf

    n_pairs = pairs.shape[0]

    # Compute per-pair cnt3 sizes (b_t*b_t*b_s int32 each).
    # Dynamically allocate cnt3 budget from free GPU memory.
    device = bin_arrs_u8.device
    free_bytes, _ = torch.cuda.mem_get_info(device)
    # Reserve 64 MB for PyTorch allocator overhead; rest goes to cnt3 buffer.
    cnt3_budget = max(0, free_bytes - 64 * 1024 * 1024) // 4  # in int32 cells

    # Compute per-pair cnt3 element counts on CPU for cheap batch splitting
    pairs_cpu   = pairs.cpu().numpy()
    n_arr_cpu   = n_bins_arr.cpu().numpy()
    Bt_arr      = n_arr_cpu[pairs_cpu[:, 0]].astype(np.int64)
    Bs_arr      = n_arr_cpu[pairs_cpu[:, 1]].astype(np.int64)
    cnt3_sizes  = Bt_arr * Bt_arr * Bs_arr        # (n_pairs,) int64

    # Check if all fits in one batch
    total_cnt3 = int(cnt3_sizes.sum())
    if total_cnt3 <= cnt3_budget:
        # Single-batch fast path
        cnt3_offsets = mod.compute_cnt3_offsets(pairs, n_bins_arr)
        te_gpu = mod.te_gmem_launch(
            bin_arrs_u8, pairs, n_bins_arr, cnt3_offsets,
            total_cnt3, T, K, tau, block_size, smem_bytes,
        )
        return te_gpu.cpu().numpy()

    # Multi-batch path: accumulate prefix sums and find batch boundaries
    prefix     = np.concatenate([[0], np.cumsum(cnt3_sizes)])
    te_out     = np.empty(n_pairs, dtype=np.float32)
    batch_start = 0
    while batch_start < n_pairs:
        # Find largest batch_end such that prefix[batch_end] - prefix[batch_start] ≤ budget
        # Binary search for the upper bound
        lo, hi = batch_start + 1, n_pairs
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if prefix[mid] - prefix[batch_start] <= cnt3_budget:
                lo = mid
            else:
                hi = mid - 1
        batch_end = lo   # exclusive

        # Handle degenerate case: single pair exceeds budget (very large b_t/b_s)
        if batch_end == batch_start:
            batch_end = batch_start + 1

        pairs_batch  = pairs[batch_start:batch_end]
        batch_total  = int(prefix[batch_end] - prefix[batch_start])
        cnt3_off_b   = mod.compute_cnt3_offsets(pairs_batch, n_bins_arr)
        te_batch     = mod.te_gmem_launch(
            bin_arrs_u8, pairs_batch, n_bins_arr, cnt3_off_b,
            batch_total, T, K, tau, block_size, smem_bytes,
        )
        te_out[batch_start:batch_end] = te_batch.cpu().numpy()
        batch_start = batch_end

    return te_out


# Backward-compatible alias
compute_te_hbm = compute_te_gmem
