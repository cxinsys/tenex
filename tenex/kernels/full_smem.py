"""
Full-SMEM kernel for Transfer Entropy.

Design
------
The entire 3-D joint histogram (cnt3) is stored in CUDA shared memory.
No global-memory traffic for cnt3 at all — giving maximum throughput.

Applicable when b_max³ ≤ ~65 536 (i.e. b_max ≤ ~40 after remap_bins).
For larger b_max (e.g. CeNGEN b_max≈55), use the Adaptive-SMEM kernel
(te_cuda_v4.py) instead.

Direct bin_arrs access
  bin_arrs is passed directly to the CUDA kernel via pair variable indices.
  Eliminates ~5 GB/batch of intermediate xt1/xt/yt tensor allocation.

  Memory pattern (per pair, b_max=7):
    Input:  bin_arrs (≈6 MB, L2-cached)     — shared across all pairs
    Shared: cnt3(1.37 KB) + cnt2a(196 B) + cnt2b(196 B) + cnt1(28 B) + warp_buf
    Output: 1 float (4 B)

Key innovation over two-stage Triton approach:
  Stage 1 (count_kernel) writes cnt3 → global memory (846 MB)
  Stage 2 (_te_from_cnt3) reads cnt3 ← global memory (846 MB × 5 passes)

  Full-SMEM kernel: cnt3 lives entirely in SHARED MEMORY (1.37 KB per block).
  No global-memory traffic for cnt3 at all.
"""


import math
import threading
from typing import Optional

import numpy as np
import torch

# ── CUDA source ───────────────────────────────────────────────────────────────

_CUDA_SRC = r"""
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
"""

# ── Module loading (compile once, cache) ─────────────────────────────────────

_te_smem_module: object = None
_te_smem_lock = threading.Lock()


def _load_module():
    global _te_smem_module
    if _te_smem_module is not None:
        return _te_smem_module

    with _te_smem_lock:
        if _te_smem_module is not None:
            return _te_smem_module

        # Try AOT-compiled extension first (from pip-installed wheel)
        try:
            import tenex._ext.te_smem as _mod
            _te_smem_module = _mod
            return _te_smem_module
        except ImportError:
            pass

        # Fall back to JIT compilation (requires CUDA toolkit + nvcc)
        from torch.utils.cpp_extension import load_inline
        _te_smem_module = load_inline(
            name='te_smem',
            cpp_sources='',
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _te_smem_module


# ── Public API ───────────────────────────────────────────────────────────────

def compute_te_smem(
    bin_arrs:   torch.Tensor,   # (n_vars, T, K) int32, on GPU
    pairs:      torch.Tensor,   # (n_pairs, 2)    int64, on GPU
    b_max:   int,
    tau:         int = 1,
    batch_size: Optional[int] = None,   # kept for API compatibility
) -> np.ndarray:
    """
    Full-SMEM Transfer Entropy kernel.

    The entire 3-D joint histogram cnt3[b_max³] lives in shared memory.
    No global-memory traffic for cnt3 — maximum throughput for small b_max.

    bin_arrs is passed directly to the CUDA kernel via pairs[] variable indices.
    No intermediate xt1/xt/yt tensors are allocated. bin_arrs (~6 MB for
    typical datasets) fits in GPU L2 cache, giving high cache-reuse across
    pairs that share target/source variables.

    Condition: b_max³ must fit in device SMEM. This is guaranteed when
    b_max ≤ ~40 (after remap_bins). For larger b_max use
    compute_te_adaptive_smem() in te_cuda_v4.py instead.

    Parameters
    ----------
    bin_arrs  : (n_vars, T, K) int32 on CUDA
    pairs     : (n_pairs, 2) int64 on CUDA
    b_max  : global max unique bins per variable (from preprocess.discretize)
    tau        : time delay
    batch_size: ignored (kept for API compat; no batching needed)

    Returns
    -------
    entropies : (n_pairs,) float32 numpy array
    """
    device  = bin_arrs.device
    T, K    = bin_arrs.shape[1], bin_arrs.shape[2]
    L       = T - tau
    N       = L * K
    b2      = b_max * b_max
    b3      = b_max * b_max * b_max

    # Thread block size: must be power of 2, >= max(N, 2*b2), <= 1024
    block_size = max(128, min(1024, _next_pow2(max(N, 2 * b2))))

    # Shared memory requirement per block (bytes)
    warps      = block_size // 32
    smem_bytes = b3 * 4 + b2 * 4 * 2 + b_max * 4 + warps * 4

    # Load module first so we can query the opt-in SMEM limit.
    # te_smem_launch calls cudaFuncSetAttribute to raise the limit before launch.
    mod = _load_module()

    # Query the maximum configurable dynamic SMEM for the current device.
    # Default hardware limit: 48 KB.  With opt-in: 99 KB (RTX A5000/4090 and
    # PRO 6000 Blackwell workstation), ~163 KB (A100), ~228 KB (data-center
    # Blackwell).  Falls back to 48 KB on failure.
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block

    if smem_bytes > max_smem:
        # Fallback: scatter_add path (always available, no triton dependency)
        from tenex.kernels.scatter_add import compute_te_scatter_add
        return compute_te_scatter_add(bin_arrs, pairs, b_max, tau=tau, batch_size=batch_size)

    bin_arrs_c = bin_arrs.contiguous()
    n_pairs = pairs.shape[0]

    # VRAM-aware batching: output tensor = n_pairs * 4 bytes.
    # On small-VRAM GPUs (e.g. 2080 Ti), large pair counts can OOM.
    if batch_size is None and device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
        available = max(0, free_mem - reserve)
        # Each pair needs: 4 bytes output + 16 bytes pairs tensor (already on GPU)
        # Only output is allocated by the kernel launch
        max_pairs = max(1024, available // 4)
    else:
        max_pairs = n_pairs

    if n_pairs <= max_pairs:
        te = mod.te_smem_launch(
            bin_arrs_c, pairs.contiguous(),
            T, K, tau, b_max, b2, b3, block_size,
        )
        return te.cpu().numpy()

    # Batched launch: process pairs in VRAM-sized chunks
    chunks = []
    for beg in range(0, n_pairs, max_pairs):
        end = min(beg + max_pairs, n_pairs)
        te_chunk = mod.te_smem_launch(
            bin_arrs_c, pairs[beg:end].contiguous(),
            T, K, tau, b_max, b2, b3, block_size,
        )
        chunks.append(te_chunk.cpu())
    return torch.cat(chunks).numpy()


def compute_te_smem_pairfree(
    bin_arrs:      torch.Tensor,   # (n_vars, T, K) int32, on GPU
    b_max:      int,
    n_vars:       int,
    tau:            int = 1,
    pair_offset:   int = 0,
    n_pairs_local: Optional[int] = None,
) -> np.ndarray:
    """
    Pair-free Full-SMEM kernel — computes (tgt, src) from linear pair index.

    Eliminates the need for an explicit pairs tensor, removing:
    - _build_pairs() CPU overhead (~15s for 25K genes)
    - pairs H2D transfer (~5-10 GB)
    - pairs GPU memory allocation

    Parameters
    ----------
    bin_arrs      : (n_vars, T, K) int32 on CUDA
    b_max      : global max unique bins per variable
    n_vars       : number of genes (for index computation)
    tau            : time delay
    pair_offset   : starting global pair index (for multi-GPU splitting)
    n_pairs_local : number of pairs this call processes (None = all)

    Returns
    -------
    entropies : (n_pairs_local,) float32 numpy array
    """
    if n_pairs_local is None:
        n_pairs_local = n_vars * (n_vars - 1) - pair_offset

    device = bin_arrs.device
    T, K = bin_arrs.shape[1], bin_arrs.shape[2]
    L = T - tau
    N = L * K
    b2 = b_max * b_max
    b3 = b_max * b_max * b_max

    block_size = max(128, min(1024, _next_pow2(max(N, 2 * b2))))
    warps = block_size // 32
    smem_bytes = b3 * 4 + b2 * 4 * 2 + b_max * 4 + warps * 4

    mod = _load_module()
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block

    if smem_bytes > max_smem:
        raise RuntimeError(
            f"Full-SMEM pair-free requires {smem_bytes} bytes SMEM but device max is {max_smem}"
        )

    bin_arrs_c = bin_arrs.contiguous()
    MAX_GRID = 2**31 - 1  # CUDA grid dimension limit

    # VRAM-aware chunking: output tensor = n_pairs_local * 4 bytes.
    # On small-VRAM GPUs this can exceed available memory.
    max_chunk = MAX_GRID
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        _reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
        _available = max(0, free_mem - _reserve)
        vram_chunk = max(1024, _available // 4)  # 4 bytes per output float
        max_chunk = min(max_chunk, vram_chunk)

    if n_pairs_local <= max_chunk:
        te = mod.te_smem_launch_pairfree(
            bin_arrs_c,
            T, K, tau, b_max, b2, b3, block_size,
            n_vars, pair_offset, n_pairs_local,
        )
        return te.cpu().numpy()

    # Chunked launch
    chunks = []
    remaining = n_pairs_local
    cur_offset = pair_offset
    while remaining > 0:
        sz = min(remaining, max_chunk)
        te_chunk = mod.te_smem_launch_pairfree(
            bin_arrs_c,
            T, K, tau, b_max, b2, b3, block_size,
            n_vars, cur_offset, sz,
        )
        chunks.append(te_chunk.cpu())
        cur_offset += sz
        remaining -= sz
    return torch.cat(chunks).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# TEKernel interface
# ─────────────────────────────────────────────────────────────────────────────

from tenex.kernels import TEKernel
from tenex.kernels import _next_pow2


class FullSMEMKernel(TEKernel):
    """Full-SMEM kernel: cnt3 entirely in shared memory — fastest for small b_max."""

    @property
    def name(self) -> str:
        return "Full-SMEM"

    @property
    def supports_pairfree(self) -> bool:
        return True

    def peak_bytes_per_pair(self, b_max, T, K, tau=1):
        # SMEM kernel: only output (4 B) + pairs tensor (16 B) in global memory
        return 20

    def supports(self, b_max, on_cuda, smem_optin, smem_bytes,
                 n_per_var, source_filter) -> bool:
        if not on_cuda:
            return False
        b3 = b_max ** 3
        return b3 <= 65536 and smem_bytes <= smem_optin

    def peak_batch_bytes(self, b_max, T, K, batch_size, tau=1):
        # Full-SMEM uses shared memory for cnt3; global memory is
        # only the output TE tensor + pair indices.
        return batch_size * 4 + batch_size * 2 * 8  # te_out + pairs

    def compute_single_gpu(self, bin_arrs, pairs, b_max, n_per_var,
                           tau, batch_size, device, **kwargs):
        return compute_te_smem(bin_arrs, pairs, b_max, tau=tau, batch_size=batch_size)

    def compute_pairfree(self, bin_arrs, n_vars, b_max, n_per_var,
                         tau, device, pair_offset=0, n_pairs_local=None):
        """Single-GPU pair-free computation."""
        return compute_te_smem_pairfree(
            bin_arrs, b_max, n_vars, tau=tau,
            pair_offset=pair_offset, n_pairs_local=n_pairs_local,
        )

    def compute_pairfree_multi_gpu(self, bin_arrs_cpu, n_vars, b_max,
                                   n_per_var_cpu, tau, device_ids):
        """Multi-GPU pair-free with pinned memory + overlapped transfer/compute."""
        import math
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tenex.kernels import _try_pin

        n_gpus = len(device_ids)
        n_pairs = n_vars * (n_vars - 1)
        chunk = math.ceil(n_pairs / n_gpus)

        # Pin CPU tensor for async H2D transfer
        bin_pinned = _try_pin(bin_arrs_cpu)

        results = [None] * n_gpus

        def _run(rank):
            dev_id = device_ids[rank]
            torch.cuda.set_device(dev_id)
            dev = torch.device(f'cuda:{dev_id}')
            offset = rank * chunk
            n_local = min(chunk, n_pairs - offset)
            if n_local <= 0:
                return rank, np.empty(0, dtype=np.float32)

            # Async transfer from pinned memory
            bin_d = bin_pinned.to(dev, non_blocking=True)
            torch.cuda.synchronize(dev)  # ensure transfer done before compute

            ents = compute_te_smem_pairfree(
                bin_d, b_max, n_vars, tau=tau,
                pair_offset=offset, n_pairs_local=n_local,
            )
            return rank, ents

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {pool.submit(_run, i): i for i in range(n_gpus)}
            for fut in as_completed(futures):
                rank, ents = fut.result()
                results[rank] = ents

        return np.concatenate([r for r in results if r is not None and len(r) > 0])
