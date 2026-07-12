"""
Adaptive-SMEM kernel for Transfer Entropy — per-var asymmetric bin sizing.

Problem solved
--------------
After per-var remap_bins, each variable g has n_per_var[g] unique bin values
in [0, n_per_var[g]).  The global b_max = max(n_per_var) may be large
(e.g. b_max=55 for CeNGEN), making b_max³ too large for shared memory.

BUT: for a pair (target variable i, source variable j):
  - xt+1, xt  ∈ [0, n_i)   where n_i = n_per_var[i]
  - yt        ∈ [0, n_j)   where n_j = n_per_var[j]
  - cnt3 dims: (n_i, n_i, n_j)  → size n_i² × n_j  (NOT global b_max³)

Example (CeNGEN, mean n_i=n_j=24):
  b_max³        = 55³ = 166,375  → 650 KB SMEM  → exceeds the 99 KB opt-in limit
  n_i²×n_j = 24² × 24 = 13,824 → 54 KB SMEM → fits on default 48 KB!

Coverage on RTX A5000 (GA102 Ampere, 99 KB opt-in SMEM):
  84 % of CeNGEN pairs fit in SMEM using asymmetric n_i² × n_j sizing.
  Remaining 16 % fall back to the global-memory kernel (gmem.py) automatically.

Key differences from the Full-SMEM kernel (te_cuda.py)
-------------------------------------------------------
- Accepts n_bins_arr (n_vars,) int32 — per-var bin count from remap_bins
- Each block reads its own b_t = n_bins_arr[target], b_s = n_bins_arr[source]
- SMEM layout uses b_t² × b_s for cnt3 (not global b_max³)
- Phase 1 indexing: xt1v * b_t * b_s + xtv * b_s + ytv
- Phase 2 marginals: loop-based (no tid < B2 conditions) → works for any BLOCK size
- block_size: _next_pow2(N) capped at 1024 (L dominates for large T datasets)
- Separate launch per SMEM group: Python side sorts pairs by smem need,
  launches groups with matching SMEM allocation

Mathematical correctness
------------------------
The asymmetric indexing is equivalent to the Full-SMEM indexing when n_i = n_j = b_max.
TE values are bit-for-bit identical to Full-SMEM for the same data (modulo fast-math
floating-point order, which differs by < 3×10⁻⁷ in practice).
"""


import math
import threading
from typing import Optional

import numpy as np
import torch

from tenex._log import vprint

# ── CUDA source ────────────────────────────────────────────────────────────────

_CUDA_SRC_ADAPTIVE_SMEM = r"""
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
"""

# ── Module loading ─────────────────────────────────────────────────────────────

_te_adaptive_smem_module: object = None
_te_adaptive_smem_lock = threading.Lock()


def _load_module():
    global _te_adaptive_smem_module
    if _te_adaptive_smem_module is not None:
        return _te_adaptive_smem_module

    with _te_adaptive_smem_lock:
        if _te_adaptive_smem_module is not None:
            return _te_adaptive_smem_module

        try:
            import tenex._ext.te_adaptive_smem as _mod
            # Validate that the AOT module has all required functions
            _required = ['te_adaptive_smem_launch', 'te_adaptive_smem_launch_pairfree',
                         'te_adaptive_smem_launch_cross', 'coarsen_bins_cuda',
                         'get_smem_optin']
            if all(hasattr(_mod, f) for f in _required):
                _te_adaptive_smem_module = _mod
                return _te_adaptive_smem_module
        except ImportError:
            pass

        from torch.utils.cpp_extension import load_inline
        _te_adaptive_smem_module = load_inline(
            name='te_adaptive_smem',
            cpp_sources='',
            cuda_sources=_CUDA_SRC_ADAPTIVE_SMEM,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _te_adaptive_smem_module


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_smem_bin_limit(smem_optin: int, warps: int = 32) -> int:
    """
    Find the maximum smem_bin_limit such that a symmetric pair (b_t = b_s = smem_bin_limit)
    fits within the given SMEM budget.

    SMEM formula for the Adaptive-SMEM kernel (symmetric case):
        smem = (B³ + 2B² + B + warps) × 4  bytes

    smem_bin_limit = max B  s.t.  smem(B) ≤ smem_optin

    After applying coarsen_bins() with smem_bin_limit, all pairs are
    guaranteed to fit in SMEM → 0% overflow to global-memory kernel.

    Parameters
    ----------
    smem_optin : int — device SMEM opt-in limit in bytes (from get_smem_optin())
    warps      : int — warps per block (= block_size // 32)

    Returns
    -------
    smem_bin_limit : int ≥ 2
    """
    B = max(2, int((smem_optin / 4) ** (1.0 / 3.0)))   # rough upper bound
    while B > 2 and (B**3 + 2 * B**2 + B + warps) * 4 > smem_optin:
        B -= 1
    return B


def compute_te_adaptive_smem(
    bin_arrs:    torch.Tensor,   # (n_vars, T, K) int32, on GPU
    n_bins_arr:  torch.Tensor,   # (n_vars,) int32, on GPU
    pairs:       torch.Tensor,   # (n_pairs, 2) int64, on GPU
    tau:          int = 1,
    batch_size:  Optional[int] = None,   # unused, kept for API compat
) -> np.ndarray:
    """
    Adaptive-SMEM Transfer Entropy kernel with per-var asymmetric bin sizing.

    For each pair (i, j), uses cnt3 of size n_i² × n_j instead of global b_max³.
    This makes ~84% of CeNGEN pairs feasible on RTX A5000 (99 KB SMEM budget).

    Pairs that exceed the SMEM budget fall back to the global-memory kernel automatically.

    Parameters
    ----------
    bin_arrs   : (n_vars, T, K) int32 on CUDA — already remapped per-var
    n_bins_arr : (n_vars,) int32 on CUDA — unique bin count per variable
    pairs      : (n_pairs, 2) int64 on CUDA
    tau         : time delay

    Returns
    -------
    entropies  : (n_pairs,) float32 numpy array
    """
    device = bin_arrs.device
    T, K   = bin_arrs.shape[1], bin_arrs.shape[2]
    L      = T - tau
    N      = L * K
    n_pairs = pairs.shape[0]

    # block_size: dominated by L for large T, capped at 1024
    block_size = max(128, min(1024, _next_pow2(N)))
    warps      = block_size // 32

    mod = _load_module()

    # Query device SMEM opt-in limit
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block

    # Compute per-pair SMEM requirement from n_bins_arr (CPU-side).
    # smem_pair[p] = (b_t²×b_s + b_t² + b_t×b_s + b_t) × 4 + warps×4
    n_arr = n_bins_arr.cpu().numpy()          # (n_vars,) int32

    # Fast path: if global worst-case fits, skip per-pair computation
    b_max = int(n_arr.max())
    smem_worst = int((b_max**2 * b_max + b_max**2
                      + b_max * b_max + b_max) * 4 + warps * 4)

    entropies_out = np.empty(n_pairs, dtype=np.float32)

    # VRAM-aware: compute max pairs per kernel launch from available VRAM.
    # Each pair needs 4 bytes output + 16 bytes pairs tensor on GPU.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        _reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
        _available = max(0, free_mem - _reserve)
        _max_pairs_per_launch = max(1024, _available // 20)  # 4B output + 16B pairs
    else:
        _max_pairs_per_launch = n_pairs

    def _batched_launch(pairs_to_launch, smem, bin_c, n_bins_c):
        """Launch kernel in VRAM-sized batches, return CPU numpy."""
        n = len(pairs_to_launch)
        if n <= _max_pairs_per_launch:
            p_gpu = (pairs_to_launch if isinstance(pairs_to_launch, torch.Tensor)
                     else torch.from_numpy(pairs_to_launch).to(device))
            te = mod.te_adaptive_smem_launch(
                bin_c, p_gpu.contiguous(), n_bins_c,
                T, K, tau, block_size, smem,
            )
            return te.cpu().numpy()
        # Batched
        results = []
        pairs_np = (pairs_to_launch.cpu().numpy()
                    if isinstance(pairs_to_launch, torch.Tensor)
                    else pairs_to_launch)
        for beg in range(0, n, _max_pairs_per_launch):
            end = min(beg + _max_pairs_per_launch, n)
            p_gpu = torch.from_numpy(pairs_np[beg:end]).to(device)
            te_b = mod.te_adaptive_smem_launch(
                bin_c, p_gpu.contiguous(), n_bins_c,
                T, K, tau, block_size, smem,
            )
            results.append(te_b.cpu().numpy())
            del p_gpu
        return np.concatenate(results)

    bin_c = bin_arrs.contiguous()
    n_bins_c = n_bins_arr.contiguous()

    if smem_worst <= max_smem:
        # All pairs fit in SMEM — launch (with VRAM batching if needed)
        return _batched_launch(pairs, smem_worst, bin_c, n_bins_c)

    # General path: partition pairs by per-pair SMEM requirement
    pairs_cpu = pairs.cpu().numpy()           # (n_pairs, 2) int64
    Bt_arr = n_arr[pairs_cpu[:, 0]]           # target variable bins
    Bs_arr = n_arr[pairs_cpu[:, 1]]           # source variable bins
    smem_pair = ((Bt_arr.astype(np.int64) ** 2 * Bs_arr
                  + Bt_arr ** 2
                  + Bt_arr * Bs_arr
                  + Bt_arr) * 4
                 + warps * 4)

    fits_mask = smem_pair <= max_smem
    n_fits    = int(fits_mask.sum())
    n_over    = n_pairs - n_fits

    if n_fits > 0:
        fits_idx  = np.where(fits_mask)[0]
        fits_smem = smem_pair[fits_mask]

        # Tiered launch: split fits into ≤48 KB (higher occupancy) and >48 KB
        t1_mask_f = fits_smem <= _SMEM_TIER1_BYTES   # within fits group
        t2_mask_f = ~t1_mask_f

        for tier_local, tier_name in [(np.where(t1_mask_f)[0], 'T1'),
                                      (np.where(t2_mask_f)[0], 'T2')]:
            if len(tier_local) == 0:
                continue
            g_idx       = fits_idx[tier_local]           # global pair indices
            tier_smem   = int(fits_smem[tier_local].max())
            te_np = _batched_launch(
                pairs_cpu[g_idx], tier_smem, bin_c, n_bins_c,
            )
            entropies_out[g_idx] = te_np

    if n_over > 0:
        from tenex.kernels.gmem import compute_te_gmem
        over_idx   = np.where(~fits_mask)[0]
        pairs_over = torch.from_numpy(pairs_cpu[over_idx]).to(device)
        te_over    = compute_te_gmem(
            bin_arrs, n_bins_arr, pairs_over, tau=tau
        )
        entropies_out[over_idx] = te_over

    return entropies_out


# Default (non-opt-in) SMEM per block on all CUDA devices.
# Pairs whose SMEM need is ≤ this limit allow ≥2 blocks/SM (higher occupancy).
_SMEM_TIER1_BYTES = 48 * 1024   # 48 KB


def compute_smem_partition(
    n_bins_arr:  np.ndarray,   # (n_vars,) int32 CPU numpy
    pairs_np:    np.ndarray,   # (n_pairs, 2) int64 CPU numpy
    T:           int,
    tau:          int = 1,
    max_smem:    Optional[int] = None,   # bytes; queried from device if None
    device_id:   int = 0,
) -> dict:
    """
    Pre-compute SMEM partitioning for multi-GPU dispatch.

    Returns a dict with:
      fits_idx    : (n_fits,)  int64 — row indices into pairs_np that fit in SMEM
      over_idx    : (n_over,)  int64 — indices that exceed SMEM (global-memory kernel fallback)
      tier1_idx   : (n_t1,)   int64 — fits ≤ 48 KB (potentially 2 blocks/SM)
      tier2_idx   : (n_t2,)   int64 — fits > 48 KB, ≤ max_smem (1 block/SM)
      smem_max    : int              — max SMEM bytes for the entire fits group
      smem_max_t1 : int              — max SMEM bytes for tier-1 group
      smem_max_t2 : int              — max SMEM bytes for tier-2 group
      block_size  : int
      max_smem    : int              — device opt-in limit used
      b_max    : int              — max(n_bins_arr)
    """
    L  = T - tau
    K  = 1  # te_cuda_v4 always K=1 in practice; generalised below
    N  = L * K
    block_size = max(128, min(1024, _next_pow2(N)))
    warps = block_size // 32

    if max_smem is None:
        mod = _load_module()
        prev_dev = torch.cuda.current_device()
        try:
            torch.cuda.set_device(device_id)
            max_smem = int(mod.get_smem_optin())
        finally:
            torch.cuda.set_device(prev_dev)
        if max_smem <= 0:
            max_smem = 48 * 1024  # default 48 KB fallback

    b_t = n_bins_arr[pairs_np[:, 0]].astype(np.int64)
    b_s = n_bins_arr[pairs_np[:, 1]].astype(np.int64)
    smem_pair = (b_t**2 * b_s + b_t**2 + b_t * b_s + b_t) * 4 + warps * 4

    fits_mask  = smem_pair <= max_smem
    tier1_mask = smem_pair <= _SMEM_TIER1_BYTES
    tier2_mask = fits_mask & ~tier1_mask

    fits_idx  = np.where(fits_mask)[0]
    over_idx  = np.where(~fits_mask)[0]
    tier1_idx = np.where(tier1_mask)[0]
    tier2_idx = np.where(tier2_mask)[0]

    smem_max_fits = int(smem_pair[fits_idx].max())  if len(fits_idx)  > 0 else 0
    smem_max_t1   = int(smem_pair[tier1_idx].max()) if len(tier1_idx) > 0 else 0
    smem_max_t2   = int(smem_pair[tier2_idx].max()) if len(tier2_idx) > 0 else 0

    return dict(
        fits_idx    = fits_idx,
        over_idx    = over_idx,
        tier1_idx   = tier1_idx,
        tier2_idx   = tier2_idx,
        smem_max    = smem_max_fits,
        smem_max_t1 = smem_max_t1,
        smem_max_t2 = smem_max_t2,
        block_size  = block_size,
        max_smem    = max_smem,
        b_max    = int(n_bins_arr.max()),
    )


def compute_te_adaptive_smem_prepartitioned(
    bin_arrs:    torch.Tensor,   # (n_vars, T, K) int32, already on GPU
    n_bins_arr:  torch.Tensor,   # (n_vars,) int32, already on GPU
    pairs_fits:  torch.Tensor,   # (n_fits, 2) int64, already on GPU
    pairs_over:  torch.Tensor,   # (n_over, 2) int64, already on GPU
    fits_idx:    np.ndarray,     # (n_fits,) original row indices
    over_idx:    np.ndarray,     # (n_over,) original row indices
    n_total:     int,            # total number of pairs
    block_size:  int,
    smem_max:    int,            # max SMEM bytes for fits group
    b_max:    int,
    tau:          int = 1,
) -> np.ndarray:
    """
    GPU-only transfer entropy computation with pre-computed pair partitioning.

    Skips the Python-side partitioning overhead (already done by caller).
    Designed for multi-GPU dispatch where partitioning is computed once.
    """
    mod = _load_module()
    T, K = bin_arrs.shape[1], bin_arrs.shape[2]

    entropies_out = np.empty(n_total, dtype=np.float32)

    if len(fits_idx) > 0:
        te_fits = mod.te_adaptive_smem_launch(
            bin_arrs.contiguous(),
            pairs_fits.contiguous(),
            n_bins_arr.contiguous(),
            T, K, tau, block_size, smem_max
        )
        entropies_out[fits_idx] = te_fits.cpu().numpy()

    if len(over_idx) > 0:
        from tenex.kernels.gmem import compute_te_gmem
        te_over = compute_te_gmem(bin_arrs, n_bins_arr, pairs_over, tau=tau)
        entropies_out[over_idx] = te_over

    return entropies_out


def smem_coverage(n_bins_arr: np.ndarray, pairs_np: np.ndarray,
                  tau: int = 1, T: int = 0) -> dict:
    """
    Estimate SMEM coverage stats for a set of variable pairs.

    Useful for benchmarking before launching a full run.

    Parameters
    ----------
    n_bins_arr : (n_vars,) int32 — from discretize()
    pairs_np   : (n_pairs, 2) int — target/source variable indices
    T          : number of time points (for warp_buf calculation)

    Returns
    -------
    dict with keys: pct_48kb, pct_99kb, pct_228kb, smem_mean_kb, smem_median_kb
    """
    _N = (T - tau) if T > 0 else 1000
    bs = max(128, min(1024, _next_pow2(_N)))
    warps = bs // 32

    b_t = n_bins_arr[pairs_np[:, 0]].astype(np.int64)
    b_s = n_bins_arr[pairs_np[:, 1]].astype(np.int64)
    smem = (b_t**2 * b_s + b_t**2 + b_t * b_s + b_t) * 4 + warps * 4

    return dict(
        pct_48kb   = float((smem <= 48  * 1024).mean() * 100),
        pct_99kb   = float((smem <= 99  * 1024).mean() * 100),
        pct_228kb  = float((smem <= 228 * 1024).mean() * 100),
        smem_mean_kb   = float(smem.mean() / 1024),
        smem_median_kb = float(np.median(smem) / 1024),
        smem_max_kb    = float(smem.max() / 1024),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pair-free API
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_adaptive_smem_pairfree(
    bin_arrs:      torch.Tensor,   # (n_vars, T, K) on GPU
    n_bins_arr:    torch.Tensor,   # (n_vars,) int32 on GPU
    n_vars:       int,
    tau:            int = 1,
    pair_offset:   int = 0,
    n_pairs_local: Optional[int] = None,
) -> np.ndarray:
    """
    Pair-free Adaptive-SMEM kernel.

    Computes (target, source) from linear pair index, eliminating the
    need for an explicit pairs tensor.  After bin coarsening, all pairs
    fit in SMEM, so no overflow/GMem fallback is needed.
    """
    if n_pairs_local is None:
        n_pairs_local = n_vars * (n_vars - 1) - pair_offset

    device = bin_arrs.device
    T, K = bin_arrs.shape[1], bin_arrs.shape[2]
    L = T - tau
    N = L * K

    block_size = max(128, min(1024, _next_pow2(N)))
    warps = block_size // 32

    mod = _load_module()
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block

    # After coarsening, worst-case SMEM is guaranteed to fit
    n_arr = n_bins_arr.cpu().numpy()
    b_max = int(n_arr.max())
    smem_worst = int((b_max**2 * b_max + b_max**2
                      + b_max * b_max + b_max) * 4 + warps * 4)
    if smem_worst > max_smem:
        raise RuntimeError(
            f"Adaptive-SMEM pair-free requires coarsening: smem_worst={smem_worst} > "
            f"max_smem={max_smem}. Ensure prepare() with coarsening is called first."
        )

    bin_arrs_c = bin_arrs.contiguous()
    n_bins_c = n_bins_arr.contiguous()
    MAX_GRID = 2**31 - 1

    # VRAM-aware chunking: output tensor = n_pairs_local * 4 bytes
    max_chunk = MAX_GRID
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        _reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
        _available = max(0, free_mem - _reserve)
        vram_chunk = max(1024, _available // 4)
        max_chunk = min(max_chunk, vram_chunk)

    if n_pairs_local <= max_chunk:
        te = mod.te_adaptive_smem_launch_pairfree(
            bin_arrs_c, n_bins_c,
            T, K, tau, block_size, smem_worst,
            n_vars, pair_offset, n_pairs_local,
        )
        return te.cpu().numpy()

    # Chunked launch
    chunks = []
    remaining = n_pairs_local
    cur_offset = pair_offset
    while remaining > 0:
        sz = min(remaining, max_chunk)
        te_chunk = mod.te_adaptive_smem_launch_pairfree(
            bin_arrs_c, n_bins_c,
            T, K, tau, block_size, smem_worst,
            n_vars, cur_offset, sz,
        )
        chunks.append(te_chunk.cpu())
        cur_offset += sz
        remaining -= sz
    return torch.cat(chunks).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Cross pair-free API (variable blocking)
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_adaptive_smem_cross(
    bin_arrs:      torch.Tensor,   # ((n_tgt + n_src), T, K) on GPU
    n_bins_arr:    torch.Tensor,   # (n_tgt + n_src,) int32 on GPU
    n_tgt:         int,
    n_src:         int,
    tau:            int = 1,
    pair_offset:   int = 0,
    n_pairs_local: Optional[int] = None,
) -> np.ndarray:
    """
    Cross pair-free Adaptive-SMEM kernel for variable blocking.

    Computes TE between two disjoint variable sets without an explicit pairs tensor.
    bin_arrs layout: first n_tgt rows are targets, last n_src rows are sources.
    pair_id = tgt_local * n_src + src_local.
    """
    total_pairs = n_tgt * n_src
    if n_pairs_local is None:
        n_pairs_local = total_pairs - pair_offset

    device = bin_arrs.device
    T, K = bin_arrs.shape[1], bin_arrs.shape[2]
    L = T - tau
    N = L * K

    block_size = max(128, min(1024, _next_pow2_local(N)))
    warps = block_size // 32

    mod = _load_module()
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block

    n_arr = n_bins_arr.cpu().numpy()
    b_max = int(n_arr.max())
    smem_worst = int((b_max**2 * b_max + b_max**2
                      + b_max * b_max + b_max) * 4 + warps * 4)
    if smem_worst > max_smem:
        raise RuntimeError(
            f"Adaptive-SMEM cross requires coarsening: smem_worst={smem_worst} > "
            f"max_smem={max_smem}. Ensure prepare() with coarsening is called first."
        )

    bin_arrs_c = bin_arrs.contiguous()
    n_bins_c = n_bins_arr.contiguous()
    MAX_GRID = 2**31 - 1

    # VRAM-aware chunking
    max_chunk = MAX_GRID
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        _reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
        _available = max(0, free_mem - _reserve)
        vram_chunk = max(1024, _available // 4)
        max_chunk = min(max_chunk, vram_chunk)

    if n_pairs_local <= max_chunk:
        te = mod.te_adaptive_smem_launch_cross(
            bin_arrs_c, n_bins_c,
            T, K, tau, block_size, smem_worst,
            n_tgt, n_src, pair_offset, n_pairs_local,
        )
        return te.cpu().numpy()

    # Chunked launch
    chunks = []
    remaining = n_pairs_local
    cur_offset = pair_offset
    while remaining > 0:
        sz = min(remaining, max_chunk)
        te_chunk = mod.te_adaptive_smem_launch_cross(
            bin_arrs_c, n_bins_c,
            T, K, tau, block_size, smem_worst,
            n_tgt, n_src, cur_offset, sz,
        )
        chunks.append(te_chunk.cpu())
        cur_offset += sz
        remaining -= sz
    return torch.cat(chunks).numpy()


def _next_pow2_local(v):
    if v <= 1:
        return 1
    v -= 1
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16
    return v + 1


# ─────────────────────────────────────────────────────────────────────────────
# TEKernel interface
# ─────────────────────────────────────────────────────────────────────────────

from tenex.kernels import TEKernel
from tenex.kernels import _next_pow2


class AdaptiveSMEMKernel(TEKernel):
    """Adaptive-SMEM kernel: per-var asymmetric cnt3 in SMEM, any b_max."""

    @property
    def name(self) -> str:
        return "Adaptive-SMEM"

    @property
    def supports_pairfree(self) -> bool:
        return True

    def peak_bytes_per_pair(self, b_max, T, K, tau=1):
        # SMEM kernel: only output (4 B) in global memory for pair-free
        return 4

    def supports(self, b_max, on_cuda, smem_optin, smem_bytes,
                 n_per_var, source_filter) -> bool:
        return on_cuda  # handles any b_max on CUDA

    def prepare(self, bin_arrs, n_per_var, b_max, **kwargs):
        """Apply bin coarsening if needed for Adaptive-SMEM kernel."""
        use_coarsening = kwargs.get('use_coarsening')
        on_cuda = kwargs.get('on_cuda', True)
        n_vars = kwargs.get('n_vars', bin_arrs.shape[0])
        tau = kwargs.get('tau', 1)

        if not on_cuda:
            return bin_arrs, n_per_var, b_max

        # Prefer the caller-supplied opt-in limit (the minimum across all
        # selected GPUs) so heterogeneous multi-GPU runs plan for the weakest
        # device. Fall back to the current device only when it is not provided.
        _smem_optin = kwargs.get('smem_optin')
        if _smem_optin is None or _smem_optin <= 0:
            _smem_optin = int(_load_module().get_smem_optin())

        L = bin_arrs.shape[1] - tau
        K = bin_arrs.shape[2]
        N = L * K

        _apply = use_coarsening is not False  # True or None → apply
        if not _apply:
            # gmem overflow path packs bins into uint8, which requires
            # b_max < 256.  Detect this up-front so the user sees a
            # clear failure instead of a deep CUDA RuntimeError.
            if b_max >= 256:
                raise ValueError(
                    f"coarsening=False but b_max={b_max} >= 256, which the "
                    f"global-memory overflow kernel cannot pack into uint8. "
                    f"Use coarsening=None (default) or kp<0.5 to reduce bins."
                )
            # The overflow kernel offloads only the 3-D joint counts to global
            # memory. The 2-D and 1-D marginals still live in shared memory,
            # sized for the worst-case pair (global b_max). Validate that
            # footprint against the device limit so a large b_max does not
            # produce an illegal-launch CUDA error at kernel dispatch.
            _gm_block = max(128, min(1024, _next_pow2(max(N, 2 * b_max * b_max))))
            _gm_warps = _gm_block // 32
            _gm_smem = (b_max * b_max * 4        # cnt2a
                        + b_max * b_max * 4      # cnt2b
                        + b_max * 4              # cnt1
                        + _gm_warps * 4)         # warp_buf
            if _gm_smem > _smem_optin:
                raise ValueError(
                    f"coarsening=False but the global-memory overflow kernel "
                    f"needs {_gm_smem} B of shared memory for the b_max={b_max} "
                    f"marginals, which exceeds the device opt-in limit "
                    f"{_smem_optin} B. Use coarsening=None (default) or kp<0.5 "
                    f"to reduce bins."
                )
            vprint("[TENEX] bin coarsening disabled: using Adaptive-SMEM + global-memory overflow")
            return bin_arrs, n_per_var, b_max

        _warps = max(128, min(1024, _next_pow2(max(N, 1)))) // 32
        _smem_bin_limit = compute_smem_bin_limit(_smem_optin, warps=_warps)

        if b_max > _smem_bin_limit:
            from tenex.preprocess import coarsen_bins
            _n_coarsened = int((n_per_var > _smem_bin_limit).sum())
            vprint(f"[TENEX] bin coarsening: b_max={b_max}->{_smem_bin_limit} "
                  f"({_n_coarsened}/{n_vars} genes coarsened, "
                  f"smem_optin={_smem_optin // 1024} KB -> 0% overflow)")
            # No clone needed: caller (tenex.py compute()) already passes clones
            # to prepare(), so in-place modification by coarsen_bins is safe.
            bin_arrs, n_per_var, b_max = coarsen_bins(
                bin_arrs, n_per_var, _smem_bin_limit
            )

        return bin_arrs, n_per_var, b_max

    def peak_batch_bytes(self, b_max, T, K, batch_size, tau=1):
        # Adaptive-SMEM uses shared memory for cnt3; global memory is
        # only the output TE tensor (batch * 4 bytes) + pair indices.
        return batch_size * 4 + batch_size * 2 * 8  # te_out + pairs

    def compute_single_gpu(self, bin_arrs, pairs, b_max, n_per_var,
                           tau, batch_size, device, **kwargs):
        return compute_te_adaptive_smem(bin_arrs, n_per_var, pairs, tau=tau)

    def compute_pairfree(self, bin_arrs, n_vars, b_max, n_per_var,
                         tau, device, pair_offset=0, n_pairs_local=None):
        """Single-GPU pair-free computation."""
        return compute_te_adaptive_smem_pairfree(
            bin_arrs, n_per_var, n_vars, tau=tau,
            pair_offset=pair_offset, n_pairs_local=n_pairs_local,
        )

    def compute_pairfree_cross(self, bin_arrs, n_per_var,
                               n_tgt, n_src, tau, device,
                               pair_offset=0, n_pairs_local=None):
        """Cross pair-free: TE between two disjoint variable blocks.

        bin_arrs: ((n_tgt + n_src), T, K) on GPU
        n_per_var: (n_tgt + n_src,) int32 on GPU
        """
        return compute_te_adaptive_smem_cross(
            bin_arrs, n_per_var, n_tgt, n_src, tau=tau,
            pair_offset=pair_offset, n_pairs_local=n_pairs_local,
        )

    def compute_multi_gpu(self, bin_arrs_cpu, pairs_np, b_max,
                          n_per_var_cpu, tau, batch_size, device_ids,
                          **kwargs):
        """Tiered multi-GPU dispatch with pre-partitioning for Adaptive-SMEM."""
        import math
        from concurrent.futures import ThreadPoolExecutor

        n_gpus = len(device_ids)
        n_pairs = len(pairs_np)

        n_arr_cpu = n_per_var_cpu.numpy()
        # Partition using the minimum opt-in across the selected GPUs (passed
        # down from compute()) so a heterogeneous set plans for the weakest
        # device. Fall back to querying device_ids[0] when not supplied.
        _min_smem = kwargs.get('smem_optin')
        if _min_smem is not None and _min_smem <= 0:
            _min_smem = None
        part = compute_smem_partition(
            n_arr_cpu, pairs_np,
            T=bin_arrs_cpu.shape[1], tau=tau,
            max_smem=_min_smem,
            device_id=device_ids[0],
        )
        tier1_idx_g = part['tier1_idx']
        tier2_idx_g = part['tier2_idx']
        over_idx_g = part['over_idx']
        smem_max_t1 = part['smem_max_t1']
        smem_max_t2 = part['smem_max_t2']
        blk_sz = part['block_size']
        _part_b_max = part['b_max']
        smem_max_f = part['smem_max']

        # Pre-transfer bin_arrs to all GPUs in parallel
        bin_per_gpu = {}
        npg_per_gpu = {}

        def _transfer(dev_id):
            dev = torch.device(f'cuda:{dev_id}')
            torch.cuda.set_device(dev_id)
            bin_per_gpu[dev_id] = bin_arrs_cpu.to(dev)
            npg_per_gpu[dev_id] = n_per_var_cpu.to(dev)
            torch.cuda.synchronize(dev)

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            list(pool.map(_transfer, device_ids))

        # GPU pre-warming
        if len(part['fits_idx']) > 0:
            _smem_warmup = max(smem_max_f, 1)

            def _warmup(dev_id):
                dev = torch.device(f'cuda:{dev_id}')
                torch.cuda.set_device(dev_id)
                dummy_fits = torch.zeros((1, 2), dtype=torch.int64, device=dev)
                dummy_over = torch.empty((0, 2), dtype=torch.int64, device=dev)
                compute_te_adaptive_smem_prepartitioned(
                    bin_per_gpu[dev_id], npg_per_gpu[dev_id],
                    dummy_fits, dummy_over,
                    np.array([0], dtype=np.int64),
                    np.array([], dtype=np.int64),
                    1, blk_sz, _smem_warmup, _part_b_max, tau=tau,
                )
                torch.cuda.synchronize(dev)

            with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                list(pool.map(_warmup, device_ids))

        # Split tiers across GPUs
        tier1_chunks = np.array_split(tier1_idx_g, n_gpus)
        tier2_chunks = np.array_split(tier2_idx_g, n_gpus)
        over_chunks = np.array_split(over_idx_g, n_gpus)
        _empty_idx = np.empty(0, dtype=np.int64)
        results = [None] * n_gpus

        def _run_gpu(rank):
            dev_id = device_ids[rank]
            torch.cuda.set_device(dev_id)
            dev = torch.device(f'cuda:{dev_id}')

            t1_local = tier1_chunks[rank]
            t2_local = tier2_chunks[rank]
            ov_local = over_chunks[rank]
            n_local = len(t1_local) + len(t2_local) + len(ov_local)

            if n_local == 0:
                return rank, np.empty(0, dtype=np.float32), t1_local, t2_local, ov_local

            arr_d = bin_per_gpu[dev_id]
            npg_d = npg_per_gpu[dev_id]
            ents = np.empty(n_local, dtype=np.float32)
            off = 0
            empty_over_d = torch.empty((0, 2), dtype=torch.int64, device=dev)

            # Bound each tier launch by the free memory on this device. The
            # working set per pair is the output float plus the (source,target)
            # int64 index pair, matching peak_batch_bytes(); a very large
            # filtered pair set would otherwise allocate it all at once and OOM.
            free_b, _ = torch.cuda.mem_get_info(dev)
            _peak_pp = 4 + 2 * 8  # te_out (float32) + pair indices (2 x int64)
            _reserve = max(256 * 1024 * 1024, int(free_b * 0.2))
            _tier_bs = max(1, int((free_b - _reserve) // _peak_pp))

            def _run_tier(local_idx, smem_max_tier):
                nonlocal off
                for _s in range(0, len(local_idx), _tier_bs):
                    sl = local_idx[_s:_s + _tier_bs]
                    p = torch.from_numpy(pairs_np[sl].astype(np.int64)).to(dev)
                    te_b = compute_te_adaptive_smem_prepartitioned(
                        arr_d, npg_d, p, empty_over_d,
                        np.arange(len(sl), dtype=np.int64), _empty_idx,
                        len(sl), blk_sz, smem_max_tier, _part_b_max, tau=tau,
                    )
                    ents[off:off + len(sl)] = te_b
                    off += len(sl)
                    del p

            if len(t1_local) > 0 and smem_max_t1 > 0:
                _run_tier(t1_local, smem_max_t1)

            if len(t2_local) > 0 and smem_max_t2 > 0:
                _run_tier(t2_local, smem_max_t2)

            if len(ov_local) > 0:
                from tenex.kernels.gmem import compute_te_gmem
                p = torch.from_numpy(pairs_np[ov_local].astype(np.int64)).to(dev)
                ov_te = compute_te_gmem(arr_d, npg_d, p, tau=tau)
                ents[off:off + len(ov_local)] = ov_te

            return rank, ents, t1_local, t2_local, ov_local

        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {pool.submit(_run_gpu, i): i for i in range(n_gpus)}
            for fut in as_completed(futures):
                rank, ents, t1, t2, ov = fut.result()
                results[rank] = (ents, t1, t2, ov)

        # Gather results in original pair order
        entropies = np.empty(n_pairs, dtype=np.float32)
        for rank in range(n_gpus):
            if results[rank] is None:
                continue
            ents, t1, t2, ov = results[rank]
            off = 0
            if len(t1) > 0:
                entropies[t1] = ents[off:off + len(t1)]; off += len(t1)
            if len(t2) > 0:
                entropies[t2] = ents[off:off + len(t2)]; off += len(t2)
            if len(ov) > 0:
                entropies[ov] = ents[off:off + len(ov)]
        return entropies
