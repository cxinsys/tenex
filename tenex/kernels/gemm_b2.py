"""
GEMM-based exact TE for binary (b_max=2) data.

For binary data (bin_arrs values ∈ {0, 1}), all n² pair-wise TE values
can be computed simultaneously using 3 matrix multiplications (GEMMs),
without any pair enumeration, sorting, or atomic operations.

Math overview
-------------
TE(j→i) = Σ_{a,b,c} (cnt3[a,b,c] / N) · log₂(cnt3[a,b,c]·cnt1[b] / (cnt2a[a,b]·cnt2b[b,c]))

where:
  cnt3[a,b,c] = count(X_i(t+tau)=a, X_i(t)=b, X_j(t)=c)  — 3-way joint
  cnt2a[a,b]  = count(X_i(t+tau)=a, X_i(t)=b)              — per variable i (no j dependency)
  cnt2b[b,c]  = count(X_i(t)=b,    X_j(t)=c)              — per pair (i,j)
  cnt1[b]     = count(X_i(t)=b)                            — per variable i

With binary variables (a,b,c ∈ {0,1}), define:
  f  = future  = bin_arrs[:, tau:]    shape (n, L)   ∈ {0, 1}
  p  = past    = bin_arrs[:, :-tau]   shape (n, L)   ∈ {0, 1}
  fp = f * p                         shape (n, L)   ∈ {0, 1}

The 8 joint counts reduce to 3 GEMMs + arithmetic:

  C[i,j] = (fp @ p.T)[i,j]  = Σ_t fp_i[t]·p_j[t]  =  cnt3[1,1,1](i,j)
  D[i,j] = (p  @ p.T)[i,j]  = Σ_t p_i[t]·p_j[t]   =  cnt2b[1,1](i,j)
  E[i,j] = (f  @ p.T)[i,j]  = Σ_t f_i[t]·p_j[t]   =  cnt[fut_i=1, past_j=1](i,j)

All 8 joint counts can be derived from C, D, E plus per-var marginals
n_f[i], n_p[i], n_fp[i].

Result convention (matches scatter_add.py / tenex.py):
  result_matrix[i, j] = TE(j → i)

Expected speedup vs Full-SMEM kernel
-------------------------------------
Tutorial (n=3281, T=459): ~600× (from ~1.2s → ~2ms)
  - Eliminates pairs_np construction (0.135s)
  - Eliminates pairs argsort (0.311s)
  - Eliminates H2D transfer of pairs (0.130s)
  - Replaces 10.76M-pair kernel (0.689s) with 3 GEMMs (~1ms)
"""


import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

__all__ = [
    'compute_te_gemm_b2',
    'compute_te_gemm_b2_multi_gpu',
]


# ─────────────────────────────────────────────────────────────────────────────
# Triton fused kernel: C, D, E → TE  (eliminates (8, n, n) intermediate tensors)
# ─────────────────────────────────────────────────────────────────────────────

if _TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64}),
            triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64}),
            triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128}),
            triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64}),
            triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128}),
        ],
        key=['n', 'm'],
    )
    @triton.jit
    def _te_b2_triton(
        C_ptr, D_ptr, E_ptr,
        nf_ptr, np_ptr, nfp_ptr, npj_ptr,
        out_ptr,
        N_val: tl.constexpr,
        n, m,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """
        Fused Triton kernel: (C, D, E) + per-var marginals → TE(j→i).
        Each program handles a (BLOCK_N, BLOCK_M) tile of the output.
        Memory traffic: read ~3×tile + small marginals, write 1×tile.
        """
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)

        i0 = pid_n * BLOCK_N
        j0 = pid_m * BLOCK_M
        rows = i0 + tl.arange(0, BLOCK_N)[:, None]   # (BN, 1)
        cols = j0 + tl.arange(0, BLOCK_M)[None, :]   # (1, BM)
        mask = (rows < n) & (cols < m)

        N = tl.cast(N_val, tl.float32)

        # Load 3 GEMM outputs: (BN, BM) tiles
        C = tl.load(C_ptr + rows * m + cols, mask=mask, other=0.0).to(tl.float32)
        D = tl.load(D_ptr + rows * m + cols, mask=mask, other=0.0).to(tl.float32)
        E = tl.load(E_ptr + rows * m + cols, mask=mask, other=0.0).to(tl.float32)

        # Load per-var i marginals: shape (BN, 1)
        row_mask = rows < n   # (BN, 1)
        nf  = tl.load(nf_ptr  + rows, mask=row_mask, other=0.0).to(tl.float32)
        np_ = tl.load(np_ptr  + rows, mask=row_mask, other=0.0).to(tl.float32)
        nfp = tl.load(nfp_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        # Load per-var j marginals: shape (1, BM)
        col_mask = cols < m   # (1, BM)
        np_j = tl.load(npj_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        # 8 joint counts (clamp to ≥ 0 for numerical safety)
        cnt_111 = tl.maximum(C, 0.0)
        cnt_011 = tl.maximum(D - C, 0.0)
        cnt_101 = tl.maximum(E - C, 0.0)
        cnt_001 = tl.maximum(np_j - D - E + C, 0.0)
        cnt_110 = tl.maximum(nfp - C, 0.0)
        cnt_010 = tl.maximum(np_ - D - nfp + C, 0.0)
        cnt_100 = tl.maximum(nf - E - nfp + C, 0.0)
        cnt_000 = tl.maximum(N - nf - np_ - np_j + nfp + D + E - C, 0.0)

        # Per-variable i marginals
        c1_1  = np_
        c1_0  = N - np_
        c2a11 = nfp
        c2a01 = np_ - nfp
        c2a10 = nf - nfp
        c2a00 = N - nf - np_ + nfp

        # Per-pair (i,j) marginals
        c2b11 = D
        c2b01 = np_j - D
        c2b10 = np_ - D
        c2b00 = N - np_ - np_j + D

        # TE = Σ (cnt/N) * log2(cnt * c1 / (c2a * c2b))
        # Each term inlined (Triton does not support nested def).
        # When cnt=0 → term=0; denom clamped ≥ 1 for safety.

        # [1,1,1]
        d = tl.maximum(c2a11 * c2b11, 1.0)
        n_ = cnt_111 * c1_1
        te = cnt_111 * tl.where(cnt_111 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [0,1,1]
        d = tl.maximum(c2a01 * c2b11, 1.0)
        n_ = cnt_011 * c1_1
        te += cnt_011 * tl.where(cnt_011 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [1,0,1]
        d = tl.maximum(c2a10 * c2b01, 1.0)
        n_ = cnt_101 * c1_0
        te += cnt_101 * tl.where(cnt_101 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [0,0,1]
        d = tl.maximum(c2a00 * c2b01, 1.0)
        n_ = cnt_001 * c1_0
        te += cnt_001 * tl.where(cnt_001 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [1,1,0]
        d = tl.maximum(c2a11 * c2b10, 1.0)
        n_ = cnt_110 * c1_1
        te += cnt_110 * tl.where(cnt_110 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [0,1,0]
        d = tl.maximum(c2a01 * c2b10, 1.0)
        n_ = cnt_010 * c1_1
        te += cnt_010 * tl.where(cnt_010 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [1,0,0]
        d = tl.maximum(c2a10 * c2b00, 1.0)
        n_ = cnt_100 * c1_0
        te += cnt_100 * tl.where(cnt_100 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        # [0,0,0]
        d = tl.maximum(c2a00 * c2b00, 1.0)
        n_ = cnt_000 * c1_0
        te += cnt_000 * tl.where(cnt_000 > 0.0, tl.log2(tl.maximum(n_ / d, 1e-30)), 0.0)

        te = te / N

        tl.store(out_ptr + rows * m + cols, te, mask=mask)


    def _te_triton_dispatch(C, D, E, n_f, n_p, n_fp, n_p_j, L, out):
        """Launch the Triton fused TE kernel."""
        n, m = C.shape
        grid = lambda meta: (
            triton.cdiv(n, meta['BLOCK_N']),
            triton.cdiv(m, meta['BLOCK_M']),
        )
        _te_b2_triton[grid](
            C, D, E,
            n_f, n_p, n_fp, n_p_j,
            out,
            int(L),   # passed as tl.constexpr (Python int → Triton scalar)
            n, m,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _te_from_gemms(
    C: torch.Tensor,      # (n, m): cnt3[1,1,1](i,j)
    D: torch.Tensor,      # (n, m): cnt2b[1,1](i,j)
    E: torch.Tensor,      # (n, m): count(fut_i=1, past_j=1)
    n_f:  torch.Tensor,   # (n,):  count(fut_i=1)
    n_p:  torch.Tensor,   # (n,):  count(past_i=1)
    n_fp: torch.Tensor,   # (n,):  count(fut_i=1 AND past_i=1)
    n_p_j: torch.Tensor,  # (m,):  count(past_j=1)  — j column marginals
    L: int,               # number of time steps
) -> torch.Tensor:        # (n, m) float32
    """
    Compute TE(j→i) for all (i, j) pairs from 3 GEMMs.

    Vectorized over all 8 binary states simultaneously to minimize
    CUDA kernel launch overhead (~15 launches vs ~80 in the loop version).

    Shapes: n = n_target (rows), m = n_source (columns).
    For the square case m=n; for chunked mode m < n.
    """
    N = float(L)

    # Broadcast per-var marginals to (n, 1) or (1, m)
    nf  = n_f.unsqueeze(1)    # (n, 1)
    np_ = n_p.unsqueeze(1)    # (n, 1)
    nfp = n_fp.unsqueeze(1)   # (n, 1)
    np_j = n_p_j.unsqueeze(0) # (1, m)

    # ── 8 joint counts stacked as (8, n, m) ──────────────────────────────────
    # State order: [1,1,1], [0,1,1], [1,0,1], [0,0,1], [1,1,0], [0,1,0], [1,0,0], [0,0,0]
    # Index notation: cnt[xt1, xt, yt] where xt1=future_i, xt=past_i, yt=past_j
    cnt = torch.stack([
        C,                                          # [1,1,1]
        D - C,                                      # [0,1,1]
        E - C,                                      # [1,0,1]
        np_j - D - E + C,                           # [0,0,1]
        nfp - C,                                    # [1,1,0]
        np_ - D - nfp + C,                          # [0,1,0]
        nf - E - nfp + C,                           # [1,0,0]
        N - nf - np_ - np_j + nfp + D + E - C,     # [0,0,0]
    ], dim=0)                                       # (8, n, m)
    cnt.clamp_(min=0.0)  # numerical safety for FP rounding

    # ── c1_b factor per state: c1[xt] — shape (8, n, 1) ─────────────────────
    # xt=1 for states [*,1,*]: indices 0,1,4,5; xt=0 for [*,0,*]: indices 2,3,6,7
    c1_1 = np_           # (n, 1) — cnt1[xt=1]
    c1_0 = N - np_       # (n, 1) — cnt1[xt=0]
    c1 = torch.stack([c1_1, c1_1, c1_0, c1_0, c1_1, c1_1, c1_0, c1_0], dim=0)  # (8, n, 1)

    # ── c2a[xt1, xt] factor per state — shape (8, n, 1) ─────────────────────
    c2a_11 = nfp                    # count(fut=1, past=1)
    c2a_01 = np_ - nfp              # count(fut=0, past=1)
    c2a_10 = nf - nfp               # count(fut=1, past=0)
    c2a_00 = N - nf - np_ + nfp    # count(fut=0, past=0)
    c2a = torch.stack([c2a_11, c2a_01, c2a_10, c2a_00,
                       c2a_11, c2a_01, c2a_10, c2a_00], dim=0)  # (8, n, 1)

    # ── c2b[xt, yt] factor per state — shape (8, n, m) ───────────────────────
    # yt=1 for states [*,1,1],[*,0,1]: indices 0,1,2,3; yt=0 for [*,*,0]: 4,5,6,7
    c2b_11 = D                        # (n, m) — count(past_i=1, past_j=1)
    c2b_01 = np_j - D                 # (n, m) — count(past_i=0, past_j=1)
    c2b_10 = np_ - D                  # (n, m) — count(past_i=1, past_j=0)
    c2b_00 = N - np_ - np_j + D      # (n, m) — count(past_i=0, past_j=0)
    c2b = torch.stack([c2b_11, c2b_11, c2b_01, c2b_01,
                       c2b_10, c2b_10, c2b_00, c2b_00], dim=0)  # (8, n, m)

    # ── TE = Σ_{8 states} (cnt/N) · log2(cnt · c1 / (c2a · c2b)) ────────────
    # When cnt=0 → term=0; when cnt>0 → denom>0 (mathematical guarantee).
    # Clamp denom ≥ 1 for numerical safety (cnt=0 branch gives 0 anyway).
    numer = cnt * c1                       # (8, n, m)
    denom = (c2a * c2b).clamp_(min=1.0)   # (8, n, m)

    zero = torch.zeros(1, dtype=cnt.dtype, device=cnt.device)
    log_r = torch.where(cnt > 0, torch.log2(numer / denom), zero)  # (8, n, m)

    te = (cnt * log_r).sum(dim=0) / N     # (n, m)
    return te


def _te_from_gemms_triton(C, D, E, n_f, n_p, n_fp, n_p_j, L):
    """
    Triton-fused path: read (C, D, E) → compute TE → write result.
    Eliminates all intermediate (n, n) tensors: memory traffic ~4× (n, n).
    Falls back to _te_from_gemms if Triton unavailable.
    """
    if not _TRITON_AVAILABLE or C.device.type != 'cuda':
        return _te_from_gemms(C, D, E, n_f, n_p, n_fp, n_p_j, L)

    n, m = C.shape
    out = torch.empty(n, m, dtype=torch.float32, device=C.device)
    _te_triton_dispatch(C, D, E, n_f, n_p, n_fp, n_p_j, L, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_gemm_b2(
    bin_arrs: np.ndarray | torch.Tensor,
    tau: int = 1,
    device: torch.device | None = None,
    chunk_size: int | None = None,
    use_triton: bool = True,
) -> np.ndarray:
    """
    Compute TE for all n*(n-1) pairs via 3 GEMMs (binary data only).

    Parameters
    ----------
    bin_arrs   : (n, T) or (n, T, 1) array of ints ∈ {0, 1}
    tau         : temporal lag (default 1)
    device     : torch.device; default cuda:0 if available, else cpu
    chunk_size : process j-axis in chunks (memory saving for large n).
                 Defaults to full-matrix computation.
    use_triton : if True (default), use Triton fused kernel for the TE
                 computation step (eliminates intermediate tensors, faster).

    Returns
    -------
    result : (n, n) float32 ndarray
             result[i, j] = TE(j → i), diagonal = 0
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Coerce to (n, T) float32 tensor on device
    if isinstance(bin_arrs, np.ndarray):
        arr = torch.as_tensor(bin_arrs, dtype=torch.float32, device=device)
    else:
        arr = bin_arrs.to(dtype=torch.float32, device=device)

    if arr.dim() == 3:
        if arr.shape[2] != 1:
            raise ValueError(
                f"GEMM-B2 kernel only supports K=1, got K={arr.shape[2]}"
            )
        arr = arr[:, :, 0]   # (n, T, 1) → (n, T)

    n, T = arr.shape
    L = T - tau

    future = arr[:, tau:]    # (n, L)
    past   = arr[:, :-tau]   # (n, L)
    fp     = future * past  # (n, L)

    # Per-variable marginals (stay on device)
    n_f  = future.sum(1)    # (n,)
    n_p  = past.sum(1)      # (n,)
    n_fp = fp.sum(1)        # (n,)

    # Select TE computation function (Triton or PyTorch)
    _te_fn = _te_from_gemms_triton if use_triton else _te_from_gemms

    # Auto chunk_size: estimate peak memory for full n*n computation.
    # Full path needs: 3 GEMMs (n,n) + TE intermediates (~6 n*n matrices in Triton,
    # ~12 n*n in PyTorch path). Use Triton estimate as lower bound.
    # If that exceeds available VRAM, auto-chunk.
    if chunk_size is None and device.type == 'cuda':
        torch.cuda.empty_cache()
        free_mem, _ = torch.cuda.mem_get_info(device)
        # 3 GEMM outputs + TE output + result = 5 * (n*n*4)
        # Triton fused path: reads 3, writes 1 = 4 * (n*n*4)
        # PyTorch path: up to 12 * (n*n*4) intermediates
        # Conservative estimate: 6 * (n*n*4) for Triton, 14 for PyTorch
        n_matrices = 6 if use_triton else 14
        full_peak = n_matrices * n * n * 4
        # Reserve 128 MB for overhead
        available = max(0, free_mem - 128 * 1024 * 1024)
        if full_peak > available:
            # Chunked: peak = 3 * (n*chunk*4) + output (n*n*4)
            # Solve: 3*n*cs*4 * factor <= available - n*n*4
            factor = 6 if use_triton else 14
            output_bytes = n * n * 4
            chunk_budget = max(0, available - output_bytes)
            chunk_size = max(64, int(chunk_budget // (factor * n * 4)))
            chunk_size = min(chunk_size, n)

    if chunk_size is None or chunk_size >= n:
        # ── Full matrix computation (3 GEMMs, O(n²) peak memory) ─────────────
        C = torch.mm(fp, past.t())      # (n, n)
        D = torch.mm(past, past.t())    # (n, n)
        E = torch.mm(future, past.t())  # (n, n)

        te = _te_fn(C, D, E, n_f, n_p, n_fp, n_p, L)
        te.fill_diagonal_(0.0)

        # ── D2H with pinned memory (7-8× faster than pageable copy) ─────────────
        if device.type == 'cuda':
            pin_buf = torch.empty(n, n, dtype=torch.float32, pin_memory=True)
            pin_buf.copy_(te)
            return pin_buf.numpy()
        return te.cpu().numpy()

    # ── Chunked computation: keep the full output on host pinned memory so
    # we don't undo the chunk-size memory savings with an (n, n) GPU buffer.
    if device.type == 'cuda':
        out_pinned = torch.empty(n, n, dtype=torch.float32, pin_memory=True)
    else:
        out_pinned = torch.empty(n, n, dtype=torch.float32)

    for j0 in range(0, n, chunk_size):
        j1 = min(j0 + chunk_size, n)
        past_j   = past[j0:j1]     # (chunk, L)
        n_p_j    = n_p[j0:j1]      # (chunk,)

        C_c = torch.mm(fp,     past_j.t())   # (n, chunk)
        D_c = torch.mm(past,   past_j.t())   # (n, chunk)
        E_c = torch.mm(future, past_j.t())   # (n, chunk)

        te_c = _te_fn(C_c, D_c, E_c, n_f, n_p, n_fp, n_p_j, L)
        out_pinned[:, j0:j1].copy_(te_c, non_blocking=False)
        del C_c, D_c, E_c, te_c

    out_np = out_pinned.numpy()
    np.fill_diagonal(out_np, 0.0)
    return out_np


def compute_te_gemm_b2_multi_gpu(
    bin_arrs: np.ndarray,
    tau: int = 1,
    device_ids: list[int] | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """
    Multi-GPU GEMM TE for binary data.

    Each GPU handles a contiguous slice of target variables (rows of the result).

    Parameters
    ----------
    bin_arrs   : (n, T) or (n, T, 1) int array ∈ {0, 1}
    tau         : temporal lag (default 1)
    device_ids : list of GPU indices (default: all available)
    chunk_size : passed to compute_te_gemm_b2 on each device

    Returns
    -------
    result : (n, n) float32 ndarray, result[i,j] = TE(j→i)
    """
    import threading

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        return compute_te_gemm_b2(bin_arrs, tau=tau)

    # Coerce input
    if isinstance(bin_arrs, torch.Tensor):
        bin_arrs = bin_arrs.numpy()
    if bin_arrs.ndim == 3:
        bin_arrs = bin_arrs[:, :, 0]

    n, T = bin_arrs.shape
    L = T - tau
    N = float(L)
    n_gpus = len(device_ids)

    # Precompute all marginals on the host. The data is binary {0,1}, so the
    # future/past/joint matrices are kept as int8 and pinned. The shared past
    # matrix is copied to every device, so an int8 (1 B) rather than float32
    # (4 B) copy cuts the per-device H2D traffic 4x. This matters because all
    # GPUs pull from host memory over one NUMA node, and the row-split replicates
    # the full past matrix per device (aggregate copy grows with the GPU count).
    # The float cast happens on-device where bandwidth is abundant.
    b = np.ascontiguousarray(bin_arrs).astype(np.int8)
    arr_f  = np.ascontiguousarray(b[:, tau:])      # (n, L) {0,1}
    arr_p  = np.ascontiguousarray(b[:, :-tau])     # (n, L) {0,1}
    arr_fp = arr_f & arr_p                          # (n, L) {0,1}  (logical AND)
    n_f  = arr_f.sum(1, dtype=np.int64).astype(np.float32)   # (n,)
    n_p  = arr_p.sum(1, dtype=np.int64).astype(np.float32)   # (n,)
    n_fp = arr_fp.sum(1, dtype=np.int64).astype(np.float32)  # (n,)

    t_f  = torch.from_numpy(arr_f).pin_memory()
    t_p  = torch.from_numpy(arr_p).pin_memory()
    t_fp = torch.from_numpy(arr_fp).pin_memory()
    tn_f  = torch.from_numpy(n_f).pin_memory()
    tn_p  = torch.from_numpy(n_p).pin_memory()
    tn_fp = torch.from_numpy(n_fp).pin_memory()

    result = np.zeros((n, n), dtype=np.float32)
    errors = []
    lock = threading.Lock()
    # Serialize the Triton fused kernel so concurrent worker threads do not race
    # on the autotune cache (which is keyed by (n, m) and not thread-safe). The
    # heavy GEMMs above still run in parallel across devices.
    triton_lock = threading.Lock()

    # Distribute target variables across GPUs as contiguous row ranges so each
    # device slices views (no fancy-index copy on the host under the GIL).
    bounds = np.linspace(0, n, n_gpus + 1).astype(int)
    ranges = [(int(bounds[i]), int(bounds[i + 1]))
              for i in range(n_gpus) if bounds[i + 1] > bounds[i]]

    def _worker(dev_idx: int, lo: int, hi: int):
        try:
            torch.cuda.set_device(device_ids[dev_idx])
            dev = torch.device(f'cuda:{device_ids[dev_idx]}')

            # Async H2D of int8, then cast to float32 on-device
            p_all = t_p.to(dev, non_blocking=True).to(torch.float32)      # (n, L)
            f_i   = t_f[lo:hi].to(dev, non_blocking=True).to(torch.float32)
            p_i   = t_p[lo:hi].to(dev, non_blocking=True).to(torch.float32)
            fp_i  = t_fp[lo:hi].to(dev, non_blocking=True).to(torch.float32)

            nf_i  = tn_f[lo:hi].to(dev, non_blocking=True)  # (n_i,)
            np_i  = tn_p[lo:hi].to(dev, non_blocking=True)  # (n_i,)
            nfp_i = tn_fp[lo:hi].to(dev, non_blocking=True)  # (n_i,)
            np_j  = tn_p.to(dev, non_blocking=True)          # (n,)

            # 3 GEMMs: (n_i, L) × (L, n)
            pT = p_all.t()
            C = torch.mm(fp_i, pT)   # (n_i, n): cnt3[1,1,1]
            D = torch.mm(p_i,  pT)   # (n_i, n)
            E = torch.mm(f_i,  pT)   # (n_i, n): cnt[fut=1, past_j=1]

            with triton_lock:
                te_block = _te_from_gemms_triton(C, D, E, nf_i, np_i, nfp_i, np_j, L)

            # Zero self-TE on diagonal rows
            rows = torch.arange(hi - lo, device=dev)
            te_block[rows, torch.arange(lo, hi, device=dev)] = 0.0

            # Disjoint row ranges per device, so this write needs no lock and
            # the D2H copies overlap across devices.
            result[lo:hi, :] = te_block.cpu().numpy()

        except Exception as e:
            with lock:
                errors.append((dev_idx, e))

    # Launch all GPUs in parallel. The triton_lock serializes the (rare) autotune
    # compilation on a cold cache so concurrent threads do not race on the shared
    # autotune dict. Once the cache is warm the lock is uncontended, so every
    # device runs its GEMMs and fused kernel fully in parallel (proper scaling).
    threads = []
    for dev_idx, (lo, hi) in enumerate(ranges):
        t = threading.Thread(target=_worker, args=(dev_idx, lo, hi))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"compute_te_gemm_b2_multi_gpu errors: {errors}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEKernel interface
# ─────────────────────────────────────────────────────────────────────────────

from tenex.kernels import TEKernel


class GEMMB2Kernel(TEKernel):
    """GEMM-based exact TE for binary (b_max=2) data — fastest kernel."""

    @property
    def name(self) -> str:
        return "GEMM-B2"

    @property
    def is_matrix_kernel(self) -> bool:
        return True

    def supports(self, b_max, on_cuda, smem_optin, smem_bytes,
                 n_per_var, source_filter) -> bool:
        return on_cuda and b_max == 2 and not source_filter

    def compute_single_gpu(self, bin_arrs, pairs, b_max, n_per_var,
                           tau, batch_size, device, **kwargs):
        te_matrix = compute_te_gemm_b2(bin_arrs, tau=tau, device=device, use_triton=True)
        p_np = pairs.cpu().numpy() if isinstance(pairs, torch.Tensor) else pairs
        return te_matrix[p_np[:, 0], p_np[:, 1]]

    def compute_matrix(self, data, n_vars, tau, device, **kwargs):
        return compute_te_gemm_b2(data, tau=tau, device=device, use_triton=True)

    def compute_matrix_multi_gpu(self, data, n_vars, tau, device_ids, **kwargs):
        arr_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
        return compute_te_gemm_b2_multi_gpu(arr_np, tau=tau, device_ids=device_ids)

    def log_description(self, **kwargs):
        return "GEMM-B2 (3 GEMMs + Triton fused, exact)"
