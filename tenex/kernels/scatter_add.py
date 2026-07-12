"""
Core Transfer Entropy computation engine for TENEX.

Key innovation vs original FastTENET/MATE:
- Replace 4x torch.unique (O(N log N)) with scatter_add (O(N))
- All pairs in a batch are processed simultaneously in one GPU kernel
- No multiprocessing overhead; multi-GPU via thread-parallel dispatch
- Optional torch.compile for graph-level fusion

TE formula (Y -> X):
    TE = sum_{xt+1, xt, yt}  p(xt+1, xt, yt) * log2[ p(xt+1,xt,yt)*p(xt) / (p(xt+1,xt)*p(xt,yt)) ]

Probabilities are estimated empirically:
    p(xt+1, xt, yt) = count(xt+1, xt, yt) / N
where N = (T - tau) * K is the number of (time, kernel) observations per variable pair.
"""


import math
import time
from typing import Optional

import numpy as np
import torch

from tenex._log import vprint


# ─────────────────────────────────────────────────────────────────────────────
# Batch TE  (scatter_add path)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_te_batch(
    bin_arrs: torch.Tensor,   # (n_vars, T, K) int32
    t_genes:  torch.Tensor,   # (batch,) int64
    s_genes:  torch.Tensor,   # (batch,) int64
    B:        int,            # global max-bin + 1
    tau:       int = 1,
) -> torch.Tensor:             # (batch,) float32
    """
    Core scatter_add-based TE computation for one batch of variable pairs.

    Memory-optimised: intermediates are freed eagerly via ``del`` and
    index / TE computations use in-place ops so that peak GPU memory
    equals exactly ``_peak_bytes_per_pair(B, T, K, tau) * batch``.

    Memory: O(batch * max(N, B^3))   where N = (T-tau)*K
    Time  : O(batch * T * K)
    """
    device = bin_arrs.device
    batch  = t_genes.shape[0]
    T, K   = bin_arrs.shape[1], bin_arrs.shape[2]
    L      = T - tau
    N      = L * K
    B2, B3 = B * B, B * B * B

    # ── 0. Flush residual cache from prior batches ──────────────────────────
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── 1. Extract + compute flat index (in-place) ─────────────────────────
    # Peak: T*K*4 + 2*L*K*8  (target + two int64 slices)
    target = bin_arrs[t_genes]                          # (batch, T, K) int32
    xt1 = target[:, tau:,  :].to(torch.int64)            # (batch, L, K) int64
    xt  = target[:, :-tau, :].to(torch.int64)            # (batch, L, K) int64
    del target

    # In-place: xt1 = xt1*B^2 + xt*B  (no temporary allocations)
    xt1.mul_(B2)
    xt1.add_(xt.mul_(B))
    del xt

    source = bin_arrs[s_genes]                          # (batch, T, K) int32
    yt = source[:, :-tau, :].to(torch.int64)             # (batch, L, K) int64
    del source

    # In-place: xt1 = xt1 + yt + pair_offset
    xt1.add_(yt)
    del yt
    pair_off = torch.arange(batch, device=device, dtype=torch.int64)
    xt1.add_(pair_off.view(-1, 1, 1) * B3)
    idx = xt1.reshape(-1)                               # (batch*N,)
    del xt1
    # Live: idx only -- L*K*8 per pair

    # ── 2. Histogram via scatter_add ───────────────────────────────────────
    # Peak: N*8 + N*4 + B^3*4  (idx + ones + cnt3_flat)
    ones = torch.ones(batch * N, dtype=torch.int32, device=device)
    cnt3_flat = torch.zeros(batch * B3, dtype=torch.int32, device=device)
    cnt3_flat.scatter_add_(0, idx, ones)
    del idx, ones

    # ── 3. Marginals as float32 (exact for L < 2^24 = 16M) ────────────────
    c3f = cnt3_flat.view(batch, B, B, B).float()        # (batch, B, B, B) f32
    del cnt3_flat

    # ── 4. Marginals + TE formula ─────────────────────────────────────────
    cnt2a = c3f.sum(dim=3)                              # (batch, B, B) f32
    cnt2b = c3f.sum(dim=1)                              # (batch, B, B) f32
    cnt1  = c3f.sum(dim=(1, 3))                         # (batch, B)   f32

    # TE = sum (c3/N) * log2( c3*c1 / (c2a*c2b) )
    # Peak: c3f + numer + denom + cnt2a + cnt2b = 3*B^3*4 + 2*B^2*4
    numer = c3f * cnt1[:, None, :, None]                # (batch, B, B, B) f32
    del cnt1
    denom = cnt2a[:, :, :, None] * cnt2b[:, None, :, :] # (batch, B, B, B) f32
    del cnt2a, cnt2b

    # In-place: ratio -> log2 -> multiply by c3f -> nan_to_num -> sum
    denom.clamp_(min=1)          # avoid div-by-zero
    numer.div_(denom)            # numer = c3*c1 / (c2a*c2b)
    del denom
    numer.log2_()                # numer = log2(ratio)
    numer.mul_(c3f)              # numer = c3f * log2(ratio)
    # When c3f=0: numer was 0, log2(0/x)=-inf, 0*(-inf)=NaN -> fix:
    numer.nan_to_num_(0.0)
    te = numer.sum(dim=(1, 2, 3)).div_(N)
    del numer, c3f
    return te


# ─────────────────────────────────────────────────────────────────────────────
# Precise memory estimation
# ─────────────────────────────────────────────────────────────────────────────

def _peak_bytes_per_pair(B: int, T: int, K: int, tau: int = 1) -> int:
    """
    Peak GPU *allocated* memory per variable pair in ``_compute_te_batch``.

    Tracks the maximum simultaneously-alive tensors at each phase:

    Phase 1 (Extract + index):
        target(int32) + xt1(int64) + xt(int64), before del target.
        => T*K*4 + 2*L*K*8

    Phase 2 (Histogram build):
        idx(int64) + ones(int32) + cnt3_flat(int32).
        => N*8 + N*4 + B^3*4  =  12*N + 4*B^3

    Phase 3 (Float conversion):
        cnt3_flat(int32) + c3f(f32) coexist before del cnt3_flat.
        => 2*B^3*4  =  8*B^3

    Phase 4 (Marginals + TE):
        c3f + numer + denom + cnt2a + cnt2b.
        => 3*B^3*4 + 2*B^2*4  =  12*B^3 + 8*B^2
    """
    L = T - tau
    N = L * K
    B2 = B * B
    B3 = B ** 3

    p_extract = T * K * 4 + 2 * L * K * 8    # target + xt1 + xt
    p_hist    = N * 12 + B3 * 4               # idx + ones + cnt3_flat
    p_conv    = 8 * B3                         # cnt3_flat + c3f
    p_te      = 12 * B3 + 8 * B2             # c3f + numer + denom + marginals

    return max(p_extract, p_hist, p_conv, p_te)


# ─────────────────────────────────────────────────────────────────────────────
# Batch-size selection: heuristic (default) + runtime auto-tuning (optional)
# ─────────────────────────────────────────────────────────────────────────────

_L2_MULTIPLIER = 350  # optimal_working_set ~ 350 * L2_cache_size
_FALLBACK_VRAM_FRAC = 0.17  # when L2 size unavailable


def _heuristic_batch_size(
    B: int, T: int, K: int,
    device: torch.device,
    tau: int = 1,
) -> int:
    """
    VRAM-aware batch size heuristic (deterministic, zero cost).

    Two-stage budget:
    1. L2-based throughput target (350x L2 cache size).
    2. VRAM safety cap: free - reserve (128 MB or 20% of free, whichever larger).
       Critical for small-VRAM GPUs like RTX 2080 Ti (11 GB).

    Falls back to 17% of total VRAM when L2 size is unavailable.
    """
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    peak_pp = _peak_bytes_per_pair(B, T, K, tau)

    # Stage 1: L2-based throughput target
    props = torch.cuda.get_device_properties(device)
    l2 = getattr(props, 'L2_cache_size', 0)
    if l2 > 0:
        target = _L2_MULTIPLIER * l2
    else:
        target = int(total * _FALLBACK_VRAM_FRAC)

    # Stage 2: VRAM safety cap
    reserve = max(128 * 1024 * 1024, int(free * 0.20))
    cap = max(0, free - reserve)
    budget = min(target, cap)

    bs = max(64, budget // max(peak_pp, 1))
    bs = (bs // 64) * 64
    return bs


# Module-level cache: (B, T, K, tau, device_index) -> batch_size
_tuned_bs_cache: dict[tuple, int] = {}


def _autotune_batch_size(
    bin_arrs: torch.Tensor,
    B: int,
    tau: int = 1,
) -> int:
    """
    Runtime auto-tuning via geometric doubling with early stopping.

    Measures actual throughput at ~7 candidate batch sizes (from max/64
    to max, doubling each step).  Stops early when throughput declines
    for 2 consecutive steps.  Results are cached per (data_shape, GPU).

    Cost: ~1-3 seconds on first call.  Zero on subsequent calls.
    Thread-safe for multi-GPU (each device calibrates independently).
    """
    device = bin_arrs.device
    T, K = bin_arrs.shape[1], bin_arrs.shape[2]
    dev_idx = device.index if device.index is not None else 0
    key = (B, T, K, tau, dev_idx)

    cached = _tuned_bs_cache.get(key)
    if cached is not None:
        return cached

    peak_pp = _peak_bytes_per_pair(B, T, K, tau)
    n_vars = bin_arrs.shape[0]

    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    max_bs = int(free * 0.90) // peak_pp
    max_bs = max(64, (max_bs // 64) * 64)

    # ~7 candidates: start from max/64, double up to max
    start = max(64, (max_bs // 64 // 64) * 64)

    best_bs = start
    best_pps = 0.0
    decline = 0
    tested = []

    bs = start
    while bs <= max_bs and decline < 2:
        actual = min((bs // 64) * 64, max_bs)
        actual = max(actual, 64)

        t_g = torch.randint(0, n_vars, (actual,), device=device, dtype=torch.int64)
        s_g = torch.randint(0, n_vars, (actual,), device=device, dtype=torch.int64)

        # warmup
        _compute_te_batch(bin_arrs, t_g, s_g, B, tau)
        torch.cuda.synchronize()

        # measure
        t0 = time.perf_counter()
        _compute_te_batch(bin_arrs, t_g, s_g, B, tau)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        pps = actual / max(t1 - t0, 1e-9)
        tested.append((actual, pps))

        if pps > best_pps:
            best_pps = pps
            best_bs = actual
            decline = 0
        else:
            decline += 1

        del t_g, s_g
        torch.cuda.empty_cache()
        bs *= 2

    _tuned_bs_cache[key] = best_bs
    vprint(f"[TENEX] auto-tuned batch_size={best_bs:,} "
          f"({best_bs * peak_pp / 2**30:.1f} GB, "
          f"{best_pps:,.0f} pairs/s) on cuda:{dev_idx} "
          f"[tested {len(tested)} sizes]")
    return best_bs


def _auto_batch_size(
    bin_arrs: torch.Tensor,
    B: int,
    tau: int = 1,
    autotune: bool = False,
) -> int:
    """
    Choose batch_size that maximises GPU throughput without OOM.

    Parameters
    ----------
    bin_arrs  : discretised data tensor (on GPU)
    B         : b_max (number of histogram bins per dimension)
    tau        : time delay
    autotune  : if True, run runtime calibration (slower but more accurate).
                if False (default), use deterministic heuristic.

    For multi-GPU, each device selects its own batch size independently.
    """
    device = bin_arrs.device
    if not device.type.startswith('cuda'):
        return 2048

    T, K = bin_arrs.shape[1], bin_arrs.shape[2]

    if autotune:
        return _autotune_batch_size(bin_arrs, B, tau)

    return _heuristic_batch_size(B, T, K, device, tau)


# ─────────────────────────────────────────────────────────────────────────────
# Public API: compute all pair TEs on one device
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_scatter_add(
    bin_arrs: torch.Tensor,   # (n_vars, T, K) int32 -- already on device
    pairs:    torch.Tensor,   # (n_pairs, 2) int64    -- already on device
    B:        int,
    tau:       int = 1,
    batch_size: Optional[int] = None,
    autotune: bool = False,
) -> np.ndarray:              # (n_pairs,) float32
    """
    Compute Transfer Entropy for all variable pairs on one GPU device (scatter_add path).

    Uses a scatter_add-based histogram (O(N) per pair). No torch.unique overhead.
    Works for any B (no SMEM size constraint).

    Parameters
    ----------
    bin_arrs   : discretised expression array (n_vars, T, K), int32 on GPU
    pairs      : (n_pairs, 2) -- column 0 = target, column 1 = source
    B          : global max-bin + 1 (from preprocess.discretize)
    tau         : time delay (default 1)
    batch_size : pairs per GPU batch (auto-selected if None)
    autotune   : use runtime calibration for batch_size (default False)

    Returns
    -------
    entropies  : (n_pairs,) float32 numpy array
    """
    device = bin_arrs.device
    n_pairs = pairs.shape[0]

    if batch_size is None:
        batch_size = _auto_batch_size(bin_arrs, B, tau=tau, autotune=autotune)

    entropy_list: list[torch.Tensor] = []

    for beg in range(0, n_pairs, batch_size):
        end = min(beg + batch_size, n_pairs)
        batch_pairs = pairs[beg:end]

        te = _compute_te_batch(
            bin_arrs,
            batch_pairs[:, 0],
            batch_pairs[:, 1],
            B, tau,
        )
        entropy_list.append(te)

    return torch.cat(entropy_list).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def compute_te_scatter_add_multi_gpu(
    bin_arrs_cpu: torch.Tensor,   # (n_vars, T, K) int32 on CPU
    pairs_cpu:    np.ndarray,     # (n_pairs, 2) int32
    B:            int,
    device_ids:   list,
    tau:           int = 1,
    batch_size:   Optional[int] = None,
    autotune:     bool = False,
) -> np.ndarray:
    """
    Distribute scatter_add TE computation across multiple GPUs (thread pool).

    Each GPU independently selects its own optimal batch size (via heuristic
    or auto-tuning), so heterogeneous GPU setups are handled correctly.

    Parameters
    ----------
    bin_arrs_cpu : discretised data on CPU (copied to each GPU)
    pairs_cpu    : all (target, source) pairs as numpy int32 array
    B            : global max-bin + 1
    device_ids   : list of CUDA device indices, e.g. [0, 1, 2, 3]
    tau           : time delay
    batch_size   : per-GPU batch size (auto if None)
    autotune     : use runtime calibration for batch_size (default False)
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import as_completed

    from tenex.kernels import _try_pin

    n_gpus  = len(device_ids)
    n_pairs = len(pairs_cpu)
    chunk   = math.ceil(n_pairs / n_gpus)

    # Pin CPU tensor for async H2D transfer
    bin_pinned = _try_pin(bin_arrs_cpu)

    results = [None] * n_gpus

    def _run(gpu_rank: int):
        dev_id = device_ids[gpu_rank]
        torch.cuda.set_device(dev_id)
        device = torch.device(f'cuda:{dev_id}')

        beg = gpu_rank * chunk
        end = min(beg + chunk, n_pairs)
        if beg >= n_pairs:
            return gpu_rank, np.empty(0, dtype=np.float32)

        # Async transfer from pinned memory
        arr = bin_pinned.to(device, non_blocking=True)
        p   = torch.from_numpy(pairs_cpu[beg:end].astype(np.int64)).to(device)

        bs = batch_size
        if bs is None:
            bs = _auto_batch_size(arr, B, tau=tau, autotune=autotune)

        ents = compute_te_scatter_add(arr, p, B, tau=tau, batch_size=bs)
        return gpu_rank, ents

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = [pool.submit(_run, i) for i in range(n_gpus)]
        for fut in as_completed(futures):
            rank, ents = fut.result()
            results[rank] = ents

    return np.concatenate([r for r in results if r is not None and len(r) > 0])


# ─────────────────────────────────────────────────────────────────────────────
# TEKernel interface
# ─────────────────────────────────────────────────────────────────────────────

from tenex.kernels import TEKernel


class ScatterAddKernel(TEKernel):
    """Universal fallback TE kernel using scatter_add (CPU or GPU)."""

    @property
    def name(self) -> str:
        return "scatter_add"

    def supports(self, b_max, on_cuda, smem_optin, smem_bytes,
                 n_per_var, source_filter) -> bool:
        return True  # universal fallback -- always supported

    def compute_single_gpu(self, bin_arrs, pairs, b_max, n_per_var,
                           tau, batch_size, device, **kwargs):
        return compute_te_scatter_add(bin_arrs, pairs, b_max,
                                      tau=tau, batch_size=batch_size)

    def compute_multi_gpu(self, bin_arrs_cpu, pairs_np, b_max,
                          n_per_var_cpu, tau, batch_size, device_ids, **kwargs):
        return compute_te_scatter_add_multi_gpu(
            bin_arrs_cpu, pairs_np, b_max, device_ids,
            tau=tau, batch_size=batch_size
        )
