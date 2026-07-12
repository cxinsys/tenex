"""
TRACE — Threshold-Refined Aggregate Causal Entropy.

TRACE is a GPU-accelerated key-driver inference method. It constructs a
directed causal network from a pairwise TE matrix via marginal surrogate
thresholding, then aggregates OutTE / InTE on each variable's connected
neighbourhood.

Relationship to POINT (Lee 2025)
--------------------------------
TRACE borrows the OutTE / InTE definitions from Julian Lee's POINT paper
(Phys. Rev. E 111, 024308, 2025) but does NOT implement POINT's procedure.
POINT prescribes conditional-MI forward selection + conditional backward
elimination + final joint TE significance test. TRACE instead uses a
marginal TE threshold for candidate generation, descending algorithmically
from TENET (Kim et al. 2021) rather than POINT.

Algorithm
---------
1. Compute pairwise TE matrix (via TENEX's fast TE kernels).
2. Threshold by ``mean + significance·std`` over random off-diagonal
   entries of the TE matrix to obtain a directed network.
3. For each variable X, compute
     OutTE(X) — transfer entropy from X to its connected targets;
     InTE(X)  — transfer entropy from X's connected sources into X.
   Each H(·) is computed on GPU via the 4-way hash-fusion kernel in
   ``tenex/csrc/trace_sort_entropy.cu`` (empirically bit-for-bit
   identical to CPU ``np.unique(structured_dtype)`` on 4 datasets × 3
   GPUs; the validation is for the entropy engine, not the pruning
   algorithm).

When NOT to use TRACE
---------------------
TRACE is a *marginal* method. If confounders or mediators are strong,
TRACE retains spurious edges and OutTE / InTE are biased. For an
inference method that provably controls such bias, use ``method='point'``
(once implemented).

References
----------
- Julian Lee, Phys. Rev. E 111, 024308 (2025) — OutTE / InTE definitions,
  and the exact procedure targeted by the planned ``method='point'``.
- Kim et al., Nucleic Acids Research (2021) — TENET, the marginal-TE
  thresholding lineage that TRACE descends from algorithmically.
"""

from dataclasses import dataclass

import numpy as np
import torch

import os
import threading

from tenex.inference import (
    InferenceMethod, GRN, build_pairs, make_grn, threshold_pairs,
)
from tenex._log import vprint

# ── CUDA kernel loading ──────────────────────────────────────────────────────

_trace_smem_module = None
_trace_smem_lock = threading.Lock()


_trace_remap_module = None
_trace_remap_lock = threading.Lock()


def _get_trace_remap_module():
    """Load the remap CUDA kernel (AOT or JIT).

    These kernels compute multivariate entropy via state remapping and
    are reusable by both TRACE and the forthcoming paper-exact POINT
    implementation.
    """
    global _trace_remap_module
    if _trace_remap_module is not None:
        return _trace_remap_module

    with _trace_remap_lock:
        if _trace_remap_module is not None:
            return _trace_remap_module

        try:
            import tenex._ext.trace_remap as _mod
            _required = ['trace_outte_remap_launch', 'trace_inte_remap_launch']
            if all(hasattr(_mod, f) for f in _required):
                _trace_remap_module = _mod
                return _trace_remap_module
        except ImportError:
            pass

        csrc_dir = os.path.join(os.path.dirname(__file__), '..', 'csrc')
        src_path = os.path.join(csrc_dir, 'trace_remap.cu')
        with open(src_path, 'r') as f:
            cuda_src = f.read()

        from torch.utils.cpp_extension import load_inline
        _trace_remap_module = load_inline(
            name="trace_remap",
            cpp_sources='',
            cuda_sources=cuda_src,
            extra_cuda_cflags=['-O3'],
            verbose=False,
        )
        return _trace_remap_module


_trace_sort_module = None
_trace_sort_lock = threading.Lock()


def _get_trace_sort_module():
    """Load the POINT sort-entropy CUDA kernel (AOT or JIT)."""
    global _trace_sort_module
    if _trace_sort_module is not None:
        return _trace_sort_module

    with _trace_sort_lock:
        if _trace_sort_module is not None:
            return _trace_sort_module

        try:
            import tenex._ext.trace_sort_entropy as _mod
            if all(hasattr(_mod, f) for f in
                   ['poly_hash_launch', 'sorted_entropy_launch']):
                _trace_sort_module = _mod
                return _trace_sort_module
        except ImportError:
            pass

        csrc_dir = os.path.join(os.path.dirname(__file__), '..', 'csrc')
        src_path = os.path.join(csrc_dir, 'trace_sort_entropy.cu')
        with open(src_path, 'r') as f:
            cuda_src = f.read()

        from torch.utils.cpp_extension import load_inline
        _trace_sort_module = load_inline(
            name="trace_sort_entropy",
            cpp_sources='',
            cuda_sources=cuda_src,
            extra_cuda_cflags=['-O3', '--allow-unsupported-compiler'],
            verbose=False,
        )
        return _trace_sort_module


_trace_fused_module = None
_trace_fused_lock = threading.Lock()


def _get_trace_fused_module():
    """Load the POINT fused hash-entropy CUDA kernel."""
    global _trace_fused_module
    if _trace_fused_module is not None:
        return _trace_fused_module

    with _trace_fused_lock:
        if _trace_fused_module is not None:
            return _trace_fused_module

        try:
            import tenex._ext.trace_hash_entropy as _mod
            if hasattr(_mod, 'fused_hash_entropy_launch'):
                _trace_fused_module = _mod
                return _trace_fused_module
        except ImportError:
            pass

        csrc_dir = os.path.join(os.path.dirname(__file__), '..', 'csrc')
        src_path = os.path.join(csrc_dir, 'trace_hash_entropy.cu')
        with open(src_path, 'r') as f:
            cuda_src = f.read()

        from torch.utils.cpp_extension import load_inline
        _trace_fused_module = load_inline(
            name="trace_hash_entropy",
            cpp_sources='',
            cuda_sources=cuda_src,
            extra_cuda_cflags=['-O3'],
            verbose=False,
        )
        return _trace_fused_module


def _remap_fast(data: np.ndarray, n_bins: np.ndarray,
                gene_indices: list[int],
                t_start: int, t_end: int,
                max_U: int | None = None) -> tuple[np.ndarray, int, bool]:
    """
    Fast multivariate state remapping with optional bin coarsening.

    Uses mixed-radix integer encoding on the needed [t_start, t_end) range
    only, then np.unique on a 1D int64 array.  This avoids:
      - structured-dtype np.unique (slow comparison-based sort)
      - full-T array copies for coarsening

    If max_U is given and B^d > max_U, bins are coarsened so that the
    product of per-variable bins fits within max_U.

    Returns (remap_int32, U, coarsened).
    """
    L = t_end - t_start
    d = len(gene_indices)
    if d == 0 or L == 0:
        return np.zeros(L, dtype=np.int32), 0, False

    # Determine coarsening
    coarsened = False
    max_B = None
    if max_U is not None:
        r_max = int(n_bins[gene_indices].max())
        if r_max ** d > max_U:
            max_B = max(2, int(max_U ** (1.0 / d)))
            coarsened = True

    # Mixed-radix encode: one pass over columns, only [t_start:t_end)
    encoded = np.zeros(L, dtype=np.int64)
    for j in gene_indices:
        col = data[j, t_start:t_end].astype(np.int64)
        if coarsened:
            old_B = int(n_bins[j])
            if old_B > max_B:
                col = col * max_B // old_B
        n_vals = int(col.max()) + 1
        encoded = encoded * n_vals + col

    _, inverse = np.unique(encoded, return_inverse=True)
    U = int(inverse.max()) + 1 if len(inverse) > 0 else 0

    # If U still exceeds max_U after bin coarsening (because L > B^d),
    # requantize the remapped indices themselves to fit within max_U.
    if max_U is not None and U > max_U:
        inverse = inverse * max_U // U
        U = int(inverse.max()) + 1
        coarsened = True

    return inverse.astype(np.int32), U, coarsened


@dataclass
class TRACEResult:
    """Result of TRACE analysis."""
    outte: np.ndarray          # (n,) float32 — OutTE for each variable
    inte: np.ndarray           # (n,) float32 — InTE for each variable
    variable_names: np.ndarray # (n,) str
    network: dict              # target_idx → list of source indices
    grn: GRN                   # thresholded GRN from pairwise TE

    def top_drivers(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k drivers ranked by OutTE."""
        idx = np.argsort(self.outte)[::-1][:k]
        return [(self.variable_names[i], float(self.outte[i])) for i in idx]

    def top_receivers(self, k: int = 10) -> list[tuple[str, float]]:
        """Return top-k receivers ranked by InTE."""
        idx = np.argsort(self.inte)[::-1][:k]
        return [(self.variable_names[i], float(self.inte[i])) for i in idx]


# ── Sort-based GPU entropy ───────────────────────────────────────────────────

def _remap_struct(data: np.ndarray, gene_indices: list[int],
                  t_start: int, t_end: int) -> tuple[np.ndarray, int]:
    """Remap multivariate state to dense [0, U) using structured dtype.

    Unlike mixed-radix encoding, this never overflows because np.unique
    compares tuples element-by-element.
    """
    L = t_end - t_start
    if len(gene_indices) == 0 or L == 0:
        return np.zeros(L, dtype=np.int64), 0
    obs = data[gene_indices, t_start:t_end].T  # (L, d)
    obs = np.ascontiguousarray(obs)
    tau = np.dtype([('', obs.dtype)] * obs.shape[1])
    _, inverse = np.unique(obs.view(tau).ravel(), return_inverse=True)
    U = int(inverse.max()) + 1 if len(inverse) > 0 else 0
    return inverse.astype(np.int64), U


# ── Batched GPU sort-entropy ─────────────────────────────────────────────────

def _pack_batch(cols: torch.Tensor, base: int,
                pack_w: int) -> list[torch.Tensor]:
    """Pack (B, d, L) int32 columns → list of (B, L) int64 keys.

    Uses fixed base for overflow-safe mixed-radix encoding.
    Groups of pack_w columns per key.
    """
    B, d, L = cols.shape
    keys = []
    for s in range(0, d, pack_w):
        e = min(s + pack_w, d)
        enc = cols[:, s, :].to(torch.int64)
        for j in range(s + 1, e):
            enc = enc * base + cols[:, j, :].to(torch.int64)
        keys.append(enc)
    return keys


def _batched_sort_entropy(keys: list[torch.Tensor],
                          L: int, device: torch.device) -> torch.Tensor:
    """Batched sort-entropy: keys = list of (B, L) int64 → (B,) float32.

    For B tasks simultaneously:
      1. Batched lexsort via stable sort on dim=1
      2. Boundary detection per row
      3. Run-length → entropy via scatter_add
    """
    B = keys[0].shape[0]
    n_keys = len(keys)
    if B == 0 or L == 0:
        return torch.zeros(B, dtype=torch.float32, device=device)

    # ── Batched lexsort (least → most significant) ──
    order = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).contiguous()
    for k in range(n_keys - 1, -1, -1):
        gathered = torch.gather(keys[k], 1, order)
        _, si = gathered.sort(dim=1, stable=True)
        order = torch.gather(order, 1, si)

    # ── Boundary detection: (B, L-1) ──
    changes = torch.zeros(B, L - 1, dtype=torch.bool, device=device)
    for k in range(n_keys):
        sk = torch.gather(keys[k], 1, order)
        changes |= (sk[:, 1:] != sk[:, :-1])

    # ── Group IDs → counts → entropy ──
    gid = torch.zeros(B, L, dtype=torch.int64, device=device)
    gid[:, 1:] = changes.to(torch.int64).cumsum(dim=1)

    row_off = torch.arange(B, device=device, dtype=torch.int64).unsqueeze(1) * L
    flat = (row_off + gid).reshape(-1)

    ones = torch.ones(B * L, dtype=torch.float32, device=device)
    counts = torch.zeros(B * L, dtype=torch.float32, device=device)
    counts.scatter_add_(0, flat, ones)
    counts = counts.reshape(B, L)

    probs = counts / L
    log_p = torch.where(probs > 0, torch.log2(probs), torch.zeros_like(probs))
    return -(probs * log_p).sum(dim=1)


# ── GPU-accelerated entropy estimation ───────────────────────────────────────

def _encode_states(columns: list[torch.Tensor], max_val: int) -> torch.Tensor:
    """
    Encode multivariate observations as single integers.

    columns: list of (L,) int tensors, each with values in [0, max_val).
    Returns: (L,) int64 tensor of encoded states.
    """
    encoded = columns[0].to(torch.int64)
    base = max_val
    for col in columns[1:]:
        encoded = encoded * base + col.to(torch.int64)
    return encoded


def _entropy_from_counts(counts: torch.Tensor, N: int) -> float:
    """H = -Σ (c/N) log2(c/N) from count tensor."""
    counts = counts[counts > 0].float()
    probs = counts / N
    return -float((probs * probs.log2()).sum())


def _empirical_entropy_gpu(columns: list[torch.Tensor], max_val: int) -> float:
    """
    H(X1, X2, ..., Xd) from empirical frequencies on GPU.

    columns: list of d tensors, each (L,) int with values in [0, max_val).
    """
    L = columns[0].shape[0]
    if L == 0:
        return 0.0
    if len(columns) == 1:
        _, counts = columns[0].unique(return_counts=True)
    else:
        encoded = _encode_states(columns, max_val)
        _, counts = encoded.unique(return_counts=True)
    return _entropy_from_counts(counts, L)


def _conditional_entropy_gpu(
    Y_cols: list[torch.Tensor],
    X_cols: list[torch.Tensor],
    max_val: int,
) -> float:
    """H(Y | X) = H(Y, X) - H(X) on GPU."""
    joint_cols = Y_cols + X_cols
    return _empirical_entropy_gpu(joint_cols, max_val) - _empirical_entropy_gpu(X_cols, max_val)


# ── CPU entropy estimation ───────────────────────────────────────────────────

_HASH_P_NP = np.int64(1000000007)


def _poly_hash_entropy_cpu(columns: list[np.ndarray]) -> float:
    """H(X1,...,Xd) via polynomial hash → np.unique → entropy.

    ~6× faster than structured-dtype np.unique for large d.
    """
    L = len(columns[0])
    if L == 0:
        return 0.0
    h = np.zeros(L, dtype=np.int64)
    for col in columns:
        h = h * _HASH_P_NP + col.astype(np.int64)
    _, counts = np.unique(h, return_counts=True)
    probs = counts / L
    return -float(np.sum(probs * np.log2(probs)))


def _empirical_entropy_cpu(sequences: np.ndarray) -> float:
    """H(sequences) from empirical frequencies (bits). Legacy."""
    N = len(sequences)
    if N == 0:
        return 0.0
    if sequences.ndim == 1:
        sequences = sequences[:, None]
    sequences = np.ascontiguousarray(sequences)
    tau = np.dtype([('', sequences.dtype)] * sequences.shape[1])
    _, counts = np.unique(sequences.view(tau).ravel(), return_counts=True)
    probs = counts / N
    return -float(np.sum(probs * np.log2(probs)))


def _conditional_entropy_cpu(Y: np.ndarray, X: np.ndarray) -> float:
    """H(Y | X) = H(Y, X) - H(X)."""
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim == 1:
        X = X[:, None]
    joint = np.concatenate([Y, X], axis=1)
    return _empirical_entropy_cpu(joint) - _empirical_entropy_cpu(X)


# ── OutTE / InTE computation ─────────────────────────────────────────────────

def compute_outte_gpu(data: torch.Tensor, var_idx: int, connected: list[int],
                      tau: int = 1, max_val: int = 256) -> float:
    """OutTE(X) on GPU using torch.unique for entropy estimation."""
    if len(connected) == 0:
        return 0.0

    L = data.shape[1] - tau
    Y_future_cols = [data[j, tau:] for j in connected]
    Y_past_cols = [data[j, :L] for j in connected]
    X_past_col = data[var_idx, :L]

    h1 = _conditional_entropy_gpu(Y_future_cols, Y_past_cols, max_val)
    h2 = _conditional_entropy_gpu(Y_future_cols, [X_past_col] + Y_past_cols, max_val)
    return max(h1 - h2, 0.0)


def compute_inte_gpu(data: torch.Tensor, var_idx: int, connected: list[int],
                     tau: int = 1, max_val: int = 256) -> float:
    """InTE(X) on GPU using torch.unique for entropy estimation."""
    if len(connected) == 0:
        return 0.0

    L = data.shape[1] - tau
    X_future_col = data[var_idx, tau:]
    X_past_col = data[var_idx, :L]
    Y_past_cols = [data[j, :L] for j in connected]

    h1 = _conditional_entropy_gpu([X_future_col], [X_past_col], max_val)
    h2 = _conditional_entropy_gpu([X_future_col], [X_past_col] + Y_past_cols, max_val)
    return max(h1 - h2, 0.0)


def compute_outte_cpu(data: np.ndarray, var_idx: int, connected: list[int],
                      tau: int = 1) -> float:
    """OutTE(X) on CPU with 4-way polynomial hash fusion."""
    if len(connected) == 0:
        return 0.0
    T = data.shape[1]
    L = T - tau
    yf_cols = [data[j, tau:tau+L] for j in connected]
    yp_cols = [data[j, :L] for j in connected]
    xp_col = data[var_idx, :L]

    # 4-way: h_yf, h_yp computed once, derive all 4 entropies
    h_yf = np.zeros(L, dtype=np.int64)
    h_yp = np.zeros(L, dtype=np.int64)
    for yf, yp in zip(yf_cols, yp_cols):
        h_yf = h_yf * _HASH_P_NP + yf.astype(np.int64)
        h_yp = h_yp * _HASH_P_NP + yp.astype(np.int64)
    h_xp = xp_col.astype(np.int64)

    d = len(connected)
    P_d = np.int64(1)
    for _ in range(d):
        P_d = np.int64(P_d * _HASH_P_NP)

    def _ent(h):
        _, counts = np.unique(h, return_counts=True)
        p = counts / L
        return -float(np.sum(p * np.log2(p)))

    e0 = _ent(h_yf * P_d + h_yp)           # H(Yf,Yp)
    e1 = _ent(h_yp)                          # H(Yp)
    e2 = _ent((h_yf * _HASH_P_NP + h_xp) * P_d + h_yp)  # H(Yf,Xp,Yp)
    e3 = _ent(h_xp * P_d + h_yp)            # H(Xp,Yp)

    return max(e0 - e1 - e2 + e3, 0.0)


def compute_inte_cpu(data: np.ndarray, var_idx: int, connected: list[int],
                     tau: int = 1) -> float:
    """InTE(X) on CPU with 4-way polynomial hash fusion."""
    if len(connected) == 0:
        return 0.0
    T = data.shape[1]
    L = T - tau
    yp_cols = [data[j, :L] for j in connected]
    xf_col = data[var_idx, tau:tau+L]
    xp_col = data[var_idx, :L]

    h_yp = np.zeros(L, dtype=np.int64)
    for yp in yp_cols:
        h_yp = h_yp * _HASH_P_NP + yp.astype(np.int64)
    h_xf = xf_col.astype(np.int64)
    h_xp = xp_col.astype(np.int64)
    h_xfxp = h_xf * _HASH_P_NP + h_xp

    d = len(connected)
    P_d = np.int64(1)
    for _ in range(d):
        P_d = np.int64(P_d * _HASH_P_NP)

    def _ent(h):
        _, counts = np.unique(h, return_counts=True)
        p = counts / L
        return -float(np.sum(p * np.log2(p)))

    e0 = _ent(h_xfxp)                  # H(Xf,Xp)
    e1 = _ent(h_xp)                     # H(Xp)
    e2 = _ent(h_xfxp * P_d + h_yp)     # H(Xf,Xp,Yp)
    e3 = _ent(h_xp * P_d + h_yp)       # H(Xp,Yp)

    return max(e0 - e1 - e2 + e3, 0.0)


# ── Causal network construction (vectorized) ─────────────────────────────────

def build_causal_network(
    te_matrix: np.ndarray,
    n_surrogates: int = 100,
    significance: float = 2.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[int, list[int]]:
    """
    Build a directed causal network from pairwise TE via surrogate testing.

    GPU-accelerated when device is a CUDA device: threshold + edge extraction
    on GPU, then transfer sparse edges to CPU for dict construction.
    """
    n = te_matrix.shape[0]
    rng = np.random.default_rng(seed)

    # Surrogate sampling (CPU — tiny, 100 values).
    # Avoid the (i, i) diagonal so the zero self-TE doesn't deflate the
    # surrogate distribution and lower the cutoff.
    rows = rng.integers(0, n, size=n_surrogates)
    cols = rng.integers(0, n, size=n_surrogates)
    mask = rows == cols
    cols[mask] = (cols[mask] + 1) % n
    surrogate_values = te_matrix[rows, cols]
    threshold = float(surrogate_values.mean() + significance * surrogate_values.std())

    # GPU path: threshold + nonzero on GPU (only if enough VRAM)
    use_gpu = False
    if device is not None and device.type == 'cuda':
        te_bytes = n * n * 4  # float32
        free, _ = torch.cuda.mem_get_info(device)
        use_gpu = (te_bytes * 2 < free)  # need ~2× for te + nonzero result

    if use_gpu:
        te_gpu = torch.from_numpy(te_matrix).to(device)
        te_gpu.fill_diagonal_(0.0)
        edges_gpu = torch.nonzero(te_gpu > threshold, as_tuple=False)
        src_idx = edges_gpu[:, 0].cpu().numpy()
        tgt_idx = edges_gpu[:, 1].cpu().numpy()
        del te_gpu, edges_gpu
        torch.cuda.empty_cache()
    else:
        # CPU path
        edges = te_matrix > threshold
        np.fill_diagonal(edges, False)
        src_idx, tgt_idx = np.where(edges)
        del edges

    # Build network: CSR-like (src_sorted, splits) + dict wrapper
    network = {j: [] for j in range(n)}
    if len(tgt_idx) > 0:
        order = np.argsort(tgt_idx, kind='stable')
        tgt_sorted = tgt_idx[order]
        src_sorted = src_idx[order].astype(np.int64)
        splits = np.searchsorted(tgt_sorted, np.arange(n + 1))
        for j in range(n):
            s, e = int(splits[j]), int(splits[j + 1])
            if s < e:
                network[j] = src_sorted[s:e]  # ndarray, avoids .tolist()

    return network


# ── TRACE inference method ───────────────────────────────────────────────────

class TRACEMethod(InferenceMethod):

    @property
    def name(self) -> str:
        return 'trace'

    def infer(self, te_matrix, variable_names, device, **kwargs) -> TRACEResult:
        """
        Run TRACE analysis (Threshold-Refined Aggregate Causal Entropy).

        TRACE prunes the causal network via a marginal pairwise-TE threshold
        (controlled by ``n_surrogates`` and ``significance``). It does NOT
        accept the ``fdr`` / ``links`` thresholding that other inference
        methods use; passing those raises an error so callers do not
        silently rely on ignored parameters.

        Additional kwargs
        -----------------
        bin_data       : (n, T) or (n, T, 1) int array — discretized time
                         series for OutTE/InTE computation. Required.
                         Multi-kernel data (K>1) is not supported.
        tau             : time lag (default 1).
        n_surrogates   : surrogates for network construction (default 100).
        significance   : std multiplier for significance (default 2.0).
        devices        : list of GPU indices for multi-GPU (default: single).
        """
        bin_data = kwargs.get('bin_data', None)
        if bin_data is None:
            raise ValueError(
                "TRACE requires 'bin_data' (discretized time series). "
                "Pass the bin arrays used for TE computation."
            )

        # TRACE uses a marginal TE threshold, not fdr/links/sources.
        # Surface a clear error if the caller passes them directly so they
        # don't think the threshold is being applied.
        for _ignored in ('fdr', 'links', 'sources'):
            if _ignored in kwargs:
                raise TypeError(
                    f"TRACE does not accept {_ignored!r}: it prunes the "
                    f"network via a marginal TE threshold (n_surrogates, "
                    f"significance). When invoking via NetWeaver these are "
                    f"dropped automatically — remove the explicit kwarg."
                )

        tau = kwargs.get('tau', 1)
        n_surrogates = kwargs.get('n_surrogates', 100)
        significance = kwargs.get('significance', 2.0)
        devices = kwargs.get('devices', None)

        te_matrix = np.asarray(te_matrix, dtype=np.float32)
        bin_data = np.asarray(bin_data)
        if bin_data.ndim == 3:
            if bin_data.shape[2] != 1:
                raise ValueError(
                    f"TRACE requires K=1 bin arrays; got shape {bin_data.shape}. "
                    f"Multi-kernel binning (e.g. FSBW-B/FSBW-T) is not supported "
                    f"by the TRACE descriptors."
                )
            bin_data = bin_data[:, :, 0]
        n = te_matrix.shape[0]

        # Step 1: Build causal network via surrogate threshold (GPU-accelerated)
        network = build_causal_network(
            te_matrix, n_surrogates=n_surrogates, significance=significance,
            device=device,
        )

        # Precompute outgoing targets via reverse index — vectorized
        # Collect all (source, target) edges from network
        all_src = []
        all_tgt = []
        for j in range(n):
            srcs = network[j]
            if len(srcs) > 0:
                srcs_arr = np.asarray(srcs, dtype=np.int64)
                all_src.append(srcs_arr)
                all_tgt.append(np.full(len(srcs_arr), j, dtype=np.int64))

        if all_src:
            all_src = np.concatenate(all_src)
            all_tgt = np.concatenate(all_tgt)
        else:
            all_src = np.empty(0, dtype=np.int64)
            all_tgt = np.empty(0, dtype=np.int64)
        n_edges = len(all_src)

        # Build outgoing: source → list of targets (reverse index)
        outgoing = {i: [] for i in range(n)}
        if n_edges > 0:
            order = np.argsort(all_src, kind='stable')
            src_sorted = all_src[order]
            tgt_sorted = all_tgt[order]
            splits = np.searchsorted(src_sorted, np.arange(n + 1))
            for i in range(n):
                s, e = int(splits[i]), int(splits[i + 1])
                if s < e:
                    outgoing[i] = tgt_sorted[s:e]  # ndarray

        # Step 2: Build GRN from causal edges (vectorized)
        if n_edges > 0:
            pairs_net = np.column_stack([all_src, all_tgt])
            te_net = te_matrix[all_src, all_tgt]
            grn = make_grn(variable_names, pairs_net, te_net)
        else:
            grn = make_grn(variable_names, np.empty((0, 2), dtype=np.int64),
                           np.empty(0, dtype=np.float32))
        vprint(f"[TENEX] TRACE:{n_edges} causal edges")

        # Step 3: Compute OutTE and InTE
        use_cuda_kernel = (device.type == 'cuda')
        n_per_var = kwargs.get('n_per_var', None)
        K = kwargs.get('K', 1)

        outte = np.zeros(n, dtype=np.float32)
        inte = np.zeros(n, dtype=np.float32)

        if use_cuda_kernel:
            # Multi-GPU if devices list provided with >1 GPUs
            if devices and len(devices) > 1:
                outte, inte = self._compute_cuda_multigpu(
                    bin_data, outgoing, network, n, tau, K, devices,
                )
            else:
                outte, inte = self._compute_cuda(
                    bin_data, n_per_var, outgoing, network, n, tau, K, device,
                )
        else:
            max_val = int(bin_data.max()) + 1
            for i in range(n):
                outte[i] = compute_outte_cpu(bin_data, i, outgoing[i], tau=tau)
                inte[i] = compute_inte_cpu(bin_data, i, network[i], tau=tau)

        n_nonzero_out = (outte > 0).sum()
        n_nonzero_in = (inte > 0).sum()
        vprint(f"[TENEX] TRACE:OutTE non-zero: {n_nonzero_out}/{n}, "
              f"InTE non-zero: {n_nonzero_in}/{n}")

        return TRACEResult(
            outte=outte,
            inte=inte,
            variable_names=variable_names,
            network=network,
            grn=grn,
        )

    @staticmethod
    def _build_descriptors_out(outgoing, n, T, tau, var_remap=None):
        """Vectorized OutTE descriptor build.
        If var_remap is provided, translates variable indices to compact form.
        """
        var_ids = [i for i in range(n) if len(outgoing[i]) > 0]
        if not var_ids:
            return [], [], np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)

        d_list = [len(outgoing[i]) for i in var_ids]
        all_targets = np.concatenate(
            [np.asarray(outgoing[i], dtype=np.int64) for i in var_ids])

        if var_remap is not None:
            all_targets = var_remap[all_targets]
            xp_idx = var_remap[np.array(var_ids, dtype=np.int64)]
        else:
            xp_idx = np.array(var_ids, dtype=np.int64)

        yf_all = all_targets * T + tau
        yp_all = all_targets * T
        xp_all = xp_idx * T
        return var_ids, d_list, yf_all, yp_all, xp_all

    @staticmethod
    def _build_descriptors_in(network, n, T, tau, var_remap=None):
        """Vectorized InTE descriptor build.
        If var_remap is provided, translates variable indices to compact form.
        """
        var_ids = [i for i in range(n) if len(network[i]) > 0]
        if not var_ids:
            return [], [], np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64)

        d_list = [len(network[i]) for i in var_ids]
        all_sources = np.concatenate(
            [np.asarray(network[i], dtype=np.int64) for i in var_ids])

        if var_remap is not None:
            all_sources = var_remap[all_sources]
            xp_idx = var_remap[np.array(var_ids, dtype=np.int64)]
        else:
            xp_idx = np.array(var_ids, dtype=np.int64)

        yp_all = all_sources * T
        xf_all = xp_idx * T + tau
        xp_all = xp_idx * T
        return var_ids, d_list, yp_all, xf_all, xp_all

    def _compute_cuda(self, bin_data, n_per_var, outgoing, network, n, tau, K, device):
        """
        TRACE via 4-way fused CUDA hash-sort-entropy pipeline.

        4-way hash fusion reads each target/source column ONCE and derives
        all 4 entropy hashes algebraically from the polynomial identity:
            h(A,B) = h(A) * P^|B| + h(B)   (int64 wrapping)

        Column-read savings: OutTE 6d+2 → 2d+1 (3×), InTE 2d+6 → d+2 (2×).
        """
        T_data = bin_data.shape[1]
        L = T_data - tau
        T = T_data

        mod = _get_trace_sort_module()
        has_4way = hasattr(mod, 'fused_outte_4way_launch')

        # ── Data upload: adaptive dtype + compact if VRAM is tight ──
        # Choose int16 if max value fits (2× less memory bandwidth)
        max_val = int(bin_data[:, :T_data].max())
        np_dtype = np.int16 if max_val < 32768 else np.int32
        elem_bytes = 2 if np_dtype == np.int16 else 4

        torch.cuda.empty_cache()
        full_data_bytes = n * T * elem_bytes
        free, _ = torch.cuda.mem_get_info(device)

        if full_data_bytes < free * 0.7:
            # Plenty of VRAM: upload full data
            data_cpu = torch.from_numpy(
                bin_data[:, :T_data].astype(np_dtype)).pin_memory()
            data_gpu = data_cpu.to(device, non_blocking=True)
            data_flat = data_gpu.reshape(-1)
            del data_cpu
            var_remap = None
        else:
            # Tight VRAM: upload only referenced genes
            ref_genes = set()
            for i in range(n):
                if len(outgoing[i]) > 0:
                    ref_genes.add(i)
                    ref_genes.update(int(x) for x in outgoing[i])
                if len(network[i]) > 0:
                    ref_genes.add(i)
                    ref_genes.update(int(x) for x in network[i])
            ref_list = sorted(ref_genes)
            n_ref = len(ref_list)

            var_remap = np.full(n, -1, dtype=np.int64)
            var_remap[ref_list] = np.arange(n_ref, dtype=np.int64)

            compact_data = bin_data[ref_list, :T_data].astype(np_dtype)
            data_cpu = torch.from_numpy(compact_data).pin_memory()
            data_gpu = data_cpu.to(device, non_blocking=True)
            data_flat = data_gpu.reshape(-1)
            del data_cpu, compact_data
            T = T_data

        # If 4-way kernels unavailable, fall back to generic pipeline
        if not has_4way:
            return self._compute_cuda_generic(
                mod, data_flat, outgoing, network, n, tau, T, L, device)

        # ── Build structured descriptors (vectorized) ───────────
        # Column reference: var_idx * T + time_offset
        # If compacted, var_remap translates original → compact index
        out_var_ids, out_d_list, out_yf_all, out_yp_all, out_xp_all = \
            self._build_descriptors_out(outgoing, n, T, tau, var_remap)
        n_out_vars = len(out_var_ids)

        in_var_ids, in_d_list, in_yp_all, in_xf_all, in_xp_all = \
            self._build_descriptors_in(network, n, T, tau, var_remap)
        n_in_vars = len(in_var_ids)

        if n_out_vars == 0 and n_in_vars == 0:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        outte = torch.zeros(n, dtype=torch.float32, device=device)
        inte  = torch.zeros(n, dtype=torch.float32, device=device)

        # ── VRAM-adaptive batch sizing ────────────────────────────
        # fused launcher allocates per batch:
        #   hashes (bs*4*L*8) + sorted (bs*4*L*8) + CUB temp (~bs*4*L)
        #   + seg_offsets (bs*4*4) ≈ bs * 4 * L * 17
        # Measure free VRAM after uploading descriptors for accurate sizing.
        per_var_bytes = L * 4 * 17  # 4 segs/var × L × (8+8+1) bytes
        # CUB segment offsets require int32: 4*batch*L < 2^31
        max_vars_int32 = max(1, (2**31 - 1) // (4 * max(L, 1)))

        def _compute_batch(device, per_var_bytes):
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            reserve = max(256 * 1024**2, int(free * 0.15))
            return max(1, min(
                int((free - reserve) / per_var_bytes), max_vars_int32))

        # ── OutTE: 4-way fused pipeline ──────────────────────────
        if n_out_vars > 0:
            out_d_vals = np.array(out_d_list, dtype=np.int32)
            out_d_offsets = np.zeros(n_out_vars + 1, dtype=np.int32)
            np.cumsum(out_d_vals, out=out_d_offsets[1:])

            yf_t = torch.from_numpy(out_yf_all).to(device)
            yp_t = torch.from_numpy(out_yp_all).to(device)
            xp_t = torch.from_numpy(out_xp_all).to(device)
            dv_t = torch.from_numpy(out_d_vals).to(device)
            do_t = torch.from_numpy(out_d_offsets).to(device)

            # Measure free VRAM AFTER descriptor upload
            batch_vars = _compute_batch(device, per_var_bytes)

            out_ent = torch.empty(n_out_vars * 4, dtype=torch.float32,
                                  device=device)
            s = 0
            while s < n_out_vars:
                bs = min(batch_vars, n_out_vars - s)
                try:
                    out_ent[s*4:(s+bs)*4] = mod.fused_outte_4way_launch(
                        data_flat, yf_t, yp_t, xp_t, dv_t, do_t,
                        s, bs, L)
                    s += bs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    per_var_bytes *= 2
                    batch_vars = _compute_batch(device, per_var_bytes)
                    continue

            idx = torch.arange(0, n_out_vars * 4, 4, device=device)
            te = out_ent[idx] - out_ent[idx+1] - out_ent[idx+2] + out_ent[idx+3]
            outte[torch.tensor(out_var_ids, device=device)] = te.clamp(min=0.0)
            del out_ent, yf_t, yp_t, xp_t, dv_t, do_t

        # ── InTE: 4-way fused pipeline ───────────────────────────
        if n_in_vars > 0:
            in_d_vals = np.array(in_d_list, dtype=np.int32)
            in_d_offsets = np.zeros(n_in_vars + 1, dtype=np.int32)
            np.cumsum(in_d_vals, out=in_d_offsets[1:])

            yp_t = torch.from_numpy(in_yp_all).to(device)
            xf_t = torch.from_numpy(in_xf_all).to(device)
            xp_t = torch.from_numpy(in_xp_all).to(device)
            dv_t = torch.from_numpy(in_d_vals).to(device)
            do_t = torch.from_numpy(in_d_offsets).to(device)

            # Re-measure after OutTE freed + InTE descriptors uploaded
            per_var_bytes = L * 4 * 17
            batch_vars = _compute_batch(device, per_var_bytes)

            in_ent = torch.empty(n_in_vars * 4, dtype=torch.float32,
                                 device=device)
            s = 0
            while s < n_in_vars:
                bs = min(batch_vars, n_in_vars - s)
                try:
                    in_ent[s*4:(s+bs)*4] = mod.fused_inte_4way_launch(
                        data_flat, yp_t, xf_t, xp_t, dv_t, do_t,
                        s, bs, L)
                    s += bs
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    per_var_bytes *= 2
                    batch_vars = _compute_batch(device, per_var_bytes)
                    continue

            idx = torch.arange(0, n_in_vars * 4, 4, device=device)
            te = in_ent[idx] - in_ent[idx+1] - in_ent[idx+2] + in_ent[idx+3]
            inte[torch.tensor(in_var_ids, device=device)] = te.clamp(min=0.0)
            del in_ent, yp_t, xf_t, xp_t, dv_t, do_t

        return outte.cpu().numpy(), inte.cpu().numpy()

    def _compute_cuda_multigpu(self, bin_data, outgoing, network, n, tau, K, devices):
        """
        Multi-GPU TRACE: split genes across GPUs by workload.

        Each GPU gets a copy of data_flat and a partition of genes.
        Genes are sorted by d (descending) and assigned round-robin
        for workload balance.
        """
        n_gpus = len(devices)
        cuda_devices = [torch.device(f'cuda:{d}') for d in devices]

        # Pre-load CUDA module before spawning threads (thread-safe init)
        mod = _get_trace_sort_module()

        # Build descriptors (same as _compute_cuda)
        T_data = bin_data.shape[1]
        T = T_data
        L = T_data - tau

        out_var_ids, out_yf_chunks, out_yp_chunks = [], [], []
        out_xp_list, out_d_list = [], []
        for i in range(n):
            tgt = outgoing[i]
            if len(tgt) == 0:
                continue
            tgt_a = np.array(tgt, dtype=np.int64)
            out_yf_chunks.append(tgt_a * T + tau)
            out_yp_chunks.append(tgt_a * T)
            out_xp_list.append(np.int64(i * T))
            out_d_list.append(len(tgt))
            out_var_ids.append(i)

        in_var_ids, in_yp_chunks, in_xf_list = [], [], []
        in_xp_list, in_d_list = [], []
        for i in range(n):
            src = network[i]
            if len(src) == 0:
                continue
            src_a = np.array(src, dtype=np.int64)
            in_yp_chunks.append(src_a * T)
            in_xf_list.append(np.int64(i * T + tau))
            in_xp_list.append(np.int64(i * T))
            in_d_list.append(len(src))
            in_var_ids.append(i)

        # Partition genes by workload (round-robin on sorted-by-d)
        def partition_vars(var_ids, d_list, chunks_list, n_gpus):
            """Partition genes round-robin by descending d for balance."""
            if not var_ids:
                return [[] for _ in range(n_gpus)]
            order = np.argsort(d_list)[::-1]  # heaviest first
            partitions = [[] for _ in range(n_gpus)]
            for rank, idx in enumerate(order):
                partitions[rank % n_gpus].append(idx)
            return partitions

        out_parts = partition_vars(out_var_ids, out_d_list, out_yf_chunks, n_gpus)
        in_parts = partition_vars(in_var_ids, in_d_list, in_yp_chunks, n_gpus)

        # Prepare per-GPU data
        bin_data_int32 = bin_data[:, :T_data].astype(np.int32)
        outte = np.zeros(n, dtype=np.float32)
        inte = np.zeros(n, dtype=np.float32)
        results = [None] * n_gpus

        def gpu_worker(gpu_idx):
            dev = cuda_devices[gpu_idx]
            torch.cuda.set_device(dev)
            data_gpu = torch.from_numpy(bin_data_int32).to(dev)
            data_flat = data_gpu.reshape(-1)

            out_result = np.zeros(n, dtype=np.float32)
            in_result = np.zeros(n, dtype=np.float32)

            # VRAM budget
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(dev)
            reserve = max(256 * 1024**2, int(free * 0.15))
            per_var_bytes = L * 8 * 4 * 2 + L * 8
            max_vars_int32 = max(1, (2**31 - 1) // (4 * max(L, 1)))
            batch_vars = max(1, min(
                int((free - reserve) / per_var_bytes), max_vars_int32))

            # OutTE for this GPU's partition
            my_out = out_parts[gpu_idx]
            if my_out:
                my_yf = np.concatenate([out_yf_chunks[j] for j in my_out])
                my_yp = np.concatenate([out_yp_chunks[j] for j in my_out])
                my_xp = np.array([out_xp_list[j] for j in my_out], dtype=np.int64)
                my_dv = np.array([out_d_list[j] for j in my_out], dtype=np.int32)
                my_do = np.zeros(len(my_out) + 1, dtype=np.int32)
                np.cumsum(my_dv, out=my_do[1:])
                my_ids = [out_var_ids[j] for j in my_out]

                yf_t = torch.from_numpy(my_yf).to(dev)
                yp_t = torch.from_numpy(my_yp).to(dev)
                xp_t = torch.from_numpy(my_xp).to(dev)
                dv_t = torch.from_numpy(my_dv).to(dev)
                do_t = torch.from_numpy(my_do).to(dev)

                n_my = len(my_out)
                out_ent = torch.empty(n_my * 4, dtype=torch.float32, device=dev)
                s = 0
                while s < n_my:
                    bs = min(batch_vars, n_my - s)
                    try:
                        out_ent[s*4:(s+bs)*4] = mod.fused_outte_4way_launch(
                            data_flat, yf_t, yp_t, xp_t, dv_t, do_t, s, bs, L)
                        s += bs
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        per_var_bytes *= 2
                        free, _ = torch.cuda.mem_get_info(dev)
                        batch_vars = max(1, int((free - reserve) / per_var_bytes))
                        continue

                idx = torch.arange(0, n_my * 4, 4, device=dev)
                te = out_ent[idx] - out_ent[idx+1] - out_ent[idx+2] + out_ent[idx+3]
                te = te.clamp(min=0.0).cpu().numpy()
                for k, gid in enumerate(my_ids):
                    out_result[gid] = te[k]
                del out_ent, yf_t, yp_t, xp_t, dv_t, do_t

            # InTE for this GPU's partition
            my_in = in_parts[gpu_idx]
            if my_in:
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(dev)
                per_var_bytes = L * 8 * 4 * 2 + L * 8
                batch_vars = max(1, min(
                    int((free - reserve) / per_var_bytes), max_vars_int32))

                my_yp = np.concatenate([in_yp_chunks[j] for j in my_in])
                my_xf = np.array([in_xf_list[j] for j in my_in], dtype=np.int64)
                my_xp = np.array([in_xp_list[j] for j in my_in], dtype=np.int64)
                my_dv = np.array([in_d_list[j] for j in my_in], dtype=np.int32)
                my_do = np.zeros(len(my_in) + 1, dtype=np.int32)
                np.cumsum(my_dv, out=my_do[1:])
                my_ids = [in_var_ids[j] for j in my_in]

                yp_t = torch.from_numpy(my_yp).to(dev)
                xf_t = torch.from_numpy(my_xf).to(dev)
                xp_t = torch.from_numpy(my_xp).to(dev)
                dv_t = torch.from_numpy(my_dv).to(dev)
                do_t = torch.from_numpy(my_do).to(dev)

                n_my = len(my_in)
                in_ent = torch.empty(n_my * 4, dtype=torch.float32, device=dev)
                s = 0
                while s < n_my:
                    bs = min(batch_vars, n_my - s)
                    try:
                        in_ent[s*4:(s+bs)*4] = mod.fused_inte_4way_launch(
                            data_flat, yp_t, xf_t, xp_t, dv_t, do_t, s, bs, L)
                        s += bs
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        per_var_bytes *= 2
                        free, _ = torch.cuda.mem_get_info(dev)
                        batch_vars = max(1, int((free - reserve) / per_var_bytes))
                        continue

                idx = torch.arange(0, n_my * 4, 4, device=dev)
                te = in_ent[idx] - in_ent[idx+1] - in_ent[idx+2] + in_ent[idx+3]
                te = te.clamp(min=0.0).cpu().numpy()
                for k, gid in enumerate(my_ids):
                    in_result[gid] = te[k]
                del in_ent, yp_t, xf_t, xp_t, dv_t, do_t

            del data_gpu, data_flat
            torch.cuda.empty_cache()
            results[gpu_idx] = (out_result, in_result)

        # Launch workers in parallel threads
        errors = [None] * n_gpus
        def safe_worker(g):
            try:
                gpu_worker(g)
            except Exception as e:
                errors[g] = e

        threads = []
        for g in range(n_gpus):
            t = threading.Thread(target=safe_worker, args=(g,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Check for errors
        for g in range(n_gpus):
            if errors[g] is not None:
                raise RuntimeError(
                    f"TRACE GPU worker {g} (cuda:{devices[g]}) failed: "
                    f"{errors[g]}"
                ) from errors[g]

        # Merge results
        for g in range(n_gpus):
            out_r, in_r = results[g]
            outte += out_r
            inte += in_r

        return outte, inte

    def _compute_cuda_generic(self, mod, data_flat, outgoing, network,
                              n, tau, T, L, device):
        """Fallback: generic per-job hash-sort-entropy (no 4-way fusion)."""
        col_chunks = []
        job_sizes = []
        out_var_ids = []
        in_var_ids = []

        for i in range(n):
            tgt = outgoing[i]
            if len(tgt) == 0:
                continue
            tgt_a = np.array(tgt, dtype=np.int64)
            yf = tgt_a * T + tau
            yp = tgt_a * T
            xp_val = np.int64(i * T)
            d = len(tgt)
            col_chunks.append(np.concatenate([yf, yp]))
            job_sizes.append(2 * d)
            col_chunks.append(yp.copy())
            job_sizes.append(d)
            col_chunks.append(np.concatenate([yf, [xp_val], yp]))
            job_sizes.append(2 * d + 1)
            col_chunks.append(np.concatenate([[xp_val], yp]))
            job_sizes.append(d + 1)
            out_var_ids.append(i)

        n_out_jobs = len(job_sizes)

        for i in range(n):
            src = network[i]
            if len(src) == 0:
                continue
            src_a = np.array(src, dtype=np.int64)
            yp = src_a * T
            xf_val = np.int64(i * T + tau)
            xp_val = np.int64(i * T)
            d = len(src)
            col_chunks.append(np.array([xf_val, xp_val]))
            job_sizes.append(2)
            col_chunks.append(np.array([xp_val]))
            job_sizes.append(1)
            col_chunks.append(np.concatenate([[xf_val, xp_val], yp]))
            job_sizes.append(d + 2)
            col_chunks.append(np.concatenate([[xp_val], yp]))
            job_sizes.append(d + 1)
            in_var_ids.append(i)

        n_jobs = len(job_sizes)
        if n_jobs == 0:
            return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

        all_col_starts = np.concatenate(col_chunks)
        all_job_offsets = np.zeros(n_jobs + 1, dtype=np.int32)
        np.cumsum(job_sizes, out=all_job_offsets[1:])

        col_starts_t = torch.from_numpy(all_col_starts).to(device)
        job_offsets_t = torch.from_numpy(all_job_offsets).to(device)

        all_ent = torch.empty(n_jobs, dtype=torch.float32, device=device)

        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        reserve = max(256 * 1024**2, int(free * 0.15))
        per_job_bytes = L * 8 * 3
        batch_size = max(1, int((free - reserve) / per_job_bytes))

        has_fused = hasattr(mod, 'fused_hash_sort_entropy_launch')

        s = 0
        while s < n_jobs:
            bs = min(batch_size, n_jobs - s)
            try:
                if has_fused:
                    all_ent[s:s+bs] = mod.fused_hash_sort_entropy_launch(
                        data_flat, col_starts_t, job_offsets_t, s, bs, L)
                else:
                    hashes = mod.poly_hash_launch(
                        data_flat, col_starts_t, job_offsets_t, s, bs, L)
                    hashes = torch.sort(hashes, dim=1).values
                    all_ent[s:s+bs] = mod.sorted_entropy_launch(hashes, bs, L)
                    del hashes
                s += bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                per_job_bytes *= 2
                free, _ = torch.cuda.mem_get_info(device)
                batch_size = max(1, int((free - reserve) / per_job_bytes))
                continue

        outte = torch.zeros(n, dtype=torch.float32, device=device)
        inte  = torch.zeros(n, dtype=torch.float32, device=device)

        if out_var_ids:
            idx = torch.arange(0, n_out_jobs, 4, device=device)
            te = all_ent[idx] - all_ent[idx+1] - all_ent[idx+2] + all_ent[idx+3]
            outte[torch.tensor(out_var_ids, device=device)] = te.clamp(min=0.0)

        if in_var_ids:
            base = n_out_jobs
            idx = torch.arange(base, base + len(in_var_ids) * 4, 4, device=device)
            te = all_ent[idx] - all_ent[idx+1] - all_ent[idx+2] + all_ent[idx+3]
            inte[torch.tensor(in_var_ids, device=device)] = te.clamp(min=0.0)

        return outte.cpu().numpy(), inte.cpu().numpy()

    @staticmethod
    def _vram_budget(device):
        """Return usable VRAM in bytes (free − 15% reserve, min 256 MB)."""
        if device.type != 'cuda':
            return 2**40
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        reserve = max(256 * 1024**2, int(free * 0.15))
        return max(0, free - reserve)

    def _compute_outte_remap(self, mod, bin_data, n_bins, outgoing, n, tau, L, block_size, device):
        """Remap multivariate states and launch OutTE kernel in VRAM-sized mini-batches."""

        budget = self._vram_budget(device)

        # ── Phase 1: Pre-compute remapped arrays with dynamic coarsening ──
        per_task = []  # None for empty, else (yf, yp, xp, meta4, hist_size)
        n_coarsened = 0
        for i in range(n):
            targets = outgoing[i]
            if len(targets) == 0:
                per_task.append(None)
                continue

            Bx = int(n_bins[i])
            # Kernel indexes cnt3 as int: a*Up*Bx + b*Bx + c.
            # To avoid int32 overflow: Uf * Up * Bx < 2^31.
            # Also constrained by VRAM budget.
            max_hist = min(budget // 4, (2**31 - 1) // 2)
            max_U = max(2, int(np.sqrt(max_hist / max(Bx, 1))))

            yf, Uf, c1 = _remap_fast(bin_data, n_bins, targets, tau, tau + L, max_U)
            yp, Up, c2 = _remap_fast(bin_data, n_bins, targets, 0, L, max_U)
            if c1 or c2:
                n_coarsened += 1

            xp = bin_data[i, :L].astype(np.int32)
            hist_size = Uf * Up * Bx
            per_task.append((yf, yp, xp, [L, Uf, Up, Bx], hist_size))

        active = [i for i in range(n) if per_task[i] is not None]
        if not active:
            return np.zeros(n, dtype=np.float32)

        # ── Phase 2: Classify tasks by VRAM budget ──
        result = np.zeros(n, dtype=np.float32)

        gpu_tasks, cpu_tasks = [], []
        for i in active:
            if per_task[i][4] * 4 > budget:  # still too large after coarsening
                cpu_tasks.append(i)
            else:
                gpu_tasks.append(i)

        # ── Phase 3: GPU mini-batched kernel launch ──
        pos, n_batches = 0, 0
        while pos < len(gpu_tasks):
            batch_bytes, end = 0, pos
            while end < len(gpu_tasks):
                task_bytes = per_task[gpu_tasks[end]][4] * 4 + L * 12
                if batch_bytes + task_bytes > budget and end > pos:
                    break
                batch_bytes += task_bytes
                end += 1

            batch_idx = gpu_tasks[pos:end]
            yf_parts, yp_parts, xp_parts = [], [], []
            meta_rows, hist_sizes = [], []
            data_off = 0
            for i in batch_idx:
                yf, yp, xp, meta4, hs = per_task[i]
                yf_parts.append(yf); yp_parts.append(yp); xp_parts.append(xp)
                meta_rows.append(meta4 + [data_off])
                hist_sizes.append(hs)
                data_off += L

            nb = len(batch_idx)
            meta_arr = np.array(meta_rows, dtype=np.int32)
            cnt3_off = np.zeros(nb + 1, dtype=np.int64)
            for j in range(nb):
                cnt3_off[j + 1] = cnt3_off[j] + hist_sizes[j]

            yf_t = torch.from_numpy(np.concatenate(yf_parts)).to(device)
            yp_t = torch.from_numpy(np.concatenate(yp_parts)).to(device)
            xp_t = torch.from_numpy(np.concatenate(xp_parts)).to(device)
            meta_t = torch.from_numpy(meta_arr).to(device)
            off_t = torch.from_numpy(cnt3_off).to(device)
            cnt3 = torch.zeros(int(cnt3_off[nb]), dtype=torch.int32, device=device)

            out = mod.trace_outte_remap_launch(
                yf_t, yp_t, xp_t, meta_t, off_t, cnt3, block_size)
            batch_res = out.cpu().numpy()
            for j, i in enumerate(batch_idx):
                result[i] = max(batch_res[j], 0.0)

            del yf_t, yp_t, xp_t, meta_t, off_t, cnt3, out
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            n_batches += 1
            pos = end

        # ── Phase 4: CPU fallback for oversized tasks ──
        for i in cpu_tasks:
            result[i] = compute_outte_cpu(bin_data, i, outgoing[i], tau=tau)

        if n_batches > 1 or cpu_tasks or n_coarsened:
            vprint(f"[TENEX] POINT OutTE: {n_batches} GPU batch(es) "
                  f"({len(gpu_tasks)} tasks, {n_coarsened} coarsened), "
                  f"{len(cpu_tasks)} CPU fallback")

        return result

    def _compute_inte_remap(self, mod, bin_data, n_bins, network, n, tau, L, block_size, device):
        """Remap multivariate states and launch InTE kernel in VRAM-sized mini-batches."""

        budget = self._vram_budget(device)

        # ── Phase 1: Pre-compute remapped arrays with dynamic coarsening ──
        per_task = []  # None for empty, else (xf, xp, yp, meta4, hist_size)
        n_coarsened = 0
        for i in range(n):
            sources = network[i]
            if len(sources) == 0:
                per_task.append(None)
                continue
            xf = bin_data[i, tau:tau + L].astype(np.int32)
            xp = bin_data[i, :L].astype(np.int32)
            Bx = int(n_bins[i])

            # Kernel indexes cnt3 as int: a*Bx*Up + b*Up + c.
            # To avoid int32 overflow: Bx * Bx * Up < 2^31.
            max_hist = min(budget // 4, (2**31 - 1) // 2)
            max_Up = max(2, int(max_hist / (Bx * Bx)))

            yp, Up, coarsened = _remap_fast(bin_data, n_bins, sources, 0, L, max_Up)
            if coarsened:
                n_coarsened += 1

            per_task.append((xf, xp, yp, [L, Bx, Up], Bx * Bx * Up))

        active = [i for i in range(n) if per_task[i] is not None]
        if not active:
            return np.zeros(n, dtype=np.float32)

        # ── Phase 2: Classify tasks by VRAM budget ──
        result = np.zeros(n, dtype=np.float32)

        gpu_tasks, cpu_tasks = [], []
        for i in active:
            if per_task[i][4] * 4 > budget:
                cpu_tasks.append(i)
            else:
                gpu_tasks.append(i)

        # ── Phase 3: GPU mini-batched kernel launch ──
        pos, n_batches = 0, 0
        while pos < len(gpu_tasks):
            batch_bytes, end = 0, pos
            while end < len(gpu_tasks):
                task_bytes = per_task[gpu_tasks[end]][4] * 4 + L * 12
                if batch_bytes + task_bytes > budget and end > pos:
                    break
                batch_bytes += task_bytes
                end += 1

            batch_idx = gpu_tasks[pos:end]
            xf_parts, xp_parts, yp_parts = [], [], []
            meta_rows, hist_sizes = [], []
            data_off = 0
            for i in batch_idx:
                xf, xp, yp, meta4, hs = per_task[i]
                xf_parts.append(xf); xp_parts.append(xp); yp_parts.append(yp)
                meta_rows.append(meta4 + [data_off])
                hist_sizes.append(hs)
                data_off += L

            nb = len(batch_idx)
            meta_arr = np.array(meta_rows, dtype=np.int32)
            cnt3_off = np.zeros(nb + 1, dtype=np.int64)
            for j in range(nb):
                cnt3_off[j + 1] = cnt3_off[j] + hist_sizes[j]

            xf_t = torch.from_numpy(np.concatenate(xf_parts)).to(device)
            xp_t = torch.from_numpy(np.concatenate(xp_parts)).to(device)
            yp_t = torch.from_numpy(np.concatenate(yp_parts)).to(device)
            meta_t = torch.from_numpy(meta_arr).to(device)
            off_t = torch.from_numpy(cnt3_off).to(device)
            cnt3 = torch.zeros(int(cnt3_off[nb]), dtype=torch.int32, device=device)

            out = mod.trace_inte_remap_launch(
                xf_t, xp_t, yp_t, meta_t, off_t, cnt3, block_size)
            batch_res = out.cpu().numpy()
            for j, i in enumerate(batch_idx):
                result[i] = max(batch_res[j], 0.0)

            del xf_t, xp_t, yp_t, meta_t, off_t, cnt3, out
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            n_batches += 1
            pos = end

        # ── Phase 4: CPU fallback for oversized tasks ──
        for i in cpu_tasks:
            result[i] = compute_inte_cpu(bin_data, i, network[i], tau=tau)

        if n_batches > 1 or cpu_tasks or n_coarsened:
            vprint(f"[TENEX] POINT InTE: {n_batches} GPU batch(es) "
                  f"({len(gpu_tasks)} tasks, {n_coarsened} coarsened), "
                  f"{len(cpu_tasks)} CPU fallback")

        return result

