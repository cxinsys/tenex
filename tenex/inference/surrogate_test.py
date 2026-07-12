"""
Surrogate-based statistical test for Transfer Entropy.

Computes two quantities from N surrogate TE matrices. Each surrogate is
obtained by time-axis shuffling **every variable's series independently**
(both target and source of each candidate pair are shuffled, with
per-variable permutations). This is the "all-variable shuffle" null: it
breaks all temporal coupling in the system, not only the source→target
link. It is a stricter null than the classical source-only shuffle and
is what both the loop and the fused CUDA paths implement — they agree
bit-for-bit on both the observed TE and the null TE per pair.

From each surrogate matrix we derive:

- **Effective TE** (bias-corrected):  ``effective_te = te_observed - mean(te_surrogate)``
  Marschinski & Kantz, *Eur. Phys. J. B* 30:275 (2002).

- **Significance** against the empirical null via one of two modes:
    - ``p_method="parametric"`` (default): per-pair Gaussian z-test using
      the surrogate mean and std. Allows arbitrarily small p-values and
      scales to millions of pairs.
    - ``p_method="mc"``: Monte Carlo p-value
      ``p = (1 + #{te_surrogate >= te_observed}) / (1 + N)`` with +1
      smoothing (Phipson & Smyth 2010). Floor of ``1/(N+1)`` makes this
      mode unusable for large ``N_pairs × BH-FDR``.
  In both modes, Benjamini–Hochberg FDR is applied to off-diagonal pairs.

Surrogate generation methods (per the ``shuffle_method`` parameter):

- ``"block"``: split each variable's time series into blocks of length
  ``block_length`` (default = ``sqrt(T)``) and permute the block order
  **independently per variable**. Preserves marginal distribution and
  short-range autocorrelation within blocks.
  Reference: Lancaster et al., *Phys. Rep.* 748:1–60 (2018).

- ``"random"``: random permutation of time points (fastest, strictest
  null), again applied independently per variable.

The loop uses streaming accumulation (``sum``, ``sum_sq``, ``count_ge``) so
memory stays at ``O(n_vars^2)`` regardless of ``n_surrogates``.

Execution paths
---------------

For ``Full-SMEM`` and ``Adaptive-SMEM`` kernels there is a fused CUDA
variant that runs all ``N`` surrogate TE computations per pair inside a
single block (target history cached in SMEM, accumulators kept on the
GPU). The fused path saves kernel launches and accumulator round-trips
but pays an indirect-indexing cost in the histogram phase, so it only
wins for short time series. Crossover sits around ``L = T - tau < 1500``;
the dispatch in :class:`SurrogateTestMethod` picks fused vs loop
automatically based on ``L``. Override with the ``fused`` kwarg
(``True`` to force, ``False`` to disable, ``None`` for auto).

Other kernels (``scatter_add``, ``Triton``) always use the
loop path because no fused variant exists for them.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from tenex.inference import GRN
from tenex.inference import InferenceMethod
from tenex.inference import bh_fdr_gpu
from tenex.inference import build_pairs
from tenex.inference import make_grn
from tenex.kernels import get_kernel
from tenex._log import vprint


@dataclass
class SurrogateTestResult:
    """Result of :class:`SurrogateTestMethod`.

    Attributes
    ----------
    observed_te       : (n, n) float32 — original TE matrix.
    mean_surrogate_te : (n, n) float32 — average TE over ``n_surrogates`` shuffles.
    std_surrogate_te  : (n, n) float32 — std of TE over ``n_surrogates`` shuffles.
    effective_te      : (n, n) float32 — ``observed_te - mean_surrogate_te``.
    p_values          : (n, n) float32 — p-values (diagonal = 1). Parametric
                        (Gaussian z-test) or Monte Carlo depending on mode.
    grn               : :class:`GRN` — edges with FDR-adjusted p < ``fdr``.
    n_surrogates      : int
    shuffle_method    : str — ``"block"`` or ``"random"``.
    block_length      : int or None — block length used (for ``"block"``).
    p_method          : str — ``"parametric"`` or ``"mc"``.
    fdr               : float — FDR cutoff applied.
    """

    observed_te: np.ndarray
    mean_surrogate_te: np.ndarray
    std_surrogate_te: np.ndarray
    effective_te: np.ndarray
    p_values: np.ndarray
    grn: GRN
    n_surrogates: int
    shuffle_method: str
    block_length: int | None
    p_method: str
    fdr: float


def _block_shuffle_gpu(
    bin_arrs: torch.Tensor,
    block_length: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Shuffle each variable's time axis in blocks of length ``block_length``.

    Each variable receives an independent block-order permutation. Marginal
    distribution is preserved exactly; short-range autocorrelation is
    preserved within blocks.

    Parameters
    ----------
    bin_arrs : (n_vars, T, K) int tensor on CUDA.
    block_length : int — block size.
    rng : torch.Generator — per-device RNG (cuda).

    Returns
    -------
    Tensor of same shape/dtype/device with time axis block-shuffled.
    """
    n_vars, T, K = bin_arrs.shape
    if block_length <= 1 or block_length >= T:
        return _random_shuffle_gpu(bin_arrs, rng)

    n_blocks = T // block_length
    t_trunc = n_blocks * block_length

    # Reshape: (n_vars, n_blocks, block_length, K)
    head = bin_arrs[:, :t_trunc].reshape(n_vars, n_blocks, block_length, K)
    perm = torch.argsort(
        torch.rand(n_vars, n_blocks, device=bin_arrs.device, generator=rng),
        dim=1,
    )
    idx = perm[:, :, None, None].expand(-1, -1, block_length, K)
    shuffled_head = torch.gather(head, 1, idx).reshape(n_vars, t_trunc, K)

    # Leave the tail untouched (fewer than block_length samples)
    if T > t_trunc:
        tail = bin_arrs[:, t_trunc:]
        return torch.cat([shuffled_head, tail], dim=1)
    return shuffled_head


def _random_shuffle_gpu(
    bin_arrs: torch.Tensor,
    rng: torch.Generator,
) -> torch.Tensor:
    """Independent random permutation of time axis per variable."""
    n_vars, T, K = bin_arrs.shape
    perm = torch.argsort(
        torch.rand(n_vars, T, device=bin_arrs.device, generator=rng),
        dim=1,
    )
    idx = perm[:, :, None].expand(-1, -1, K)
    return torch.gather(bin_arrs, 1, idx)


def _build_block_perm(
    n_vars: int,
    T: int,
    L: int,
    shuffle_method: str,
    block_length: int,
    k_start: int,
    k_end: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Build the ``(n_surrogates_chunk, n_vars, n_blocks)`` int32 permutation
    tensor that the fused surrogate kernel consumes.

    The RNG schedule matches the existing Python loop exactly: surrogate ``k``
    is drawn with ``gen.manual_seed(seed + k)``, so bit-for-bit agreement
    with ``_block_shuffle_gpu`` / ``_random_shuffle_gpu`` is preserved.

    Returns ``(block_perm, effective_block_length, n_blocks)`` where the
    effective block length / n_blocks reflect the ``block_length > T`` → random
    fall-back inside ``_block_shuffle_gpu``.
    """
    if shuffle_method == 'random' or block_length <= 1 or block_length >= T:
        # _random_shuffle_gpu permutes over T (full time axis) using rand(n_vars, T).
        eff_block_length = 1
        n_blocks = T
    else:
        eff_block_length = int(block_length)
        n_blocks = T // eff_block_length  # matches reshape in _block_shuffle_gpu

    n_chunk = k_end - k_start
    out = torch.empty(
        (n_chunk, n_vars, n_blocks), dtype=torch.int32, device=device,
    )
    gen = torch.Generator(device=device)
    for ki, k in enumerate(range(k_start, k_end)):
        gen.manual_seed(seed + k)
        if shuffle_method == 'random' or block_length <= 1 or block_length >= T:
            rnd = torch.rand(n_vars, T, device=device, generator=gen)
        else:
            rnd = torch.rand(n_vars, n_blocks, device=device, generator=gen)
        out[ki].copy_(torch.argsort(rnd, dim=1).to(torch.int32))
    return out, eff_block_length, n_blocks


def _run_surrogate_chunk_fused(
    bin_cpu: torch.Tensor,
    observed_te: np.ndarray,
    device: torch.device,
    n_vars: int,
    b_max: int,
    tau: int,
    shuffle_method: str,
    block_length: int,
    k_start: int,
    k_end: int,
    seed: int,
    log_prefix: str = "",
    log_stride: int = 0,
    pair_offset: int = 0,
    n_pairs_local: int | None = None,
    perm_batch: int = 8,
    kernel_name: str = 'Full-SMEM',
    n_per_var_cpu: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused-kernel variant of :func:`_run_surrogate_chunk`.

    Supports both Full-SMEM and Adaptive-SMEM fused kernels. Dispatch is
    controlled by ``kernel_name``. The Adaptive-SMEM path additionally needs
    ``n_per_var_cpu`` (forwarded to the CUDA kernel as an int32 device
    tensor).
    """
    bin_tensor = bin_cpu.to(device)
    if bin_tensor.dtype not in (torch.int8, torch.int16, torch.int32):
        bin_tensor = bin_tensor.to(torch.int32)
    T, K = bin_tensor.shape[1], bin_tensor.shape[2]
    L = T - tau

    obs_t = torch.as_tensor(observed_te, dtype=torch.float32, device=device)

    if n_pairs_local is None:
        n_pairs_local = n_vars * (n_vars - 1) - pair_offset

    # Resolve kernel-specific compute callable.
    if kernel_name == 'Full-SMEM':
        from tenex.kernels.full_smem_surrogate_test import compute_full_smem_surrogate_test

        def _compute(bin_t, perm_t, eff_bl, n_chunk):
            return compute_full_smem_surrogate_test(
                bin_t, perm_t, obs_t, None,
                b_max=b_max, tau=tau,
                block_length=eff_bl,
                n_surrogates=n_chunk,
                pair_offset=pair_offset,
                n_pairs_local=n_pairs_local,
            )
    elif kernel_name == 'Adaptive-SMEM':
        if n_per_var_cpu is None:
            raise ValueError(
                "Adaptive-SMEM fused path requires ``n_per_var_cpu``"
            )
        from tenex.kernels.adaptive_smem_surrogate_test import (
            compute_adaptive_smem_surrogate_test,
        )
        n_per_var_t = n_per_var_cpu.to(device).to(torch.int32)

        def _compute(bin_t, perm_t, eff_bl, n_chunk):
            return compute_adaptive_smem_surrogate_test(
                bin_t, n_per_var_t, perm_t, obs_t,
                tau=tau,
                block_length=eff_bl,
                n_surrogates=n_chunk,
                pair_offset=pair_offset,
                n_pairs_local=n_pairs_local,
            )
    else:
        raise ValueError(
            f"fused surrogate dispatch does not support kernel {kernel_name!r}"
        )

    sum_te_total    = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    sum_sq_te_total = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    count_ge_total  = torch.zeros((n_vars, n_vars), dtype=torch.int32,   device=device)

    total = k_end - k_start
    # Batch surrogates into sub-chunks to cap ``block_perm`` memory
    # (n_chunk * n_vars * n_blocks * 4 bytes).
    for sub_start in range(k_start, k_end, perm_batch):
        sub_end = min(sub_start + perm_batch, k_end)
        block_perm, eff_bl, n_blocks = _build_block_perm(
            n_vars, T, L, shuffle_method, block_length,
            sub_start, sub_end, seed, device,
        )
        n_chunk = sub_end - sub_start

        sum_te, sum_sq_te, count_ge = _compute(bin_tensor, block_perm, eff_bl, n_chunk)
        sum_te_total    += sum_te
        sum_sq_te_total += sum_sq_te
        count_ge_total  += count_ge

        del block_perm
        if log_stride and (sub_start - k_start + n_chunk) // log_stride > (sub_start - k_start) // log_stride:
            done = sub_start - k_start + n_chunk
            vprint(f"[TENEX] surrogate_test: {log_prefix}{done}/{total}")

    # NOTE: the pair-free kernel never emits the diagonal (i, i); the
    # reference Python loop, by contrast, evaluates ``te_k >= observed_te``
    # over the full matrix and so increments count_ge[i, i] by 1 per
    # iteration (both sides are 0). The diagonal patch is applied by the
    # *caller* after cross-GPU reduction, not here, so multi-GPU fused
    # mode does not overcount by ``n_gpus``.
    sum_np    = sum_te_total.to(torch.float64).cpu().numpy()
    sum_sq_np = sum_sq_te_total.to(torch.float64).cpu().numpy()
    count_np  = count_ge_total.cpu().numpy()
    return sum_np, sum_sq_np, count_np


def _run_surrogate_chunk(
    bin_cpu: torch.Tensor,
    n_per_var_cpu: torch.Tensor,
    observed_te: np.ndarray,
    device: torch.device,
    kernel,
    n_vars: int,
    b_max: int,
    tau: int,
    shuffle_method: str,
    block_length: int,
    k_start: int,
    k_end: int,
    seed: int,
    log_prefix: str = "",
    log_stride: int = 0,
    kernel_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run surrogate iterations ``[k_start, k_end)`` on a single device.

    Each call generates shuffled copies, recomputes TE via the provided
    kernel, and accumulates streaming statistics locally. Returns the
    per-chunk (sum, sum_sq, count_ge) accumulators so callers can reduce
    across multiple device workers.

    When ``kernel_name`` is one of ``'Full-SMEM'`` / ``'Adaptive-SMEM'``
    the inner loop bypasses the existing ``.cpu().numpy()`` conversion
    in ``kernel.compute_pairfree`` and keeps entropies / accumulators on
    GPU — saving N D2H transfers and N CPU accumulator passes. For all
    other kernels it falls back to the CPU-accumulator path.

    Parameters
    ----------
    bin_cpu : (n, T, K) int tensor on CPU — transferred to ``device`` here.
    n_per_var_cpu : (n,) int tensor on CPU.
    observed_te : (n, n) observed TE matrix (CPU).
    device : target CUDA device (or CPU for debugging).
    kernel : pairwise TE kernel (with ``compute_pairfree`` method).
    k_start, k_end : surrogate index range (half-open).
    seed : base seed; the per-iteration RNG is seeded with ``seed + k``.
    log_prefix : optional string to tag progress prints (e.g. ``"gpu0 "``).
    log_stride : print every ``log_stride`` iterations; 0 disables.
    kernel_name : string name of the kernel. If ``'Full-SMEM'`` or
        ``'Adaptive-SMEM'``, the GPU-accumulator fast path is used.
    """
    bin_tensor = bin_cpu.to(device)
    if bin_tensor.dtype not in (torch.int8, torch.int16, torch.int32):
        bin_tensor = bin_tensor.to(torch.int32)
    n_per_var_t = n_per_var_cpu.to(device).to(torch.int32)

    gen = torch.Generator(device=device)

    # Fast path: keep everything on GPU for supported kernels.
    use_gpu_accum = (
        device.type == 'cuda'
        and kernel_name in ('Full-SMEM', 'Adaptive-SMEM')
    )

    if use_gpu_accum:
        call_pairfree = _PairfreeTensorCall(
            kernel_name, bin_tensor, n_per_var_t, b_max, tau, n_vars,
        )
        obs_gpu = torch.as_tensor(
            observed_te, dtype=torch.float32, device=device,
        )
        # float64 accumulators on GPU for numerical stability when
        # n_surrogates is large. Peak memory: 3 * n^2 * 8 bytes.
        sum_gpu    = torch.zeros((n_vars, n_vars), dtype=torch.float64, device=device)
        sum_sq_gpu = torch.zeros((n_vars, n_vars), dtype=torch.float64, device=device)
        count_gpu  = torch.zeros((n_vars, n_vars), dtype=torch.int32,   device=device)

        for k in range(k_start, k_end):
            gen.manual_seed(seed + k)
            if shuffle_method == 'block':
                bin_shuf = _block_shuffle_gpu(bin_tensor, block_length, gen)
            else:
                bin_shuf = _random_shuffle_gpu(bin_tensor, gen)

            ent_gpu = call_pairfree(bin_shuf)
            te_k_gpu = _pairfree_to_matrix_gpu(ent_gpu, n_vars)
            te_k_d = te_k_gpu.to(torch.float64)

            sum_gpu    += te_k_d
            sum_sq_gpu += te_k_d * te_k_d
            count_gpu  += (te_k_gpu >= obs_gpu).to(torch.int32)

            if log_stride and (k + 1) % log_stride == 0:
                vprint(f"[TENEX] surrogate_test: {log_prefix}{k + 1}/{k_end}")

        return (
            sum_gpu.cpu().numpy(),
            sum_sq_gpu.cpu().numpy(),
            count_gpu.cpu().numpy(),
        )

    # Fallback accumulator path for kernels without a GPU pair-free fast path:
    #   matrix kernels (GEMM-B2) via compute_matrix, per-pair kernels (e.g. CPU
    #   scatter_add) via explicit (target, source) pairs, and any kernel that
    #   does expose compute_pairfree. Each branch returns [i, j] = TE(i -> j)
    #   to match observed_te.
    _is_matrix    = getattr(kernel, "is_matrix_kernel", False)
    _has_pairfree = hasattr(kernel, "compute_pairfree")
    _sur_pairs = None
    if not _is_matrix and not _has_pairfree:
        _tgt, _src = np.meshgrid(np.arange(n_vars), np.arange(n_vars), indexing="ij")
        _msk = _tgt != _src
        _sur_pairs = np.stack([_tgt[_msk], _src[_msk]], axis=1).astype(np.int64)

    def _surrogate_te(bin_shuf):
        if _is_matrix:
            m = kernel.compute_matrix(bin_shuf, n_vars, tau, device)
            m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
            return m.T
        if _has_pairfree:
            ent = kernel.compute_pairfree(
                bin_shuf, n_vars, b_max, n_per_var_t, tau, device,
            )
            return _pairfree_to_matrix(ent, n_vars)
        ent = kernel.compute_single_gpu(
            bin_shuf, _sur_pairs, b_max, n_per_var_t, tau, None, device,
        )
        ent = ent.detach().cpu().numpy() if hasattr(ent, "detach") else np.asarray(ent)
        te = np.zeros((n_vars, n_vars), dtype=np.float32)
        te[_sur_pairs[:, 0], _sur_pairs[:, 1]] = ent
        return te.T

    sum_surr    = np.zeros((n_vars, n_vars), dtype=np.float64)
    sum_sq_surr = np.zeros((n_vars, n_vars), dtype=np.float64)
    count_ge    = np.zeros((n_vars, n_vars), dtype=np.int32)

    for k in range(k_start, k_end):
        gen.manual_seed(seed + k)
        if shuffle_method == 'block':
            bin_shuf = _block_shuffle_gpu(bin_tensor, block_length, gen)
        else:
            bin_shuf = _random_shuffle_gpu(bin_tensor, gen)

        te_k = _surrogate_te(bin_shuf)

        sum_surr    += te_k
        sum_sq_surr += te_k.astype(np.float64) ** 2
        count_ge    += (te_k >= observed_te).astype(np.int32)

        if log_stride and (k + 1) % log_stride == 0:
            vprint(f"[TENEX] surrogate_test: {log_prefix}{k + 1}/{k_end}")

    return sum_surr, sum_sq_surr, count_ge


def _pairfree_to_matrix(entropies: np.ndarray, n_vars: int) -> np.ndarray:
    """Reshape pair-free (n_pairs,) entropies into (n, n) TE matrix.

    Pair ordering used by the kernels:
      ``pair_id = target * (n - 1) + src_local``, with ``src_local``
      skipping the diagonal.

    Returns a matrix indexed as ``result[i, j] = TE(i -> j)``.
    """
    nm1 = n_vars - 1
    matrix = np.zeros((n_vars, n_vars), dtype=np.float32)
    ent2d = entropies.reshape(n_vars, nm1)
    for i in range(n_vars):
        matrix[i, :i] = ent2d[i, :i]
        matrix[i, i + 1:] = ent2d[i, i:]
    return matrix.T  # [i, j] = TE(i -> j)


def _pairfree_to_matrix_gpu(entropies: torch.Tensor, n_vars: int) -> torch.Tensor:
    """GPU version of :func:`_pairfree_to_matrix`.

    ``ent2d[t, src_local]`` is the TE of pair (target=t, source=src_local
    with src_local mapping around the diagonal). We want
    ``M[i, j] = TE(i -> j)`` where ``j != i``:

        result_intermediate[t, j] = ent2d[t, j - (1 if j > t else 0)]
                                   (diagonals zero)
        M = result_intermediate.T

    Fully vectorised; runs in one kernel launch for each elementwise op.
    """
    nm1 = n_vars - 1
    device = entropies.device
    ent2d = entropies.view(n_vars, nm1)
    row_idx = torch.arange(n_vars, device=device).view(n_vars, 1)
    col_idx = torch.arange(n_vars, device=device).view(1, n_vars)
    off_diag = col_idx != row_idx
    # src_local = j - (1 if j > t else 0); clamped for the diagonal column
    # (its value doesn't matter because we mask it out).
    src_local = (col_idx - (col_idx > row_idx).to(torch.long)).clamp(min=0, max=nm1 - 1)
    src_local = src_local.expand(n_vars, n_vars)
    matrix = ent2d.gather(1, src_local)
    matrix = matrix * off_diag.to(matrix.dtype)
    return matrix.T.contiguous()


class _PairfreeTensorCall:
    """Caches per-(kernel_name, T, K, tau, b_max) launch params so the
    surrogate loop does not re-derive block_size / SMEM layout each call.

    Returns entropies as a GPU ``torch.Tensor`` — the existing
    :func:`kernel.compute_pairfree` wrappers always perform an implicit
    D2H via ``.cpu().numpy()``; for the GPU-accumulator loop we need to
    keep them on the device.
    """

    def __init__(
        self,
        kernel_name: str,
        bin_arrs: torch.Tensor,
        n_per_var: torch.Tensor,
        b_max: int,
        tau: int,
        n_vars: int,
    ):
        self.kernel_name = kernel_name
        self.n_vars = n_vars
        self.tau = tau
        self.n_pairs_total = n_vars * (n_vars - 1)
        self.T, self.K = bin_arrs.shape[1], bin_arrs.shape[2]
        L = self.T - tau
        N = L * self.K
        self.n_per_var = n_per_var.contiguous()

        if kernel_name == 'Full-SMEM':
            from tenex.kernels.full_smem import _load_module, _next_pow2
            self.mod = _load_module()
            self.b_max = b_max
            self.b2 = b_max * b_max
            self.b3 = self.b2 * b_max
            self.block_size = max(128, min(1024, _next_pow2(max(N, 2 * self.b2))))
        elif kernel_name == 'Adaptive-SMEM':
            from tenex.kernels.adaptive_smem import _load_module, _next_pow2
            self.mod = _load_module()
            self.block_size = max(128, min(1024, _next_pow2(max(N, 1))))
            warps = max(1, self.block_size // 32)
            n_arr = n_per_var.cpu().numpy()
            b_max_arr = int(n_arr.max())
            self.smem_worst = int(
                (b_max_arr ** 3 + b_max_arr ** 2 * 2 + b_max_arr) * 4
                + warps * 4
            )
        else:
            raise NotImplementedError(
                f"_PairfreeTensorCall: {kernel_name!r} has no tensor path"
            )

    def __call__(self, bin_arrs: torch.Tensor) -> torch.Tensor:
        bin_c = bin_arrs.contiguous()
        if self.kernel_name == 'Full-SMEM':
            return self.mod.te_smem_launch_pairfree(
                bin_c,
                self.T, self.K, self.tau,
                self.b_max, self.b2, self.b3, self.block_size,
                self.n_vars, 0, self.n_pairs_total,
            )
        # Adaptive-SMEM
        return self.mod.te_adaptive_smem_launch_pairfree(
            bin_c, self.n_per_var,
            self.T, self.K, self.tau,
            self.block_size, self.smem_worst,
            self.n_vars, 0, self.n_pairs_total,
        )


class SurrogateTestMethod(InferenceMethod):
    """Surrogate-based statistical test with effective-TE bias correction.

    Produces:
      - Effective TE matrix (bias-corrected via surrogate mean subtraction)
      - Monte Carlo p-values (from empirical null distribution)
      - BH-FDR-thresholded :class:`GRN`
    """

    @property
    def name(self) -> str:
        return 'surrogate_test'

    def infer(self, te_matrix, variable_names, device, **kwargs) -> SurrogateTestResult:
        """Run surrogate test.

        Additional kwargs
        -----------------
        bin_data          : (n, T) or (n, T, K) int array — required.
        n_per_var         : (n,) int array — required.
        b_max          : int — required (global max bin count).
        kernel            : str — kernel name (e.g., ``"Full-SMEM"``); required.
        tau                : int, default 1.
        n_surrogates      : int, default 100.
        shuffle_method    : ``"block"`` | ``"random"``, default ``"block"``.
        block_length      : int or None, default None (``sqrt(T - tau)``).
        p_method          : ``"parametric"`` | ``"mc"``, default ``"parametric"``.
                            Parametric = per-pair Gaussian z-test (scales to
                            millions of pairs). MC = empirical rank with +1
                            smoothing (p ≥ 1/(N+1), only useful for small N_pairs).
        fdr               : float, default 0.01.
        seed              : int, default 42.
        """
        bin_data = kwargs.get('bin_data')
        n_per_var = kwargs.get('n_per_var')
        b_max = kwargs.get('b_max')
        kernel_name = kwargs.get('kernel')
        if bin_data is None or n_per_var is None or b_max is None or not kernel_name:
            raise ValueError(
                "surrogate_test requires bin_data, n_per_var, b_max, and kernel. "
                "Pass a TransferEntropyResult to NetWeaver (these are auto-filled)."
            )

        tau = int(kwargs.get('tau', 1))
        n_surrogates = int(kwargs.get('n_surrogates', 100))
        shuffle_method = kwargs.get('shuffle_method', 'block')
        block_length = kwargs.get('block_length')
        p_method = kwargs.get('p_method', 'parametric')
        fdr = float(kwargs.get('fdr', 0.01))
        seed = int(kwargs.get('seed', 42))
        devices = kwargs.get('devices', None)  # list[int] of CUDA device ids

        if shuffle_method not in ('block', 'random'):
            raise ValueError(
                f"shuffle_method must be 'block' or 'random' (got {shuffle_method!r})"
            )
        if n_surrogates < 2:
            # std_surrogate_te is degenerate with fewer than 2 samples, which
            # makes the parametric z-test NaN / ±inf and leaves MC p-values
            # pinned at 1/(1+N) with no useful resolution. Reject up front.
            raise ValueError(
                f"n_surrogates must be >= 2 (got {n_surrogates}); "
                f"parametric z-test needs a non-degenerate std estimate and "
                f"MC p-values need at least two samples to be informative."
            )
        if p_method not in ('parametric', 'mc'):
            raise ValueError(
                f"p_method must be 'parametric' or 'mc' (got {p_method!r})"
            )

        # Normalise bin_data shape to (n, T, K)
        bin_data = np.asarray(bin_data)
        if bin_data.ndim == 2:
            bin_data = bin_data[..., None]
        n_vars, T, K = bin_data.shape

        if T - tau < 2:
            raise ValueError(f"T - tau = {T - tau} is too short for TE computation")

        if block_length is None:
            block_length = max(2, int(np.sqrt(T - tau)))
        block_length = int(block_length)

        kernel = get_kernel(kernel_name)
        observed_te = np.asarray(te_matrix, dtype=np.float32)

        # CPU-side pinned tensors (one H2D copy per worker in multi-GPU path).
        bin_cpu = torch.as_tensor(bin_data)
        n_per_var_cpu = torch.as_tensor(np.asarray(n_per_var))

        # Resolve devices: kwargs['devices'] > device param
        if devices is None or len(devices) == 0:
            device_ids = [device.index] if device.type == 'cuda' else []
        else:
            device_ids = list(devices)
        n_gpus = len(device_ids)

        vprint(f"[TENEX] surrogate_test: {n_surrogates} shuffles "
              f"(method={shuffle_method}, block_length={block_length}, "
              f"kernel={kernel_name}, p={p_method}, "
              f"devices={device_ids or 'cpu'})")

        # Prefer the fused kernel when applicable — it computes all N+1 TE
        # matrices per pair in a single block, keeping sum / sum_sq /
        # count_ge accumulators entirely on the GPU. Both Full-SMEM and
        # Adaptive-SMEM have fused variants; all other kernels fall through
        # to the Python loop path.
        #
        # Heuristic: the fused kernel reuses the target's per-block SMEM but
        # pays per-thread indirect indexing during the permuted source/target
        # reads in Phase 1, plus larger SMEM traffic per surrogate in
        # Phases 2-3. Empirically it wins for short time series but loses on
        # long ones (crossover ~L=1500). Pre-shuffling + a coalesced kernel
        # call (the loop path) wins for long L because the indirection cost
        # dominates the launch / accumulator savings.
        #
        # The ``fused`` kwarg overrides:
        #   None  (default)  -> auto-select via heuristic above
        #   True             -> force fused (raises if unsupported)
        #   False            -> force loop
        L_eff = T - tau
        FUSED_L_THRESHOLD = 1500  # mESC L=458 wins; Skin L=7489 / Zebrafish L=26021 lose
        fused_supported = kernel_name in ('Full-SMEM', 'Adaptive-SMEM')
        fused_kw = kwargs.get('fused', None)
        force_fused = fused_kw is True
        if fused_kw is None:
            use_fused = fused_supported and n_gpus >= 1 and L_eff < FUSED_L_THRESHOLD
        elif force_fused:
            if not fused_supported:
                raise ValueError(
                    f"fused=True requested but kernel {kernel_name!r} has no "
                    f"fused variant (supported: 'Full-SMEM', 'Adaptive-SMEM')"
                )
            if n_gpus == 0:
                raise ValueError(
                    "fused=True requires a CUDA device; none resolved from "
                    f"kwargs['devices']={devices!r} or engine device={device!r}"
                )
            use_fused = True
        else:  # fused_kw is False
            use_fused = False

        def _fused_unavailable(msg: str) -> None:
            """Handle a preflight failure: raise when forced, else warn + fallback."""
            if force_fused:
                raise RuntimeError(f"fused=True preflight failed: {msg}")
            vprint(f"[TENEX] surrogate_test: {msg}; falling back to loop")

        if use_fused:
            try:
                if kernel_name == 'Full-SMEM':
                    from tenex.kernels.full_smem_surrogate_test import (
                        _load_module as _load_fused,
                    )
                    mod = _load_fused()
                    max_smem = int(mod.get_smem_optin())
                    if max_smem <= 0:
                        max_smem = (
                            torch.cuda.get_device_properties(
                                torch.device(f"cuda:{device_ids[0]}")
                            ).shared_memory_per_block
                        )
                    b3 = b_max ** 3
                    b2 = b_max * b_max
                    N = (T - tau) * K
                    bs = max(128, min(1024, 1 << max(0, (max(N, 2 * b2) - 1).bit_length())))
                    warps = bs // 32
                    smem_needed = b3 * 4 + b2 * 4 * 2 + b_max * 4 + warps * 4
                    if smem_needed > max_smem:
                        _fused_unavailable(
                            f"Full-SMEM fused path needs {smem_needed} B SMEM "
                            f"but device opt-in is {max_smem} B"
                        )
                        use_fused = False
                else:  # Adaptive-SMEM
                    from tenex.kernels.adaptive_smem_surrogate_test import (
                        _load_module as _load_fused,
                        _select_block_size_for_pairfree,
                        _smem_bytes_for_pair,
                    )
                    mod = _load_fused()
                    max_smem = int(mod.get_smem_optin())
                    if max_smem <= 0:
                        max_smem = (
                            torch.cuda.get_device_properties(
                                torch.device(f"cuda:{device_ids[0]}")
                            ).shared_memory_per_block
                        )
                    # Per-pair SMEM feasibility: worst case across pairs.
                    npv_arr = np.asarray(n_per_var).astype(np.int32)
                    N = (T - tau) * K
                    bs = _select_block_size_for_pairfree(N)
                    warps = bs // 32
                    # Check the global worst-case (max Bt, max Bs) — tight
                    # upper bound for any pair.
                    max_b = int(npv_arr.max())
                    smem_needed = _smem_bytes_for_pair(max_b, max_b, warps)
                    if smem_needed > max_smem:
                        _fused_unavailable(
                            f"Adaptive-SMEM fused path needs {smem_needed} B "
                            f"SMEM but device opt-in is {max_smem} B"
                        )
                        use_fused = False
            except Exception as _e:
                if force_fused:
                    raise
                vprint(f"[TENEX] surrogate_test: fused init failed ({_e!r}); "
                      f"falling back to loop")
                use_fused = False

        if use_fused:
            vprint(f"[TENEX] surrogate_test: fused {kernel_name} path")
            # Adaptive-SMEM needs n_per_var on each worker's device.
            fused_kwargs = dict(kernel_name=kernel_name)
            if kernel_name == 'Adaptive-SMEM':
                fused_kwargs['n_per_var_cpu'] = n_per_var_cpu

            if n_gpus <= 1:
                dev_id = device_ids[0]
                dev = torch.device(f'cuda:{dev_id}')
                sum_surr, sum_sq_surr, count_ge = _run_surrogate_chunk_fused(
                    bin_cpu, observed_te, dev,
                    n_vars, b_max, tau, shuffle_method, block_length,
                    0, n_surrogates, seed,
                    log_stride=max(1, n_surrogates // 10),
                    **fused_kwargs,
                )
            else:
                # Multi-GPU fused path: partition pairs (not surrogates) so
                # each GPU owns a non-overlapping (target, source) range and
                # reruns the full N surrogates there.
                n_pairs_total = n_vars * (n_vars - 1)
                boundaries_p = np.linspace(0, n_pairs_total, n_gpus + 1, dtype=int)

                def _worker_fused(rank: int):
                    dev_id = device_ids[rank]
                    torch.cuda.set_device(dev_id)
                    dev = torch.device(f'cuda:{dev_id}')
                    p_beg = int(boundaries_p[rank])
                    p_end = int(boundaries_p[rank + 1])
                    return _run_surrogate_chunk_fused(
                        bin_cpu, observed_te, dev,
                        n_vars, b_max, tau, shuffle_method, block_length,
                        0, n_surrogates, seed,
                        log_prefix=f"gpu{dev_id} ",
                        log_stride=max(1, n_surrogates // 5),
                        pair_offset=p_beg,
                        n_pairs_local=p_end - p_beg,
                        **fused_kwargs,
                    )

                with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                    chunks = list(pool.map(_worker_fused, range(n_gpus)))

                # Each GPU wrote into a disjoint (i,j) range; sum works
                # because the off-range slots are zero.
                sum_surr = sum(c[0] for c in chunks)
                sum_sq_surr = sum(c[1] for c in chunks)
                count_ge = sum(c[2] for c in chunks)

            # Fused kernels skip the diagonal entirely. The loop/Python
            # reference path evaluates ``te_k >= observed_te`` over the
            # whole matrix and thus increments ``count_ge[i, i]`` by 1 per
            # iteration (both are 0). Apply that same +N boost here, once,
            # after any cross-GPU reduction — if we did it per worker the
            # multi-GPU path would overcount by ``n_gpus``.
            diag = np.arange(n_vars)
            count_ge[diag, diag] += n_surrogates
        elif n_gpus <= 1:
            dev = (torch.device(f'cuda:{device_ids[0]}')
                   if device_ids else torch.device('cpu'))
            sum_surr, sum_sq_surr, count_ge = _run_surrogate_chunk(
                bin_cpu, n_per_var_cpu, observed_te, dev, kernel,
                n_vars, b_max, tau, shuffle_method, block_length,
                0, n_surrogates, seed,
                log_stride=max(1, n_surrogates // 10),
                kernel_name=kernel_name,
            )
        else:
            # Partition surrogates across GPUs (contiguous chunks so seeds
            # remain reproducible and non-overlapping).
            boundaries = np.linspace(0, n_surrogates, n_gpus + 1, dtype=int)

            def _worker(rank: int):
                dev_id = device_ids[rank]
                torch.cuda.set_device(dev_id)
                dev = torch.device(f'cuda:{dev_id}')
                return _run_surrogate_chunk(
                    bin_cpu, n_per_var_cpu, observed_te, dev, kernel,
                    n_vars, b_max, tau, shuffle_method, block_length,
                    int(boundaries[rank]), int(boundaries[rank + 1]), seed,
                    log_prefix=f"gpu{dev_id} ",
                    log_stride=max(1, (boundaries[rank + 1]
                                       - boundaries[rank]) // 5),
                    kernel_name=kernel_name,
                )

            with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                chunks = list(pool.map(_worker, range(n_gpus)))

            sum_surr = sum(c[0] for c in chunks)
            sum_sq_surr = sum(c[1] for c in chunks)
            count_ge = sum(c[2] for c in chunks)

        mean_surr = (sum_surr / n_surrogates).astype(np.float32)
        var_surr = sum_sq_surr / n_surrogates - (sum_surr / n_surrogates) ** 2
        std_surr = np.sqrt(np.maximum(var_surr, 0.0)).astype(np.float32)
        effective_te = (observed_te - mean_surr).astype(np.float32)

        if p_method == 'parametric':
            # Per-pair one-sided z-test against Gaussian null fit to surrogates.
            # Uses complementary error function for numerical stability at small p.
            from scipy.special import erfc
            eps = np.float32(1e-12)
            z = (observed_te - mean_surr) / np.maximum(std_surr, eps)
            p_values = (0.5 * erfc(z / np.sqrt(2.0))).astype(np.float32)
        else:
            # Monte Carlo p-value with +1 smoothing (Phipson & Smyth 2010)
            p_values = ((1.0 + count_ge) / (1.0 + n_surrogates)).astype(np.float32)

        np.fill_diagonal(p_values, 1.0)

        # BH-FDR on the computed off-diagonal pairs only. When a source (TF)
        # filter was used for the observed matrix, only the source -> target
        # pairs were meaningfully computed and the remaining rows are structural
        # zeros. Restricting BH to those pairs keeps the correction denominator
        # equal to the number of tested pairs instead of the full n*(n-1),
        # which would otherwise make the FDR threshold far too conservative.
        sources = kwargs.get('sources')
        if sources is not None:
            pairs_bh = build_pairs(np.asarray(variable_names), sources)
            mask = np.zeros((n_vars, n_vars), dtype=bool)
            mask[pairs_bh[:, 0], pairs_bh[:, 1]] = True
        else:
            mask = ~np.eye(n_vars, dtype=bool)
        p_off = p_values[mask]
        pvals_t = torch.from_numpy(p_off).to(device)
        fdr_adj = bh_fdr_gpu(pvals_t, alpha=fdr).cpu().numpy()
        sig_off = fdr_adj < fdr

        # Reconstruct mask and enforce positive effective TE
        sig_full = np.zeros((n_vars, n_vars), dtype=bool)
        sig_full[mask] = sig_off
        sig_full &= (effective_te > 0)

        pairs = np.argwhere(sig_full).astype(np.int64)
        te_values = effective_te[pairs[:, 0], pairs[:, 1]]
        grn = make_grn(np.asarray(variable_names), pairs, te_values)
        vprint(f"[TENEX] surrogate_test: {len(grn)} significant edges (FDR<{fdr})")

        return SurrogateTestResult(
            observed_te=observed_te,
            mean_surrogate_te=mean_surr,
            std_surrogate_te=std_surr,
            effective_te=effective_te,
            p_values=p_values,
            grn=grn,
            n_surrogates=n_surrogates,
            shuffle_method=shuffle_method,
            block_length=block_length if shuffle_method == 'block' else None,
            p_method=p_method,
            fdr=fdr,
        )
