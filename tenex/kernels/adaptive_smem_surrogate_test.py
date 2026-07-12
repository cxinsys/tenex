"""
Fused Adaptive-SMEM kernel for the surrogate-test inference method.

Sister module of :mod:`tenex.kernels.full_smem_surrogate_test` but with
per-pair asymmetric bin sizing (``Bt = n_per_var[target]``,
``Bs = n_per_var[source]``). For each (target, source) pair, one CUDA block:

    1. computes ``te_obs`` (read once from the pre-supplied observed-TE matrix);
    2. iterates over ``n_surrogates`` block-permuted copies of the target and
       source variables and recomputes the TE formula inside SMEM (same layout
       and phases as :mod:`tenex.kernels.adaptive_smem`);
    3. accumulates ``sum_te``, ``sum_sq_te`` and ``count_ge`` in registers and
       writes the three scalars to the (i, j) slot of the pre-zeroed output
       tensors.

This removes per-iteration kernel-launch overhead and the (L, K) shuffle
memory traffic that the Python loop in
:mod:`tenex.inference.surrogate_test` pays on every surrogate — matching the
Full-SMEM fused path but for datasets whose ``b_max`` (after coarsening)
exceeds the Full-SMEM ceiling.
"""

import threading
from typing import Optional

import torch


# ── Module loading (compile once, cache) ──────────────────────────────────────

_module: object = None
_lock = threading.Lock()


def _load_module():
    """Try AOT extension first, then JIT-compile from the packaged .cu source."""
    global _module
    if _module is not None:
        return _module

    with _lock:
        if _module is not None:
            return _module

        # AOT build (from wheel / ``python setup.py build_ext --inplace``).
        try:
            import tenex._ext.adaptive_smem_surrogate_test as _mod
            _module = _mod
            return _module
        except ImportError:
            pass

        # JIT fall-back: compile the packaged .cu directly. We use
        # torch.utils.cpp_extension.load so we can point at the shipped file
        # rather than duplicating the source here.
        import os
        from torch.utils.cpp_extension import load

        this_dir = os.path.dirname(os.path.abspath(__file__))
        cu_path = os.path.normpath(
            os.path.join(this_dir, '..', 'csrc', 'adaptive_smem_surrogate_test.cu')
        )
        _module = load(
            name="adaptive_smem_surrogate_test",
            sources=[cu_path],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _module


# ── Public API ───────────────────────────────────────────────────────────────


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def _select_block_size_for_pairfree(N: int) -> int:
    """Block-size heuristic matching ``compute_te_adaptive_smem_pairfree``.

    Kept in sync with :mod:`tenex.kernels.adaptive_smem` (block_size dominated
    by L for large T, capped at 1024, lower bound 128). The surrogate-test
    Python dispatcher calls this helper so the feasibility check and the
    kernel launch agree on block size.
    """
    return max(128, min(1024, _next_pow2(max(int(N), 1))))


def _smem_bytes_for_pair(bt: int, bs: int, warps: int) -> int:
    """SMEM layout size for one Adaptive-SMEM pair (see .cu)."""
    return int((bt * bt * bs + bt * bt + bt * bs + bt + warps) * 4)


def _worst_case_smem_bytes(n_per_var, block_size: int, pair_offset: int,
                           n_pairs_local: int, n_vars: int) -> int:
    """Iterate over pairs in [pair_offset, pair_offset+n_pairs_local) and
    return the worst-case SMEM requirement in bytes.

    Mirrors the kernel's (target, source) decoding of the linear pair id.
    """
    import numpy as np
    warps = block_size // 32
    nm1 = n_vars - 1
    n_arr = np.asarray(n_per_var)
    # Fast path: global worst-case pair (max Bt, max Bs).
    max_b = int(n_arr.max())
    # A tight upper bound regardless of which pairs are in the range.
    return _smem_bytes_for_pair(max_b, max_b, warps)


def compute_adaptive_smem_surrogate_test(
    bin_arrs: torch.Tensor,          # (n_vars, T, K) int on CUDA
    n_per_var: torch.Tensor,         # (n_vars,) int32 CUDA
    block_perm: torch.Tensor,        # (n_surrogates, n_vars, n_blocks) int32 CUDA
    observed_te: torch.Tensor,       # (n_vars, n_vars) float32 CUDA
    tau: int,
    block_length: int,
    n_surrogates: int,
    block_size: Optional[int] = None,
    pair_offset: int = 0,
    n_pairs_local: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused surrogate-test Adaptive-SMEM kernel.

    Returns (sum_te, sum_sq_te, count_ge), each ``(n_vars, n_vars)`` on CUDA.
    The slots outside the local pair range ``[pair_offset, pair_offset +
    n_pairs_local)`` (including the diagonal) are zero.
    """
    assert bin_arrs.is_cuda, "bin_arrs must be on CUDA"
    assert n_per_var.is_cuda, "n_per_var must be on CUDA"
    assert block_perm.is_cuda, "block_perm must be on CUDA"
    assert observed_te.is_cuda, "observed_te must be on CUDA"
    assert n_per_var.dtype == torch.int32, "n_per_var must be int32"
    assert block_perm.dtype == torch.int32, "block_perm must be int32"

    device = bin_arrs.device
    n_vars, T, K = bin_arrs.shape
    L = T - tau
    N = L * K

    # Block size: same heuristic as compute_te_adaptive_smem_pairfree.
    bs_sel = _select_block_size_for_pairfree(N)
    if block_size is None:
        block_size = bs_sel
    else:
        block_size = max(int(block_size), bs_sel)
    warps = block_size // 32

    if n_pairs_local is None:
        n_pairs_local = n_vars * (n_vars - 1) - pair_offset

    # Worst-case SMEM bytes across all variables (safe upper bound for any
    # pair range). Matches the guard in
    # ``compute_te_adaptive_smem_pairfree``.
    n_arr_cpu = n_per_var.detach().cpu().numpy()
    max_b = int(n_arr_cpu.max())
    smem_bytes = _smem_bytes_for_pair(max_b, max_b, warps)

    mod = _load_module()
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block
    if smem_bytes > max_smem:
        raise RuntimeError(
            f"Fused surrogate Adaptive-SMEM requires {smem_bytes} B SMEM but "
            f"device max is {max_smem} B (b_max={max_b}). Ensure "
            f"coarsening is applied before calling this kernel, or fall back "
            f"to the Python loop path."
        )

    sum_te    = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    sum_sq_te = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    count_ge  = torch.zeros((n_vars, n_vars), dtype=torch.int32,   device=device)

    # Ensure contiguous layout.
    bin_c   = bin_arrs.contiguous()
    npv_c   = n_per_var.contiguous()
    perm_c  = block_perm.contiguous()
    obs_c   = observed_te.contiguous()

    MAX_GRID = 2**31 - 1
    max_chunk = min(MAX_GRID, int(n_pairs_local))
    remaining = int(n_pairs_local)
    cur_offset = int(pair_offset)

    while remaining > 0:
        sz = min(remaining, max_chunk)
        mod.adaptive_smem_surrogate_test_launch(
            bin_c, npv_c, perm_c, obs_c,
            sum_te, sum_sq_te, count_ge,
            int(T), int(K), int(tau),
            int(block_length), int(n_surrogates),
            int(block_size),
            int(smem_bytes),
            int(n_vars),
            int(cur_offset), int(sz),
        )
        remaining -= sz
        cur_offset += sz

    return sum_te, sum_sq_te, count_ge
