"""
Fused Full-SMEM kernel for the surrogate-test inference method.

For each (target, source) variable pair, one CUDA block:
    1. computes ``te_obs`` (read once from the pre-supplied observed-TE matrix);
    2. iterates over ``n_surrogates`` block-permuted copies of the source time
       axis and recomputes the TE formula inside SMEM (same layout/phases as
       :mod:`tenex.kernels.full_smem`);
    3. accumulates ``sum_te``, ``sum_sq_te`` and ``count_ge`` in registers and
       writes the three scalars to the (i, j) slot of the pre-zeroed output
       tensors.

This removes per-iteration kernel-launch overhead and the (L, K) shuffle
memory traffic that the Python loop in
``tenex.inference.surrogate_test`` pays on every surrogate.
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
            import tenex._ext.full_smem_surrogate_test as _mod
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
            os.path.join(this_dir, '..', 'csrc', 'full_smem_surrogate_test.cu')
        )
        _module = load(
            name="full_smem_surrogate_test",
            sources=[cu_path],
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _module


# ── Public API ───────────────────────────────────────────────────────────────


def _next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def compute_full_smem_surrogate_test(
    bin_arrs: torch.Tensor,          # (n_vars, T, K) int on CUDA
    block_perm: torch.Tensor,        # (n_surrogates, n_vars, n_blocks) int32 CUDA
    observed_te: torch.Tensor,       # (n_vars, n_vars) float32 CUDA
    n_per_var: Optional[torch.Tensor],   # (n_vars,) int32 — for API parity; unused
    b_max: int,
    tau: int,
    block_length: int,
    n_surrogates: int,
    block_size: int = 256,
    pair_offset: int = 0,
    n_pairs_local: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused surrogate-test Full-SMEM kernel.

    Returns (sum_te, sum_sq_te, count_ge), each ``(n_vars, n_vars)`` on CUDA.
    The slots outside the local pair range ``[pair_offset, pair_offset +
    n_pairs_local)`` (including the diagonal) are zero.
    """
    assert bin_arrs.is_cuda, "bin_arrs must be on CUDA"
    assert block_perm.is_cuda, "block_perm must be on CUDA"
    assert observed_te.is_cuda, "observed_te must be on CUDA"
    assert block_perm.dtype == torch.int32, "block_perm must be int32"

    device = bin_arrs.device
    n_vars, T, K = bin_arrs.shape
    L = T - tau
    N = L * K
    b = int(b_max)
    b2 = b * b
    b3 = b * b * b

    # Block size: same rule as full_smem (power of 2, >= max(N, 2*B^2), <= 1024).
    bs = max(128, min(1024, _next_pow2(max(N, 2 * b2))))
    block_size = max(block_size, bs)
    warps = block_size // 32
    smem_bytes = b3 * 4 + b2 * 4 * 2 + b * 4 + warps * 4

    mod = _load_module()
    max_smem = int(mod.get_smem_optin())
    if max_smem <= 0:
        max_smem = torch.cuda.get_device_properties(device).shared_memory_per_block
    if smem_bytes > max_smem:
        raise RuntimeError(
            f"Fused surrogate Full-SMEM requires {smem_bytes} B SMEM but device "
            f"max is {max_smem} B (b_max={b_max}). Fall back to the "
            f"Python loop for Adaptive-SMEM etc."
        )

    if n_pairs_local is None:
        n_pairs_local = n_vars * (n_vars - 1) - pair_offset

    sum_te    = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    sum_sq_te = torch.zeros((n_vars, n_vars), dtype=torch.float32, device=device)
    count_ge  = torch.zeros((n_vars, n_vars), dtype=torch.int32,   device=device)

    # Ensure contiguous layout.
    bin_c  = bin_arrs.contiguous()
    perm_c = block_perm.contiguous()
    obs_c  = observed_te.contiguous()

    MAX_GRID = 2**31 - 1
    max_chunk = min(MAX_GRID, int(n_pairs_local))
    remaining = int(n_pairs_local)
    cur_offset = int(pair_offset)

    while remaining > 0:
        sz = min(remaining, max_chunk)
        mod.full_smem_surrogate_test_launch(
            bin_c, perm_c, obs_c,
            sum_te, sum_sq_te, count_ge,
            int(T), int(K), int(tau),
            int(b), int(b2), int(b3),
            int(block_length), int(n_surrogates),
            int(block_size),
            int(n_vars),
            int(cur_offset), int(sz),
        )
        remaining -= sz
        cur_offset += sz

    return sum_te, sum_sq_te, count_ge
