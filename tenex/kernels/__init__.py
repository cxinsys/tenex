"""
TENEX kernel registry — Strategy Pattern for Transfer Entropy computation.

Each kernel implements the TEKernel interface. The registry orders kernels
by priority so auto_select() returns the fastest feasible kernel for the
given data configuration.

Priority (highest first):
  1. GEMM-B2       — binary data (b_max=2), no TF filter, CUDA
  2. Full-SMEM     — cnt3 fits in device SMEM, CUDA
  3. Adaptive-SMEM — any b_max, CUDA (with optional bin coarsening)
  4. ScatterAdd    — universal fallback (CPU or GPU)
"""


import time
from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import torch

from tenex._log import vprint


# ── Utilities ────────────────────────────────────────────────────────────────

def _try_pin(t: torch.Tensor) -> torch.Tensor:
    """Pin a CPU tensor for async H2D transfer. Falls back to unpinned on failure."""
    if not t.is_cuda and not t.is_pinned():
        try:
            return t.contiguous().pin_memory()
        except RuntimeError:
            pass
    return t


def _next_pow2(x: int) -> int:
    """Round up to next power of 2."""
    return 1 << max(0, (x - 1).bit_length())


def query_smem_optin(device_ids=0) -> int:
    """Query device SMEM opt-in size (bytes).

    Accepts either an int device index or an iterable of indices.  When a
    list/tuple is given, returns the minimum opt-in across the listed
    devices so heterogeneous multi-GPU runs plan for the weakest GPU.
    Returns 48 KB if SMEM cannot be queried (driver/extension failure).
    """
    if isinstance(device_ids, (list, tuple)):
        ids = list(device_ids)
        if not ids:
            return 48 * 1024
    else:
        ids = [int(device_ids)]

    try:
        from tenex.kernels.full_smem import _load_module
        prev_dev = torch.cuda.current_device()
        try:
            mins = []
            for d in ids:
                torch.cuda.set_device(int(d))
                mins.append(int(_load_module().get_smem_optin()))
            return min(mins) if mins else 48 * 1024
        finally:
            torch.cuda.set_device(prev_dev)
    except Exception:
        return 48 * 1024


def compute_smem_bytes(b_max: int, n_time: int, n_kernels: int, tau: int) -> int:
    """Compute SMEM requirement for the Full-SMEM kernel layout (bytes)."""
    b2 = b_max * b_max
    b3 = b_max ** 3
    N = (n_time - tau) * n_kernels
    _bs = max(128, min(1024, _next_pow2(max(N, 2 * b2))))
    return b3 * 4 + b2 * 4 * 2 + b_max * 4 + (_bs // 32) * 4


# ── TEKernel ABC ─────────────────────────────────────────────────────────────

class TEKernel(ABC):
    """Abstract base class for all TE computation kernels."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable kernel name for logging (e.g. 'Full-SMEM')."""

    @property
    def is_matrix_kernel(self) -> bool:
        """True for kernels that compute the full (n,n) matrix directly."""
        return False

    @property
    def supports_pairfree(self) -> bool:
        """True if kernel can compute TE from linear pair indices (no pairs tensor)."""
        return False

    @abstractmethod
    def supports(
        self,
        b_max: int,
        on_cuda: bool,
        smem_optin: int,
        smem_bytes: int,
        n_per_var: Optional[torch.Tensor],
        source_filter: bool,
    ) -> bool:
        """Return True if this kernel can handle the given data configuration."""

    def peak_batch_bytes(
        self,
        b_max: int,
        T: int,
        K: int,
        batch_size: int,
        tau: int = 1,
    ) -> int:
        """Estimate peak GPU memory (bytes) the kernel needs per batch.

        Used by the orchestrator to reserve memory when deciding how many
        pairs to transfer to GPU.  Default implementation uses
        scatter_add's precise formula (conservative upper bound for all
        kernels).  CUDA kernels that use SMEM can override with a
        tighter estimate.
        """
        from tenex.kernels.scatter_add import _peak_bytes_per_pair
        return _peak_bytes_per_pair(b_max, T, K, tau) * batch_size

    @abstractmethod
    def compute_single_gpu(
        self,
        bin_arrs: torch.Tensor,
        pairs: torch.Tensor,
        b_max: int,
        n_per_var: torch.Tensor,
        tau: int,
        batch_size: Optional[int],
        device: torch.device,
        **kwargs,
    ) -> np.ndarray:
        """Compute TE for all pairs on a single device. Returns (n_pairs,) float32."""

    def compute_multi_gpu(
        self,
        bin_arrs_cpu: torch.Tensor,
        pairs_np: np.ndarray,
        b_max: int,
        n_per_var_cpu: torch.Tensor,
        tau: int,
        batch_size: Optional[int],
        device_ids: list[int],
        **kwargs,
    ) -> np.ndarray:
        """Multi-GPU dispatch with pinned memory + async transfer."""
        import math
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_gpus = len(device_ids)
        n_pairs = len(pairs_np)
        chunk = math.ceil(n_pairs / n_gpus)
        results: list[Optional[np.ndarray]] = [None] * n_gpus

        # Pin CPU tensors for async H2D transfer
        bin_pinned = _try_pin(bin_arrs_cpu)
        npg_pinned = _try_pin(n_per_var_cpu)

        def _run(rank: int):
            dev_id = device_ids[rank]
            torch.cuda.set_device(dev_id)
            dev = torch.device(f'cuda:{dev_id}')
            beg = rank * chunk
            end = min(beg + chunk, n_pairs)
            if beg >= n_pairs:
                return rank, np.empty(0, dtype=np.float32)
            # Async transfer using non_blocking (effective with pinned memory)
            arr_d = bin_pinned.to(dev, non_blocking=True)
            npg_d = npg_pinned.to(dev, non_blocking=True)

            # Check if pairs chunk fits in GPU memory; if not, sub-chunk
            local_pairs = pairs_np[beg:end]
            pairs_bytes = local_pairs.nbytes
            free_mem, _ = torch.cuda.mem_get_info(dev)
            T_dim, K_dim = arr_d.shape[1], arr_d.shape[2]
            kernel_reserve = self.peak_batch_bytes(
                b_max, T_dim, K_dim, batch_size or 4096, tau,
            )
            pairs_budget = max(0, free_mem - kernel_reserve)

            if pairs_bytes <= pairs_budget:
                p_d = torch.from_numpy(local_pairs).to(dev)
                ents = self.compute_single_gpu(
                    arr_d, p_d, b_max, npg_d, tau, batch_size, dev, **kwargs
                )
            else:
                # Sub-chunk to avoid OOM
                bytes_per_pair = 16  # 2 × int64
                max_pairs = max(1024, pairs_budget // bytes_per_pair)
                n_local = end - beg
                sub_chunks = []
                for si in range(0, n_local, max_pairs):
                    se = min(si + max_pairs, n_local)
                    p_sub = torch.from_numpy(local_pairs[si:se]).to(dev)
                    ent_sub = self.compute_single_gpu(
                        arr_d, p_sub, b_max, npg_d, tau, batch_size, dev,
                        **kwargs
                    )
                    sub_chunks.append(ent_sub)
                    del p_sub
                ents = np.concatenate(sub_chunks)
            return rank, ents

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {pool.submit(_run, i): i for i in range(n_gpus)}
            for fut in as_completed(futures):
                rank, ents = fut.result()
                results[rank] = ents

        return np.concatenate([r for r in results if r is not None and len(r) > 0])

    # ── Matrix kernel interface (override for GEMM-B2) ───────────

    def compute_matrix(
        self,
        data,
        n_vars: int,
        tau: int,
        device: torch.device,
        **kwargs,
    ) -> np.ndarray:
        """Compute full (n_vars, n_vars) TE matrix. Override for matrix kernels."""
        raise NotImplementedError(f"{self.name} is a pair-based kernel")

    def compute_matrix_multi_gpu(
        self,
        data,
        n_vars: int,
        tau: int,
        device_ids: list[int],
        **kwargs,
    ) -> np.ndarray:
        """Multi-GPU full matrix computation. Override for matrix kernels."""
        raise NotImplementedError(f"{self.name} is a pair-based kernel")

    # ── Batch-size selection ─────────────────────────────────────────────

    _L2_MULTIPLIER = 350  # optimal_working_set ~ 350 * L2_cache_size
    _FALLBACK_VRAM_FRAC = 0.17  # when L2 size unavailable

    def peak_bytes_per_pair(
        self, b_max: int, T: int, K: int, tau: int = 1,
    ) -> int:
        """
        Estimated peak GPU memory per variable pair for this kernel.

        Default uses scatter_add formula (conservative upper bound).
        CUDA kernels should override with their actual (much smaller) peak.
        """
        from tenex.kernels.scatter_add import _peak_bytes_per_pair
        return _peak_bytes_per_pair(b_max, T, K, tau)

    def heuristic_batch_size(
        self,
        bin_arrs: torch.Tensor,
        b_max: int,
        tau: int = 1,
    ) -> int:
        """
        VRAM-aware batch size heuristic (deterministic, zero cost).

        Two-stage budget calculation:

        1. **L2-based target** (throughput-optimal working set):
           Larger L2 allows larger batches before cache thrashing.
           Calibrated on RTX 3090 (6 MB L2) and PRO 6000 BW (128 MB L2).

        2. **VRAM safety cap** (prevents OOM on small-VRAM GPUs):
           Subtracts a reserve from free VRAM to account for PyTorch
           allocator fragmentation and CUDA driver overhead.
           Uses 80% of free VRAM (not 90%) to provide extra headroom
           on GPUs like RTX 2080 Ti (11 GB) where every MB counts.

        The final budget is min(L2_target, VRAM_cap).
        Subclasses may override for kernel-specific heuristics.
        """
        device = bin_arrs.device
        if not device.type.startswith('cuda'):
            return 2048

        T, K = bin_arrs.shape[1], bin_arrs.shape[2]
        peak_pp = self.peak_bytes_per_pair(b_max, T, K, tau)

        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)

        # Stage 1: L2-based throughput target
        props = torch.cuda.get_device_properties(device)
        l2 = getattr(props, 'L2_cache_size', 0)
        if l2 > 0:
            target = self._L2_MULTIPLIER * l2
        else:
            target = int(total * self._FALLBACK_VRAM_FRAC)

        # Stage 2: VRAM safety cap
        # Reserve 128 MB or 20% of free VRAM (whichever is larger) for
        # allocator overhead, fragmentation, and CUDA driver state.
        # This is critical for small-VRAM GPUs (e.g. 2080 Ti 11 GB).
        reserve = max(128 * 1024 * 1024, int(free * 0.20))
        cap = max(0, free - reserve)

        budget = min(target, cap)

        bs = max(64, budget // max(peak_pp, 1))
        bs = (bs // 64) * 64
        return bs

    # Module-level cache: (kernel_name, B, T, K, tau, dev_idx) -> batch_size
    _autotune_cache: dict[tuple, int] = {}

    def autotune_batch_size(
        self,
        bin_arrs: torch.Tensor,
        b_max: int,
        n_per_var: torch.Tensor,
        tau: int,
        device: torch.device,
    ) -> int:
        """
        Auto-tune batch_size by geometric doubling with early stopping.

        Runs the actual kernel at ~7 candidate batch sizes (from max/64
        to max, doubling each step) and picks the one with the highest
        throughput.  Stops early when throughput declines for 2 consecutive
        steps.  Results are cached per (kernel, data_shape, GPU).

        This is generic and works for any TEKernel subclass because it
        calls ``compute_single_gpu`` with real data.

        Cost: ~1-3 seconds on first call.  Zero on subsequent calls.
        Thread-safe for multi-GPU (each device auto-tunes independently).
        """
        T, K = bin_arrs.shape[1], bin_arrs.shape[2]
        dev_idx = device.index if device.index is not None else 0
        key = (self.name, b_max, T, K, tau, dev_idx)

        cached = TEKernel._autotune_cache.get(key)
        if cached is not None:
            return cached

        n_vars = bin_arrs.shape[0]

        # Rough upper bound: 90% of free memory / estimated bytes per pair
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        # Estimate peak per pair (use scatter_add formula as conservative bound)
        from tenex.kernels.scatter_add import _peak_bytes_per_pair
        peak_pp = _peak_bytes_per_pair(b_max, T, K, tau)
        max_bs = int(free * 0.90) // max(peak_pp, 1)
        max_bs = max(64, (max_bs // 64) * 64)

        # ~7 candidates: start from max/64, double up to max
        start = max(64, (max_bs // 64 // 64) * 64)

        best_bs = start
        best_pps = 0.0
        decline = 0
        n_tested = 0

        bs = start
        while bs <= max_bs and decline < 2:
            actual = min((bs // 64) * 64, max_bs)
            actual = max(actual, 64)

            # Generate random pairs for calibration
            pairs = torch.stack([
                torch.randint(0, n_vars, (actual,), dtype=torch.int64),
                torch.randint(0, n_vars, (actual,), dtype=torch.int64),
            ], dim=1).to(device)

            try:
                # warmup
                self.compute_single_gpu(
                    bin_arrs, pairs, b_max, n_per_var,
                    tau, actual, device,
                )
                torch.cuda.synchronize()

                # measure
                t0 = time.perf_counter()
                self.compute_single_gpu(
                    bin_arrs, pairs, b_max, n_per_var,
                    tau, actual, device,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                pps = actual / max(t1 - t0, 1e-9)
                n_tested += 1

                if pps > best_pps:
                    best_pps = pps
                    best_bs = actual
                    decline = 0
                else:
                    decline += 1

            except RuntimeError:
                # OOM or other error at this batch size: stop growing
                decline = 2

            del pairs
            torch.cuda.empty_cache()
            bs *= 2

        TEKernel._autotune_cache[key] = best_bs
        vprint(f"[TENEX] auto-tuned batch_size={best_bs:,} "
              f"({best_pps:,.0f} pairs/s) for {self.name} "
              f"on cuda:{dev_idx} [{n_tested} sizes tested]")
        return best_bs

    # ── Hooks ────────────────────────────────────────────────────────────

    def prepare(
        self,
        bin_arrs: torch.Tensor,
        n_per_var: torch.Tensor,
        b_max: int,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Optional preprocessing hook (e.g., bin coarsening). Default: no-op."""
        return bin_arrs, n_per_var, b_max

    def log_description(self, **kwargs) -> str:
        """Descriptive string for logging. Override for extra detail."""
        b_max = kwargs.get('b_max')
        if b_max is not None:
            return f"{self.name} (b_max={b_max})"
        return self.name


# ── Kernel Registry ──────────────────────────────────────────────────────────

_REGISTRY: list[TEKernel] = []
_REGISTERED = False


def register(kernel: TEKernel) -> TEKernel:
    """Register a kernel instance. Returns the kernel for decorator-style use."""
    _REGISTRY.append(kernel)
    return kernel


def get_kernel(name: str) -> TEKernel:
    """Look up a registered kernel by name (case-insensitive).

    Raises KeyError if not found.
    """
    needle = name.lower()
    for k in _REGISTRY:
        if k.name.lower() == needle:
            return k
    raise KeyError(
        f"Kernel '{name}' not found. Available: "
        f"{[k.name for k in _REGISTRY]}"
    )


def auto_select(
    b_max: int,
    on_cuda: bool,
    smem_optin: int,
    smem_bytes: int,
    n_per_var: Optional[torch.Tensor],
    source_filter: bool,
    exclude: Optional[set[str]] = None,
) -> TEKernel:
    """Select the best kernel for the given configuration (priority-ordered)."""
    for kernel in _REGISTRY:
        if exclude and kernel.name in exclude:
            continue
        if kernel.supports(b_max, on_cuda, smem_optin, smem_bytes,
                           n_per_var, source_filter):
            return kernel
    raise RuntimeError(
        f"No suitable TE kernel found for b_max={b_max}, "
        f"on_cuda={on_cuda}, smem_optin={smem_optin}"
    )


def registered_kernels() -> list[TEKernel]:
    """Return the current priority-ordered list of registered kernels."""
    return list(_REGISTRY)


# ── Register all kernels (import triggers registration) ──────────────────────

def _register_all():
    """Import all kernel modules to trigger registration. Called once."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from tenex.kernels.gemm_b2 import GEMMB2Kernel
    from tenex.kernels.full_smem import FullSMEMKernel
    from tenex.kernels.adaptive_smem import AdaptiveSMEMKernel
    from tenex.kernels.scatter_add import ScatterAddKernel

    register(GEMMB2Kernel())
    register(FullSMEMKernel())
    register(AdaptiveSMEMKernel())
    register(ScatterAddKernel())  # universal fallback (CPU or GPU)


_register_all()
