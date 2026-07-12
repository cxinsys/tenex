"""
TENEX — a high-performance Transfer Entropy computation engine.

Design goals
------------
1. Replace the 4x torch.unique bottleneck with scatter_add (O(N) vs O(N log N)).
2. Move discretization to the GPU (PyTorch) instead of CPU (NumPy).
3. Multiple optimized kernel backends via Strategy Pattern (auto-selected).
4. Multi-GPU support via Python threads — no multiprocessing spawn overhead.

Kernel selection (auto mode — adaptive based on data and hardware):
  1. GEMM-B2       — 3 GEMMs + Triton fused kernel (~950 M pairs/s, EXACT)
  2. Full-SMEM     — cnt3 in shared memory, no global-memory traffic  (~45 M pairs/s)
  3. Adaptive-SMEM — per-gene asymmetric SMEM, any b_max
  4. scatter_add   — pure PyTorch fallback (~3.7 M pairs/s)
"""


import time
import warnings
from typing import Optional
from typing import Union

import numpy as np
import torch

from tenex.kernels import compute_smem_bytes
from tenex.kernels import get_kernel
from tenex.kernels import query_smem_optin
from tenex.preprocess import discretize
from tenex.binarize import binarize as _binarize_fn
from tenex.result import TransferEntropyResult
from tenex.utils import get_device_list
from tenex._log import set_verbose
from tenex._log import vprint


class TransferEntropyEngine:
    """
    High-performance Transfer Entropy computation engine.

    Computes pairwise TE for all variable pairs in a multivariate time series.
    Kernel selection is automatic based on data characteristics and hardware.

    Parameters
    ----------
    data           : (n_vars, T) float32 ndarray — multivariate time series,
                     ordered along the time axis (column index = time step).
    variable_names : (n_vars,) str ndarray — name for each variable.
    sources        : (optional) str ndarray — subset of variable_names to
                     use as TE sources. If None, all pairs are computed.
    """

    def __init__(
        self,
        data: np.ndarray,
        variable_names: np.ndarray,
        sources: Optional[np.ndarray] = None,
    ):
        if data is None or variable_names is None:
            raise ValueError(
                "data and variable_names are required.\n"
                "For scRNA-seq data, use tenex.io.load_scrna() to construct "
                "a TransferEntropyEngine from expression / pseudotime / branch."
            )
        self._data = np.asarray(data, dtype=np.float32)
        self._variable_names = np.asarray(variable_names)
        if self._data.ndim != 2 or self._data.size == 0:
            raise ValueError(
                f"data must be a non-empty 2-D array of shape (n_vars, T); "
                f"got shape {self._data.shape}"
            )
        if not np.isfinite(self._data).all():
            raise ValueError("data contains NaN or infinite values")
        if self._variable_names.ndim != 1:
            raise ValueError(
                f"variable_names must be 1-D; got shape {self._variable_names.shape}"
            )
        n_vars = self._data.shape[0]
        if self._variable_names.shape[0] != n_vars:
            raise ValueError(
                f"variable_names length ({self._variable_names.shape[0]}) must match "
                f"the number of variables ({n_vars})"
            )
        if len(np.unique(self._variable_names)) != n_vars:
            raise ValueError("variable_names must be unique")
        if sources is not None:
            _src = np.atleast_1d(np.asarray(sources))
            if _src.ndim != 1:
                raise ValueError(f"sources must be 1-D; got shape {_src.shape}")
            _missing = np.setdiff1d(_src, self._variable_names)
            if _missing.size > 0:
                raise ValueError(
                    f"source names not found in variable_names: {list(_missing)}"
                )
            if len(np.unique(_src)) != _src.shape[0]:
                raise ValueError("sources must not contain duplicate names")
            # Copy so later mutation of the caller's array cannot change which
            # rows downstream inference treats as computed.
            self._sources = _src.copy()
        else:
            self._sources = None
        self._result_matrix = None
        self._bin_arrs_cache: dict = {}
        self._pairs_cache: dict = {}
        self._last_run_info: Optional[dict] = None

    @property
    def variable_names(self) -> np.ndarray:
        """Variable names array, shape (n_vars,)."""
        return self._variable_names

    # ── Cache management ───────────────────────────────────────────────────────

    def clear_cache(self):
        """Free all cached GPU tensors and pair arrays."""
        self._bin_arrs_cache.clear()
        self._pairs_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Result I/O ────────────────────────────────────────────────────────────

    def save_result_matrix(self, spath: str):
        """Save the result matrix as a tab-separated file."""
        if self._result_matrix is None:
            raise ValueError("compute() has not been called yet")

        tmp = np.concatenate(
            [self._variable_names[:, None], self._result_matrix.astype(str)], axis=1
        )
        header = np.concatenate((['TE'], self._variable_names))
        tmp = np.concatenate([header[None, :], tmp])
        np.savetxt(spath, tmp, delimiter='\t', fmt='%s')
        vprint(f"Saved result matrix -> {spath}")

    # ── Device resolution (PyTorch Lightning–style) ─────────────────────────

    @staticmethod
    def _resolve_accelerator_and_devices(
        accelerator: str = "auto",
        devices: Union[int, list, str, None] = "auto",
    ) -> list[int]:
        """Resolve ``accelerator`` + ``devices`` to a list of CUDA device indices.

        An empty list means CPU.

        Parameters
        ----------
        accelerator : "auto" | "cpu" | "gpu"
            "auto" — GPU if available, else CPU.
            "cpu"  — force CPU (``devices`` is ignored).
            "gpu"  — require CUDA; raises if unavailable.
        devices : "auto" | int | list[int] | None
            "auto"/None — all available GPUs (when accelerator="gpu"/"auto").
            int > 0     — use the first *N* GPUs.
            -1          — all available GPUs.
            list[int]   — specific GPU indices, e.g. [0, 2].
        """
        accel = accelerator.lower()

        # ── CPU path ─────────────────────────────────────────────────────
        if accel == "cpu":
            return []

        # ── Auto path ────────────────────────────────────────────────────
        if accel == "auto":
            if not torch.cuda.is_available():
                return []
            # fall through to GPU logic with devices
        elif accel == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "accelerator='gpu' but no CUDA device is available"
                )
        else:
            raise ValueError(
                f"Unknown accelerator '{accelerator}'. "
                f"Choose from: 'auto', 'cpu', 'gpu'"
            )

        # ── GPU path: resolve devices ────────────────────────────────────
        all_gpus = get_device_list()

        if devices is None or (isinstance(devices, str) and devices == "auto"):
            return all_gpus

        if isinstance(devices, int):
            if devices == -1:
                return all_gpus
            if devices <= 0:
                raise ValueError(
                    f"devices as int must be >= 1 or -1 (got {devices})"
                )
            if devices > len(all_gpus):
                raise ValueError(
                    f"Requested {devices} GPUs but only "
                    f"{len(all_gpus)} available"
                )
            return all_gpus[:devices]

        if isinstance(devices, (list, tuple)):
            for d in devices:
                if d not in all_gpus:
                    raise ValueError(
                        f"GPU {d} not available. "
                        f"Available: {all_gpus}"
                    )
            return list(devices)

        raise TypeError(
            f"devices must be 'auto', int, list[int], or None "
            f"(got {type(devices).__name__})"
        )

    @staticmethod
    def _primary_device(device_ids: list[int]) -> torch.device:
        if not device_ids:
            return torch.device('cpu')
        return torch.device(f'cuda:{device_ids[0]}')

    # ── Data access ────────────────────────────────────────────────────────

    def _get_aligned_data(self) -> np.ndarray:
        return self._data

    # ── Discretization (cached per-device) ──────────────────────────────────

    def _discretize(self, arr, binning_method, kp, device, **kwargs):
        # use_numpy_bins is part of the cache key: numpy (FastTENET-exact) and
        # GPU paths produce different float32 std values, so a single key would
        # let one path's bins poison the other.
        use_numpy = kwargs.get('use_numpy_bins', True)
        base_key = (binning_method.upper(), kp, bool(use_numpy))
        dev_key = (*base_key, str(device))

        # Fast path: already cached for this exact device
        if dev_key in self._bin_arrs_cache:
            return self._bin_arrs_cache[dev_key]

        # Check CPU cache (avoids re-running discretize)
        cpu_key = (*base_key, 'cpu')
        if cpu_key in self._bin_arrs_cache:
            bin_cpu, n_bins_cpu, b_max, npg_cpu = self._bin_arrs_cache[cpu_key]
            if str(device) == 'cpu':
                return bin_cpu, n_bins_cpu, b_max, npg_cpu
            bin_arrs = bin_cpu.to(device)
            n_bins = n_bins_cpu.to(device) if isinstance(n_bins_cpu, torch.Tensor) else n_bins_cpu
            n_per_var = npg_cpu.to(device)
        else:
            # use_numpy_bins controls whether binning runs on CPU (numpy) or GPU.
            # numpy: exact match with original FastTENET, safe.
            # GPU: ~5x faster for large datasets, corr ~1.0 vs numpy.
            # Default: follow compute()'s use_numpy_bins parameter.
            # Note: OOM here is a bug, not a fallback condition — discretize()
            # is responsible for adaptive chunking that fits available VRAM.
            bin_arrs, n_bins, b_max, n_per_var = discretize(
                arr, method=binning_method, kp=kp, device=device,
                use_numpy_bins=use_numpy,
            )
            # Cache CPU copy for potential multi-GPU use
            self._bin_arrs_cache[cpu_key] = (
                bin_arrs.cpu() if bin_arrs.is_cuda else bin_arrs.clone(),
                n_bins.cpu() if isinstance(n_bins, torch.Tensor) and n_bins.is_cuda
                else (n_bins.clone() if isinstance(n_bins, torch.Tensor) else n_bins),
                b_max,
                n_per_var.cpu() if n_per_var.is_cuda else n_per_var.clone(),
            )

        # Cache device-specific tensors (avoid re-transfer on next compute())
        self._bin_arrs_cache[dev_key] = (bin_arrs, n_bins, b_max, n_per_var)
        return bin_arrs, n_bins, b_max, n_per_var

    def _binarize(self, arr, method, device, use_numpy=True, backend=None,
                  params=None):
        """Binarize the aligned data to {0, 1} (b_max=2), cached per device.

        Returns the same 4-tuple as :meth:`_discretize` so the binarization and
        multi-bin paths are interchangeable downstream. ``backend`` selects the
        binarize backend ('numpy', 'torch', 'cupy'); None follows ``use_numpy``.
        ``params`` holds method-specific options (e.g. ``{'q': 0.9}``).
        """
        params = params or {}
        param_key = tuple(sorted(params.items()))
        base_key = ('BIN:' + method.upper(), bool(use_numpy), backend, param_key)
        dev_key = (*base_key, str(device))
        if dev_key in self._bin_arrs_cache:
            return self._bin_arrs_cache[dev_key]

        cpu_key = (*base_key, 'cpu')
        if cpu_key in self._bin_arrs_cache and str(device) != 'cpu':
            bin_cpu, n_bins_cpu, b_max, npg_cpu = self._bin_arrs_cache[cpu_key]
            bin_arrs = bin_cpu.to(device)
            n_bins = n_bins_cpu.to(device)
            n_per_var = npg_cpu.to(device)
        else:
            bin_arrs, n_bins, b_max, n_per_var = _binarize_fn(
                arr, method=method, device=device, use_numpy=use_numpy,
                backend=backend, **params,
            )
            self._bin_arrs_cache[cpu_key] = (
                bin_arrs.cpu() if bin_arrs.is_cuda else bin_arrs.clone(),
                n_bins.cpu() if n_bins.is_cuda else n_bins.clone(),
                b_max,
                n_per_var.cpu() if n_per_var.is_cuda else n_per_var.clone(),
            )

        self._bin_arrs_cache[dev_key] = (bin_arrs, n_bins, b_max, n_per_var)
        return bin_arrs, n_bins, b_max, n_per_var

    # ── Auto-scaling ──────────────────────────────────────────────────────────

    @staticmethod
    def _auto_scale_gpus(device_ids, kernel_obj, n_vars, b_max, T, tau):
        """Reduce GPU count if multi-GPU overhead exceeds compute benefit.

        Uses a bandwidth-based overhead model measured on PCIe Gen5 x16:
        the dominant cost of adding GPUs is the host-to-device (H2D) transfer
        of bin_arrs via pinned memory.  With TENEX's pipelined dispatch,
        GPU 0 computes immediately while non-primary GPUs receive data in
        parallel, so the overhead equals a single H2D transfer time:

            overhead ≈ data_bytes / bandwidth

        Measured bandwidth: ~57 GB/s on RTX PRO 6000 Blackwell (PCIe Gen5).
        A conservative 50 GB/s is used to account for system variance.

        Returns a (possibly shorter) device_ids list.
        """
        if len(device_ids) <= 1:
            return device_ids

        n_pairs = n_vars * (n_vars - 1)

        # Conservative single-GPU pair throughput estimates (pairs/sec)
        _rates = {
            "GEMM-B2":       500_000_000,
            "Full-SMEM":      40_000_000,
            "Triton":          4_500_000,
            "Adaptive-SMEM":   5_000_000,
            "scatter_add":     3_000_000,
        }
        rate = _rates.get(kernel_obj.name, 3_000_000)
        est_compute_sec = n_pairs / rate

        # Bandwidth-based H2D overhead (measured: ~57 GB/s, conservative: 50 GB/s)
        data_bytes = n_vars * T * 4  # bin_arrs: int32
        h2d_bandwidth = 50e9  # bytes/sec (conservative PCIe Gen5 estimate)
        overhead_sec = data_bytes / h2d_bandwidth + 0.001  # +1 ms for thread + sync

        n_gpus = len(device_ids)
        compute_saving = est_compute_sec * (1 - 1.0 / n_gpus)

        if compute_saving < overhead_sec:
            return device_ids[:1]
        return device_ids

    # ── Pipelined multi-GPU pair-free ─────────────────────────────────────────

    @staticmethod
    def _compute_pairfree_pipelined(kernel_obj, bin_arrs, n_per_var,
                                     n_vars, b_max, tau, device_ids):
        """Pipeline: GPU 0 computes immediately while others receive data.

        bin_arrs and n_per_var reside on device_ids[0].  GPU 0 starts
        computing its pair range with zero transfer overhead.  Other GPUs
        receive data via CPU-staged pinned-memory transfer (D2D is broken
        on some multi-GPU systems), then compute their own pair ranges.

        Results are written directly into a pre-allocated output array
        to avoid costly np.concatenate on large result vectors.
        """
        import math
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_gpus = len(device_ids)
        n_pairs = n_vars * (n_vars - 1)
        chunk = math.ceil(n_pairs / n_gpus)

        # Pre-allocate output: each GPU writes to its own slice
        output = np.empty(n_pairs, dtype=np.float32)

        # Stage data to pinned CPU memory once (for non-primary GPUs)
        from tenex.kernels import _try_pin

        bin_cpu = _try_pin(bin_arrs.cpu())
        npg_cpu = (_try_pin(n_per_var.cpu())
                   if isinstance(n_per_var, torch.Tensor) else n_per_var)

        def _run_chunked_pairfree(bin_d, npg_d, dev, offset, n_local, out_slice):
            """Run pair-free with per-GPU VRAM-aware chunking, write to out_slice."""
            output_bytes = n_local * 4
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info(dev)
            reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
            available = max(0, free_mem - reserve)

            if output_bytes <= available:
                ents = kernel_obj.compute_pairfree(
                    bin_d, n_vars, b_max, npg_d,
                    tau, dev, pair_offset=offset, n_pairs_local=n_local,
                )
                out_slice[:] = ents
                return
            # Chunk within this GPU
            max_pp = max(1024, available // 4)
            cur = offset
            written = 0
            remaining = n_local
            while remaining > 0:
                sz = min(max_pp, remaining)
                ent = kernel_obj.compute_pairfree(
                    bin_d, n_vars, b_max, npg_d,
                    tau, dev, pair_offset=cur, n_pairs_local=sz,
                )
                out_slice[written:written + sz] = ent
                cur += sz
                written += sz
                remaining -= sz

        def _run(rank):
            dev_id = device_ids[rank]
            torch.cuda.set_device(dev_id)
            dev = torch.device(f'cuda:{dev_id}')
            offset = rank * chunk
            n_local = min(chunk, n_pairs - offset)
            if n_local <= 0:
                return

            out_slice = output[offset:offset + n_local]

            if rank == 0:
                # GPU 0: data already present, compute immediately
                _run_chunked_pairfree(
                    bin_arrs, n_per_var, dev, offset, n_local, out_slice,
                )
            else:
                # Other GPUs: CPU-staged transfer (D2D broken on some systems)
                bin_d = bin_cpu.to(dev, non_blocking=True)
                npg_d = (npg_cpu.to(dev, non_blocking=True)
                         if isinstance(npg_cpu, torch.Tensor) else npg_cpu)
                torch.cuda.synchronize(dev)
                _run_chunked_pairfree(
                    bin_d, npg_d, dev, offset, n_local, out_slice,
                )

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = [pool.submit(_run, i) for i in range(n_gpus)]
            for fut in futures:
                fut.result()

        return output

    # ── Pair construction (cached) ────────────────────────────────────────────

    def _build_pairs(self, n_vars: int) -> np.ndarray:
        src_id = hash(tuple(self._sources)) if self._sources is not None else None
        cache_key = (n_vars, src_id)
        if cache_key in self._pairs_cache:
            return self._pairs_cache[cache_key]

        if self._sources is not None:
            _, inds_source, _ = np.intersect1d(
                self._variable_names, self._sources, return_indices=True
            )
            if len(inds_source) == 0:
                raise ValueError(
                    "No source names found in variable_names. "
                    "Check that sources and variable_names share common entries."
                )
            srcs = inds_source.astype(np.int64)
            pairs_list = []
            for t in range(n_vars):
                s_valid = srcs[srcs != t]
                if len(s_valid) > 0:
                    block = np.empty((len(s_valid), 2), dtype=np.int64)
                    block[:, 0] = t
                    block[:, 1] = s_valid
                    pairs_list.append(block)
            if not pairs_list:
                pairs_np = np.empty((0, 2), dtype=np.int64)
            else:
                pairs_np = np.concatenate(pairs_list)
        else:
            n_pairs = n_vars * (n_vars - 1)
            pairs_np = np.empty((n_pairs, 2), dtype=np.int64)
            idx = 0
            for t in range(n_vars):
                pairs_np[idx:idx + n_vars - 1, 0] = t
                pairs_np[idx:idx + t, 1] = np.arange(t, dtype=np.int64)
                pairs_np[idx + t:idx + n_vars - 1, 1] = np.arange(t + 1, n_vars, dtype=np.int64)
                idx += n_vars - 1

        # Sort by target gene for L2 cache reuse
        _sort_idx = np.argsort(pairs_np[:, 0], kind='stable')
        pairs_np = pairs_np[_sort_idx]
        self._pairs_cache[cache_key] = pairs_np
        return pairs_np

    # ── Profiling helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _collect_timings(events, t0, mode):
        """Collect phase timings from CUDA events. Cost: one synchronize (already done)."""
        # Events layout (pair-free / pair-based):
        #   0: start              1: after data_align     2: after discretize
        #   3: after kernel_select  4: after prepare      5: before TE kernel
        #   6: after TE kernel    7: after assemble
        # Events layout (matrix):
        #   0-3: same              4: before matrix kernel  5: after matrix kernel
        def ms(a, b):
            try:
                return events[a].elapsed_time(events[b]) / 1000.0
            except (RuntimeError, ValueError, IndexError):
                # torch >= 2.10 raises ValueError ("Both events must be
                # recorded ...") for an unrecorded event; older raised
                # RuntimeError. Treat a missing phase as 0 s either way.
                return 0.0

        timings = {
            'total': time.time() - t0,
            'data_align': ms(0, 1),
            'discretize': ms(1, 2),
            'kernel_select': ms(2, 3),
        }
        if mode == 'matrix':
            timings['prepare'] = ms(3, 4)
            timings['compute'] = ms(4, 5)
        else:
            timings['prepare'] = ms(3, 4)
            timings['compute'] = ms(5, 6)
            timings['assemble'] = ms(6, 7)
        return timings

    # ── Main entry point ──────────────────────────────────────────────────────

    def compute(
        self,
        # device / parallelism (PyTorch Lightning–style)
        accelerator: str = "auto",
        devices: Union[int, list, str, None] = "auto",
        # discretisation
        binning_method: str = 'FSBW-L',
        binarization: Optional[str] = None,
        binarization_backend: Optional[str] = None,
        binarization_kwargs: Optional[dict] = None,
        kp: float = 0.5,
        tau: int = 1,
        # batch control
        batch_size: Optional[int] = None,
        autotune: bool = False,
        # kernel selection (auto by default)
        kernel: Optional[str] = None,
        # kernel options
        coarsening: Optional[bool] = None,
        # binning options
        use_numpy_bins: Optional[bool] = None,
        # profiling
        profile: bool = False,
        # logging
        verbose: bool = False,
    ) -> TransferEntropyResult:
        """
        Compute pairwise Transfer Entropy.

        Parameters
        ----------
        accelerator     : 'auto' (default), 'cpu', or 'gpu'.
                          'auto' uses GPU if available, else CPU.
        devices         : 'auto' (all GPUs), int (GPU count), list[int]
                          (specific GPUs), or -1 (all GPUs).
                          Ignored when accelerator='cpu'.
        binning_method  : 'FSBW-L' (default), 'FSBW', 'FSBW-R', 'FSBW-B'
        binarization    : None (default) or a binarization method name (e.g.
                          'nonzero'). When set, the data is binarized to
                          {0, 1} (b_max=2) instead of multi-bin discretized,
                          which routes the compute to the GEMM-B2 kernel.
                          See tenex.binarize.available_binarizers().
        binarization_backend : None (follow use_numpy_bins) or 'numpy' /
                          'torch' / 'cupy'. Selects the backend that computes
                          the binarization. 'cupy' needs the optional cupy
                          package and a CUDA device.
        binarization_kwargs : optional dict of method-specific parameters for
                          the binarization (e.g. {'q': 0.9} for 'quantile').
        kp              : bin-width fraction (0.5 recommended)
        tau              : time-lag for TE (default 1)
        batch_size      : pairs per GPU batch; None = auto-selected via
                          L2-based heuristic (zero cost, deterministic).
        autotune        : if True and batch_size is None, determine batch
                          size by runtime calibration (~1-3s, cached).
                          Ignored when batch_size is explicitly set.
        kernel          : kernel name or None for auto-selection.
                          Valid: "GEMM-B2", "Full-SMEM",
                          "Adaptive-SMEM", "scatter_add"
        coarsening      : True/False/None — bin coarsening for Adaptive-SMEM
        use_numpy_bins  : None (default) — follow the device: bin on GPU when
                          accelerator resolves to CUDA (~5x faster discretize
                          on large data, corr~1.0 vs numpy but not bit-for-bit),
                          and on CPU/numpy (FastTENET-exact) otherwise.
                          True/False overrides this device-based choice.
        verbose         : print [TENEX] status messages during the run
                          (default False).

        Returns
        -------
        result : TransferEntropyResult
                 Wraps the (n_vars, n_vars) float32 TE matrix
                 (``result.matrix[i, j] = TE(var_i -> var_j)``) together with
                 the variable names, bin arrays, and kernel metadata.
        """
        set_verbose(verbose)
        t0 = time.time()

        # ── Profiling setup ───────────────────────────────────────────────────
        _timings = {}  # phase -> seconds (populated only when profile=True)

        # ── 1. Resolve devices (Lightning-style) ─────────────────────────────
        device_ids = self._resolve_accelerator_and_devices(accelerator, devices)
        primary_device = self._primary_device(device_ids)
        dev_str = f"gpu:{device_ids}" if device_ids else "cpu"
        vprint(f"[TENEX] accelerator: {dev_str}")

        on_cuda_early = primary_device.type == 'cuda'

        # Device-based binning default: GPU device -> GPU binning (fast),
        # CPU device -> numpy binning (FastTENET-exact). Explicit True/False
        # passed by the caller overrides this.
        if use_numpy_bins is None:
            use_numpy_bins = not on_cuda_early

        if profile and on_cuda_early:
            _ev = [torch.cuda.Event(enable_timing=True) for _ in range(8)]
            _ev[0].record(torch.cuda.current_stream(primary_device))

        # ── 2. Data alignment ─────────────────────────────────────────────────
        arr = self._get_aligned_data()
        n_vars, T = arr.shape
        if isinstance(tau, bool) or not isinstance(tau, (int, np.integer)) or tau < 1:
            raise ValueError(f"tau must be an integer >= 1; got {tau!r}")
        if T <= tau:
            raise ValueError(
                f"Time series length T={T} must be greater than tau={tau}"
            )
        if n_vars < 2:
            # Fewer than two variables: no directed pairs. Return early before any
            # kernel dispatch so no code path launches a zero-block grid.
            self._result_matrix = np.zeros((n_vars, n_vars), dtype=np.float32)
            self._last_run_info = {}
            self._last_bin_arrs = None
            self._last_n_per_var = None
            vprint("[TENEX] fewer than two variables; returning zero matrix")
            return self._wrap_result()
        vprint(f"[TENEX] data: {n_vars} variables x {T} time-points")
        if profile and on_cuda_early:
            _ev[1].record(torch.cuda.current_stream(primary_device))

        # ── 3. Discretization (or binarization) ───────────────────────────────
        if binarization is not None:
            bin_arrs, n_bins, b_max, n_per_var = self._binarize(
                arr, binarization, primary_device,
                use_numpy=use_numpy_bins, backend=binarization_backend,
                params=binarization_kwargs,
            )
            be = binarization_backend or ('numpy' if use_numpy_bins
                                          or not on_cuda_early else 'torch')
            disc_label = f"binarization={binarization} (backend={be})"
        else:
            bin_arrs, n_bins, b_max, n_per_var = self._discretize(
                arr, binning_method, kp, primary_device,
                use_numpy_bins=use_numpy_bins,
            )
            disc_label = f"binning={binning_method}, kp={kp}"
        K = bin_arrs.shape[2]
        # Histogram counts are accumulated in int32 (SMEM/scatter kernels); the
        # binary GEMM-B2 path accumulates counts in float32. Guard the sample
        # count N = l_eff * K against silent counter overflow / loss of unit
        # resolution before dispatching a kernel.
        _n_samples = (bin_arrs.shape[1] - tau) * K
        if _n_samples >= 2**31:
            raise ValueError(
                f"sample count l_eff*K={_n_samples} exceeds the int32 histogram "
                f"limit (2^31); counts would overflow"
            )
        on_cuda = primary_device.type == 'cuda'
        if on_cuda:
            torch.cuda.synchronize(primary_device)
        vprint(f"[TENEX] {disc_label}, K={K}, b_max={b_max}")
        if profile and on_cuda:
            _ev[2].record(torch.cuda.current_stream(primary_device))

        # ── 5. Kernel selection via registry ──────────────────────────────────
        b_max_before_coarsen = b_max
        smem_bytes = compute_smem_bytes(b_max, bin_arrs.shape[1], K, tau)
        # Use the minimum SMEM opt-in across selected GPUs so heterogeneous
        # multi-GPU runs plan for the weakest device.
        smem_optin = query_smem_optin(device_ids) if on_cuda else 0
        source_filter = self._sources is not None

        kernel_obj = self._select_kernel(
            kernel, b_max, on_cuda, smem_optin, smem_bytes,
            n_per_var, source_filter,
        )

        # GEMM-B2 accumulates the 3-D joint counts in float32. Above 2^24 samples
        # the integer counts can no longer be represented exactly. Emit a real
        # warning (not verbose-gated) so the precision loss is visible under the
        # default settings, and only for the kernel that is actually affected.
        if kernel_obj.name == "GEMM-B2" and _n_samples > 2**24:
            warnings.warn(
                f"GEMM-B2 accumulates joint counts in float32 and the sample "
                f"count l_eff*K={_n_samples} exceeds 2^24, so counts may lose "
                f"unit resolution. Use a multi-bin (SMEM) kernel for exact counts.",
                RuntimeWarning, stacklevel=2,
            )

        # Build run info dict (updated as compute progresses)
        _run_info = {
            'n_vars': n_vars, 'T': T, 'K': K, 'tau': tau,
            'n_pairs': n_vars * (n_vars - 1),
            'device_ids': list(device_ids),
            'n_gpus': len(device_ids),
            'binning_method': binning_method if binarization is None else binarization,
            'binarization': binarization,
            'kp': kp,
            'use_numpy_bins': use_numpy_bins,
            'b_max_before_coarsen': b_max_before_coarsen,
            'b_max': b_max,
            'kernel': kernel_obj.name,
            'smem_optin': smem_optin,
            'smem_bytes': smem_bytes,
            'source_filter': source_filter,
            'bin_dtype': str(bin_arrs.dtype),
            'bin_bytes': bin_arrs.nelement() * bin_arrs.element_size(),
        }

        # ── 6. Auto-scale: reduce GPUs if compute is too fast ────────────────
        if len(device_ids) > 1:
            scaled_ids = self._auto_scale_gpus(
                device_ids, kernel_obj, n_vars, b_max,
                bin_arrs.shape[1], tau,
            )
            if len(scaled_ids) < len(device_ids):
                released = [d for d in device_ids if d not in scaled_ids]
                for dev_id in released:
                    torch.cuda.set_device(dev_id)
                    torch.cuda.empty_cache()
                # Restore current device to primary after releasing
                torch.cuda.set_device(scaled_ids[0])
                vprint(f"[TENEX] auto-scale: {len(scaled_ids)}/{len(device_ids)} GPUs "
                      f"(compute too fast for multi-GPU overhead)")
                device_ids = scaled_ids
                primary_device = self._primary_device(device_ids)

        if profile and on_cuda:
            _ev[3].record(torch.cuda.current_stream(primary_device))

        # ── 7. Matrix kernel path (GEMM-B2) ──────────────────────────────────
        if kernel_obj.is_matrix_kernel:
            _run_info.update(mode='matrix', pair_free=False, coarsened=False)
            self._last_run_info = _run_info
            self._last_bin_arrs = bin_arrs
            self._last_n_per_var = n_per_var
            if profile and on_cuda:
                _ev[4].record(torch.cuda.current_stream(primary_device))
            result = self._run_matrix_kernel(
                kernel_obj, bin_arrs, n_vars, tau, device_ids, t0,
            )
            if profile and on_cuda:
                _ev[5].record(torch.cuda.current_stream(primary_device))
                torch.cuda.synchronize(primary_device)
                _timings.update(self._collect_timings(_ev, t0, 'matrix'))
                self._last_run_info['timings'] = _timings
            return result

        # ── 8. Memory check: can we clone + prepare on GPU? ─────────────────
        #   prepare() clones bin_arrs. coarsen_bins() uses a zero-memory
        #   CUDA kernel (in-place, no temporaries). However, the pair-free
        #   TE kernel has a known issue with CUDA async errors when both
        #   bin_arrs and large output tensors coexist on GPU after
        #   in-process discretize. Offload prepare to CPU when memory is
        #   tight to ensure the fast pair-free path has maximum VRAM headroom.
        if on_cuda:
            bin_bytes = bin_arrs.element_size() * bin_arrs.nelement()
            free_mem, _ = torch.cuda.mem_get_info(primary_device)
            n_pairs = n_vars * (n_vars - 1)
            output_bytes = n_pairs * 4  # float32
            # Need: bin_arrs + output + 15% reserve for allocator
            total_needed = bin_bytes + output_bytes
            reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
            needs_offload = (total_needed > free_mem - reserve)
        else:
            needs_offload = False

        if needs_offload:
            # ── 8a. Offloaded prepare: coarsening on CPU to avoid peak VRAM ──
            vprint(f"[TENEX] bin_arrs={bin_bytes / 2**30:.2f} GB, "
                  f"free={free_mem / 2**30:.2f} GB → offloading prepare to CPU")
            bin_arrs_cpu = bin_arrs.cpu()
            npg_cpu = n_per_var.cpu()
            del bin_arrs, n_per_var
            torch.cuda.empty_cache()

            npg_pre = npg_cpu.clone()  # per-var bin counts before coarsening
            bin_arrs_cpu, npg_cpu, b_max = kernel_obj.prepare(
                bin_arrs_cpu.clone(), npg_cpu.clone(), b_max,
                use_coarsening=coarsening, on_cuda=on_cuda,
                n_vars=n_vars, tau=tau, smem_optin=smem_optin,
            )
            vprint(f"[TENEX] kernel: {kernel_obj.log_description(b_max=b_max)}")

            # After prepare, try to move back to GPU for fast pair-free path.
            # prepared bin_arrs is same size or smaller (coarsening is in-place).
            prepared_bytes = (bin_arrs_cpu.nelement()
                              * bin_arrs_cpu.element_size())
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info(primary_device)
            # Need: bin_arrs + n_per_var + output chunk (at least 1 chunk)
            min_output_chunk = min(n_vars * (n_vars - 1), 1024 * 1024) * 4
            fits_on_gpu = (prepared_bytes + min_output_chunk
                           < free_after - 128 * 1024 * 1024)

            if fits_on_gpu:
                # Move back to GPU and use normal pair-free path
                bin_arrs = bin_arrs_cpu.to(primary_device)
                n_per_var = npg_cpu.to(primary_device)
                del bin_arrs_cpu, npg_cpu
                vprint(f"[TENEX] prepared bin_arrs fits on GPU "
                      f"({prepared_bytes / 2**30:.2f} GB < "
                      f"{free_after / 2**30:.2f} GB free)")
                # Fall through to normal pair-free/pair-based path below
            else:
                # Truly doesn't fit: use gene-blocked compute
                vprint(f"[TENEX] prepared bin_arrs too large for GPU "
                      f"({prepared_bytes / 2**30:.2f} GB > "
                      f"{free_after / 2**30:.2f} GB free), "
                      f"using gene-blocked")
                _coarsened_gb = (b_max != _run_info['b_max_before_coarsen'])
                _run_info.update(
                    mode='gene-blocked', pair_free=True, b_max=b_max,
                    coarsened=_coarsened_gb,
                    # Count only variables whose bin count actually changed
                    # during coarsening (pre/post comparison), not every
                    # variable that happens to sit at the final b_max.
                    n_coarsened_vars=int((npg_cpu != npg_pre).sum().item()),
                )
                self._last_run_info = _run_info
                self._last_bin_arrs = bin_arrs_cpu
                self._last_n_per_var = npg_cpu
                self._compute_gene_blocked(
                    kernel_obj, bin_arrs_cpu, npg_cpu,
                    n_vars, b_max, tau, batch_size,
                    device_ids, t0,
                )
                return self._wrap_result()

        # ── 8b. Normal path: clone + prepare on GPU ──────────────────────────
        ba_clone = bin_arrs.clone()
        npg_clone = n_per_var.clone()
        # Free original tensors before prepare to maximize VRAM
        del bin_arrs, n_per_var
        if on_cuda:
            torch.cuda.empty_cache()
        bin_arrs, n_per_var, b_max = kernel_obj.prepare(
            ba_clone, npg_clone, b_max,
            use_coarsening=coarsening, on_cuda=on_cuda,
            n_vars=n_vars, tau=tau, smem_optin=smem_optin,
        )

        # Store bin_arrs and n_per_var for TransferEntropyResult
        self._last_bin_arrs = bin_arrs
        self._last_n_per_var = n_per_var

        # ── 9. Update run_info after prepare ─────────────────────────────────
        _coarsened = (b_max != _run_info['b_max_before_coarsen'])
        _n_coarsened = 0
        if _coarsened:
            _n_coarsened = int((n_per_var == b_max).sum().item()) if _coarsened else 0
        _run_info.update(
            b_max=b_max, coarsened=_coarsened,
            n_coarsened_vars=_n_coarsened,
            bin_dtype=str(bin_arrs.dtype),
            bin_bytes=bin_arrs.nelement() * bin_arrs.element_size(),
        )

        # ── 10. Pair-free fast path ───────────────────────────────────────────
        use_pairfree = (kernel_obj.supports_pairfree and not source_filter)
        # Adaptive-SMEM pair-free can't fall back to gmem per-pair: when
        # coarsening is disabled and the worst-case SMEM still exceeds the
        # budget, the pair-free kernel raises.  In that case downgrade to
        # the pair-based path which has a per-pair gmem fallback.
        if (use_pairfree and on_cuda
                and kernel_obj.name == "Adaptive-SMEM" and coarsening is False):
            _post_smem = compute_smem_bytes(b_max, bin_arrs.shape[1], K, tau)
            if _post_smem > smem_optin:
                use_pairfree = False
                vprint(
                    f"[TENEX] coarsening disabled and smem_worst={_post_smem} "
                    f"> smem_optin={smem_optin}: using pair-based path so "
                    f"per-pair gmem fallback can handle overflow"
                )
        n_pairs = n_vars * (n_vars - 1)

        if profile and on_cuda:
            _ev[4].record(torch.cuda.current_stream(primary_device))

        if use_pairfree:
            _run_info['mode'] = 'pair-free'
            _run_info['pair_free'] = True
            vprint(f"[TENEX] pairs: {n_pairs:,} (pair-free)")
            vprint(f"[TENEX] kernel: {kernel_obj.log_description(b_max=b_max)}")

            if profile and on_cuda:
                _ev[5].record(torch.cuda.current_stream(primary_device))

            if len(device_ids) <= 1:
                entropies = self._compute_pairfree_chunked(
                    kernel_obj, bin_arrs, n_per_var,
                    n_vars, b_max, tau, primary_device,
                )
            else:
                entropies = self._compute_pairfree_pipelined(
                    kernel_obj, bin_arrs, n_per_var,
                    n_vars, b_max, tau, device_ids,
                )

            if profile and on_cuda:
                _ev[6].record(torch.cuda.current_stream(primary_device))

            self._last_run_info = _run_info
            result = self._assemble_result_pairfree(entropies, n_vars, t0)

            if profile and on_cuda:
                _ev[7].record(torch.cuda.current_stream(primary_device))
                torch.cuda.synchronize(primary_device)
                _timings.update(self._collect_timings(_ev, t0, 'pair-free'))
                self._last_run_info['timings'] = _timings
            return result

        # ── 11. Standard pair-based path ─────────────────────────────────────
        _run_info['mode'] = 'pair-based'
        _run_info['pair_free'] = False
        pairs_np = self._build_pairs(n_vars)
        n_pairs = len(pairs_np)
        vprint(f"[TENEX] pairs: {n_pairs:,}")
        vprint(f"[TENEX] kernel: {kernel_obj.log_description(b_max=b_max)}")

        # Select batch_size if not explicitly specified
        if batch_size is None and on_cuda:
            if autotune:
                batch_size = kernel_obj.autotune_batch_size(
                    bin_arrs, b_max, n_per_var, tau, primary_device,
                )
            else:
                batch_size = kernel_obj.heuristic_batch_size(
                    bin_arrs, b_max, tau,
                )
            peak_pp = kernel_obj.peak_bytes_per_pair(b_max, bin_arrs.shape[1],
                                                      bin_arrs.shape[2], tau)
            vprint(f"[TENEX] batch_size={batch_size:,} "
                  f"({batch_size * peak_pp / 2**20:.0f} MB working set, "
                  f"{peak_pp} B/pair)")

        if n_pairs == 0:
            # Edge case: TF filter produced no valid pairs
            self._last_run_info = _run_info
            return self._assemble_result(
                np.empty(0, dtype=np.float32), pairs_np, n_vars, t0,
            )

        if len(device_ids) <= 1:
            # Transfer pairs in chunks to avoid OOM on small-VRAM GPUs.
            # Reserve memory for the kernel's per-batch peak allocation:
            # the pairs tensor AND the kernel batch internals must both fit.
            pairs_bytes = pairs_np.nbytes  # n_pairs × 2 × 8 bytes
            if on_cuda:
                torch.cuda.empty_cache()
                free_mem, _ = torch.cuda.mem_get_info(primary_device)
                kernel_reserve = kernel_obj.peak_batch_bytes(
                    b_max, bin_arrs.shape[1], bin_arrs.shape[2],
                    batch_size or 4096, tau,
                )
                # Additional reserve for allocator fragmentation (important on small-VRAM GPUs)
                frag_reserve = max(64 * 1024 * 1024, int(free_mem * 0.10))
                pairs_budget = max(0, free_mem - kernel_reserve - frag_reserve)
            else:
                # CPU path: no VRAM constraint; run in one shot.
                free_mem = 0
                pairs_budget = pairs_bytes
            if pairs_bytes <= pairs_budget:
                # Fits in one shot
                pairs_t = torch.from_numpy(pairs_np).to(primary_device)
                entropies = kernel_obj.compute_single_gpu(
                    bin_arrs, pairs_t, b_max, n_per_var,
                    tau, batch_size, primary_device,
                )
            else:
                # Chunked transfer: compute in batches, keep pairs on CPU
                bytes_per_pair = pairs_np.strides[0]  # 16 bytes (2 × int64)
                max_pairs_per_chunk = max(1024, pairs_budget // bytes_per_pair)
                n_chunks = (n_pairs + max_pairs_per_chunk - 1) // max_pairs_per_chunk
                vprint(f"[TENEX] pairs chunked: {n_chunks} chunks "
                      f"({max_pairs_per_chunk:,} pairs/chunk, "
                      f"free={free_mem / 2**30:.1f} GB)")
                chunks = []
                for ci in range(n_chunks):
                    beg = ci * max_pairs_per_chunk
                    end = min(beg + max_pairs_per_chunk, n_pairs)
                    p_chunk = torch.from_numpy(pairs_np[beg:end]).to(primary_device)
                    ent_chunk = kernel_obj.compute_single_gpu(
                        bin_arrs, p_chunk, b_max, n_per_var,
                        tau, batch_size, primary_device,
                    )
                    chunks.append(ent_chunk)
                    del p_chunk
                entropies = np.concatenate(chunks)
        else:
            # Use post-prepare data (kernel_obj.prepare may modify bin_arrs,
            # e.g. Adaptive-SMEM bin coarsening — CPU cache is pre-prepare)
            bin_cpu = bin_arrs.cpu() if bin_arrs.is_cuda else bin_arrs
            npg_cpu = n_per_var.cpu() if n_per_var.is_cuda else n_per_var
            entropies = kernel_obj.compute_multi_gpu(
                bin_cpu, pairs_np, b_max, npg_cpu,
                tau, batch_size, device_ids, smem_optin=smem_optin,
            )

        # ── 12. Assemble result ───────────────────────────────────────────────
        if profile and on_cuda:
            _ev[6].record(torch.cuda.current_stream(primary_device))

        self._last_run_info = _run_info
        result = self._assemble_result(entropies, pairs_np, n_vars, t0)

        if profile and on_cuda:
            _ev[7].record(torch.cuda.current_stream(primary_device))
            torch.cuda.synchronize(primary_device)
            _timings.update(self._collect_timings(_ev, t0, 'pair-based'))
            self._last_run_info['timings'] = _timings
        return result

    # ── Private: variable-blocked computation (memory-constrained GPUs) ──

    def _compute_gene_blocked(self, kernel_obj, bin_arrs_cpu, n_per_var_cpu,
                              n_vars, b_max, tau, batch_size,
                              device_ids, t0):
        """Compute TE via 2D variable blocks for memory-constrained GPUs.

        bin_arrs stays on CPU.  For each (target_block, source_block),
        only the required variable rows are transferred to GPU, TE is
        computed, and the GPU memory is freed before the next block.
        """
        from concurrent.futures import ThreadPoolExecutor

        # Determine block size from the smallest GPU's free memory.
        # Budget must cover: bin_arrs subset + output tensor.
        # Cross pair-free eliminates the pairs tensor (16B/pair savings).
        #   bin_arrs: 2 * B * bytes_per_var   (worst case: disjoint target/source)
        #   output:   B * B * 4                (float32 per pair)
        # Total per-pair: 4 bytes (output only, no pairs tensor)
        # Solve:  4*B² + 2*bpg*B ≤ budget   ->  quadratic in B
        # Fallback: if kernel has no cross pair-free, use 20*B² (with pairs tensor)
        import math as _math
        use_cross_pairfree = (hasattr(kernel_obj, 'compute_pairfree_cross')
                              and self._sources is None)
        torch.cuda.empty_cache()
        min_free = min(torch.cuda.mem_get_info(d)[0] for d in device_ids)
        bytes_per_var = (bin_arrs_cpu[0].nelement()
                          * bin_arrs_cpu.element_size())
        # Reserve 15% for allocator fragmentation + CUDA driver
        reserve = max(128 * 1024 * 1024, int(min_free * 0.15))
        budget = max(0, min_free - reserve)
        # Quadratic: a*B² + 2*bpg*B - budget ≤ 0
        a = 4 if use_cross_pairfree else 20  # output only vs pairs+output
        b_coef = 2 * bytes_per_var
        c = -budget
        disc = b_coef * b_coef - 4 * a * c
        block_size = max(1, min(int((-b_coef + _math.isqrt(int(disc))) // (2 * a)),
                                n_vars))

        # Multi-GPU: ensure enough blocks for GPU distribution.
        # Total jobs = n_blocks^2. We need n_blocks^2 >= n_gpus for any
        # distribution at all. For decent load balance with round-robin,
        # n_blocks >= n_gpus is sufficient (n_blocks^2 >= n_gpus^2 jobs).
        # Larger n_blocks = more overhead (data transfers), so keep it minimal.
        n_gpus = len(device_ids)
        if n_gpus > 1:
            min_blocks = n_gpus
            max_block_for_dist = max(1, n_vars // min_blocks)
            block_size = min(block_size, max_block_for_dist)

        n_blocks = (n_vars + block_size - 1) // block_size
        n_pairs_total = n_vars * (n_vars - 1)

        # Source filter: valid source variable indices
        source_idx_arr = None
        if self._sources is not None:
            _, inds_source, _ = np.intersect1d(
                self._variable_names, self._sources, return_indices=True,
            )
            source_idx_arr = inds_source.astype(np.int64)

        _mode = "cross pair-free" if use_cross_pairfree else "pair-based"
        vprint(f"[TENEX] var-blocked ({_mode}): {block_size} vars/block, "
              f"{n_blocks}x{n_blocks}={n_blocks ** 2} blocks "
              f"(budget={budget / 2**30:.1f} GB, "
              f"bin/var={bytes_per_var / 1024:.0f} KB)")
        vprint(f"[TENEX] pairs: {n_pairs_total:,}")

        result_matrix = np.zeros((n_vars, n_vars), dtype=np.float32)

        # Build block jobs: (t_start, t_end, s_start, s_end)
        jobs = []
        for tb in range(n_blocks):
            t_s = tb * block_size
            t_e = min(t_s + block_size, n_vars)
            for sb in range(n_blocks):
                s_s = sb * block_size
                s_e = min(s_s + block_size, n_vars)
                jobs.append((t_s, t_e, s_s, s_e))

        def _process_block(t_start, t_end, s_start, s_end, device):
            """Process one (target, source) variable block on device."""
            targets = np.arange(t_start, t_end, dtype=np.int64)
            sources = np.arange(s_start, s_end, dtype=np.int64)
            is_diagonal = (t_start == s_start and t_end == s_end)

            if use_cross_pairfree and not is_diagonal:
                # ── Cross pair-free: no pairs tensor needed ──
                # bin_sub layout: [tgt_rows | src_rows]
                var_indices = np.concatenate([targets, sources])
                bin_sub = bin_arrs_cpu[var_indices].to(device)
                npg_sub = n_per_var_cpu[var_indices].to(device)

                n_tgt = len(targets)
                n_src = len(sources)
                ents = kernel_obj.compute_pairfree_cross(
                    bin_sub, npg_sub, n_tgt, n_src, tau, device,
                )
                del bin_sub, npg_sub

                # Build global pair indices for result_matrix assignment
                tt, ss = np.meshgrid(targets, sources, indexing='ij')
                pairs_global = np.column_stack([tt.ravel(), ss.ravel()])
                return pairs_global, ents

            elif use_cross_pairfree and is_diagonal:
                # ── Diagonal block: standard pair-free (self-pairs excluded) ──
                bin_sub = bin_arrs_cpu[targets].to(device)
                npg_sub = n_per_var_cpu[targets].to(device)

                n_local = len(targets)
                ents = kernel_obj.compute_pairfree(
                    bin_sub, n_local, b_max, npg_sub, tau, device,
                )
                del bin_sub, npg_sub

                # Build global pair indices (excluding self-pairs)
                n_local = len(targets)
                idx = np.arange(n_local * (n_local - 1), dtype=np.int64)
                tgt_local = idx // (n_local - 1)
                src_local = idx % (n_local - 1)
                src_local[src_local >= tgt_local] += 1
                pairs_global = np.column_stack([
                    targets[tgt_local], targets[src_local],
                ])
                return pairs_global, ents

            # ── Fallback: pair-based (TF filter or no cross support) ──
            tt, ss = np.meshgrid(targets, sources, indexing='ij')
            pairs_all = np.column_stack([tt.ravel(), ss.ravel()])

            # Remove self-pairs
            mask = pairs_all[:, 0] != pairs_all[:, 1]

            # Apply TF filter if present
            if source_idx_arr is not None:
                mask &= np.isin(pairs_all[:, 1], source_idx_arr)
            pairs_global = pairs_all[mask]

            if len(pairs_global) == 0:
                return None

            # Unique variables = target_range ∪ source_range
            unique_vars = np.union1d(targets, sources)

            # Global → local index mapping
            inv_map = np.empty(n_vars, dtype=np.int64)
            inv_map[unique_vars] = np.arange(len(unique_vars), dtype=np.int64)
            local_pairs = inv_map[pairs_global]

            # Transfer variable subset to GPU
            bin_sub = bin_arrs_cpu[unique_vars].to(device)
            npg_sub = n_per_var_cpu[unique_vars].to(device)

            # VRAM-aware: chunk pairs to avoid OOM from pairs tensor + output
            n_block_pairs = len(local_pairs)
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info(device)
            _reserve = max(64 * 1024 * 1024, int(free_mem * 0.15))
            _avail = max(0, free_mem - _reserve)
            
            # Each pair needs 16B (pairs tensor) + 4B (output) = 20B
            max_pairs_chunk = max(1024, _avail // 20)

            if n_block_pairs <= max_pairs_chunk:
                pairs_t = torch.from_numpy(local_pairs).to(device)
                ents = kernel_obj.compute_single_gpu(
                    bin_sub, pairs_t, b_max, npg_sub,
                    tau, batch_size, device,
                )
                del pairs_t
            else:
                ent_chunks = []
                for ci in range(0, n_block_pairs, max_pairs_chunk):
                    ce = min(ci + max_pairs_chunk, n_block_pairs)
                    p_chunk = torch.from_numpy(local_pairs[ci:ce]).to(device)
                    ec = kernel_obj.compute_single_gpu(
                        bin_sub, p_chunk, b_max, npg_sub,
                        tau, batch_size, device,
                    )
                    ent_chunks.append(ec)
                    del p_chunk
                ents = np.concatenate(ent_chunks)

            del bin_sub, npg_sub
            return pairs_global, ents

        if len(device_ids) <= 1:
            dev = self._primary_device(device_ids)
            for ji, (t_s, t_e, s_s, s_e) in enumerate(jobs):
                ret = _process_block(t_s, t_e, s_s, s_e, dev)
                if ret is not None:
                    pg, ents = ret
                    result_matrix[pg[:, 0], pg[:, 1]] = ents
        else:
            # Multi-GPU: each GPU gets its own worker thread
            n_gpus = len(device_ids)
            jobs_per_gpu = [[] for _ in range(n_gpus)]
            for ji, job in enumerate(jobs):
                jobs_per_gpu[ji % n_gpus].append(job)

            def _gpu_worker(rank):
                dev_id = device_ids[rank]
                torch.cuda.set_device(dev_id)
                dev = torch.device(f'cuda:{dev_id}')
                local_results = []
                for (t_s, t_e, s_s, s_e) in jobs_per_gpu[rank]:
                    ret = _process_block(t_s, t_e, s_s, s_e, dev)
                    if ret is not None:
                        local_results.append(ret)
                return local_results

            with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                futures = [pool.submit(_gpu_worker, r) for r in range(n_gpus)]
                for fut in futures:
                    for pg, ents in fut.result():
                        result_matrix[pg[:, 0], pg[:, 1]] = ents

        self._result_matrix = result_matrix.T  # [i,j] = TE(i→j)

        elapsed = time.time() - t0
        vprint(f"[TENEX] done -- {elapsed:.2f}s  "
              f"({n_pairs_total / elapsed:,.0f} pairs/s)")
        return self._result_matrix

    # ── Private: VRAM-aware pair-free computation ──────────────────────────────

    @staticmethod
    def _compute_pairfree_chunked(kernel_obj, bin_arrs, n_per_var,
                                   n_vars, b_max, tau, device):
        """Pair-free with VRAM-aware chunking.

        The output tensor for pair-free is n_pairs * 4 bytes.  For large
        datasets (e.g. CeNGEN: 504M pairs = 1.88 GB) this can exceed
        available VRAM on small GPUs (e.g. 2080 Ti 11 GB).

        This method queries free VRAM, computes how many pairs fit in one
        chunk (output + kernel overhead), and calls compute_pairfree in
        chunks, concatenating the results on CPU.
        """
        n_pairs = n_vars * (n_vars - 1)
        output_bytes = n_pairs * 4  # float32 output

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info(device)
            # Reserve for allocator fragmentation and CUDA driver state
            reserve = max(128 * 1024 * 1024, int(free_mem * 0.15))
            available = max(0, free_mem - reserve)
        else:
            available = output_bytes  # CPU: no limit

        if output_bytes <= available:
            # All pairs fit in one shot
            return kernel_obj.compute_pairfree(
                bin_arrs, n_vars, b_max, n_per_var,
                tau, device,
            )

        # Chunk: each chunk output = chunk_pairs * 4 bytes
        max_pairs_per_chunk = max(1024, available // 4)
        n_chunks = (n_pairs + max_pairs_per_chunk - 1) // max_pairs_per_chunk
        vprint(f"[TENEX] pair-free chunked: {n_chunks} chunks "
              f"({max_pairs_per_chunk:,} pairs/chunk, "
              f"free={free_mem / 2**30:.1f} GB, "
              f"output={output_bytes / 2**30:.1f} GB)")

        # Pre-allocate output to avoid np.concatenate
        output = np.empty(n_pairs, dtype=np.float32)
        offset = 0
        for _ in range(n_chunks):
            n_local = min(max_pairs_per_chunk, n_pairs - offset)
            if n_local <= 0:
                break
            ent_chunk = kernel_obj.compute_pairfree(
                bin_arrs, n_vars, b_max, n_per_var,
                tau, device, pair_offset=offset, n_pairs_local=n_local,
            )
            output[offset:offset + n_local] = ent_chunk
            offset += n_local

        return output

    # ── Private: kernel selection ─────────────────────────────────────────────

    @staticmethod
    def _select_kernel(kernel_name, b_max, on_cuda, smem_optin,
                       smem_bytes, n_per_var, source_filter):
        """Select kernel: explicit name or auto_select().

        When a kernel is explicitly requested, its supports() method is
        checked against the current data configuration.  If the kernel
        is incompatible, a ValueError is raised instead of silently
        producing incorrect results.
        """
        from tenex.kernels import auto_select, get_kernel as _get

        if kernel_name is not None:
            k = _get(kernel_name)
            # Hard constraint specific to GEMM-B2 (better error than supports()):
            if k.name == "GEMM-B2" and b_max != 2:
                raise ValueError(
                    f"kernel='GEMM-B2' requires binary data (b_max=2), "
                    f"but this data has b_max={b_max}. "
                    f"Use kernel=None for auto-selection."
                )
            # Generic supports() check: same predicate auto_select() uses.
            if not k.supports(b_max, on_cuda, smem_optin, smem_bytes,
                              n_per_var, source_filter):
                raise ValueError(
                    f"kernel={kernel_name!r} does not support this configuration "
                    f"(b_max={b_max}, on_cuda={on_cuda}, "
                    f"source_filter={source_filter}). "
                    f"Use kernel=None for auto-selection."
                )
            return k
        return auto_select(b_max, on_cuda, smem_optin, smem_bytes,
                           n_per_var, source_filter)

    # ── Private: matrix kernel path ───────────────────────────────────────────

    def _run_matrix_kernel(self, kernel, data, n_vars, tau, device_ids, t0,
                           **kwargs):
        """Execute a matrix kernel (GEMM-B2) and return result."""
        n_pairs = n_vars * (n_vars - 1)
        vprint(f"[TENEX] pairs: {n_pairs:,}")
        vprint(f"[TENEX] kernel: {kernel.log_description(tau=tau, **kwargs)}")

        if len(device_ids) <= 1:
            primary = self._primary_device(device_ids)
            te_matrix = kernel.compute_matrix(data, n_vars, tau, primary, **kwargs)
        else:
            te_matrix = kernel.compute_matrix_multi_gpu(
                data, n_vars, tau, device_ids, **kwargs
            )

        self._result_matrix = te_matrix.T  # [i,j] = TE(i→j)
        elapsed = time.time() - t0
        vprint(f"[TENEX] done -- {elapsed:.3f}s  ({n_pairs / elapsed:,.0f} pairs/s)")
        return self._wrap_result()

    # ── Private: pair-based result assembly ───────────────────────────────────

    def _assemble_result(self, entropies, pairs_np, n_vars, t0):
        """Assemble per-pair entropies into (n_vars, n_vars) result matrix."""
        n_pairs = len(pairs_np)
        result_matrix = np.zeros((n_vars, n_vars), dtype=np.float32)
        result_matrix[pairs_np[:, 0], pairs_np[:, 1]] = entropies
        self._result_matrix = result_matrix.T  # [i,j] = TE(i→j)

        elapsed = time.time() - t0
        vprint(f"[TENEX] done -- {elapsed:.2f}s  ({n_pairs / elapsed:,.0f} pairs/s)")
        return self._wrap_result()

    def _assemble_result_pairfree(self, entropies, n_vars, t0):
        """Assemble pair-free entropies (linear order) into (n_vars, n_vars) matrix.

        Linear pair ordering: pair_id = tgt * (n_vars-1) + src_local
        where src = src_local + (1 if src_local >= tgt else 0).
        """
        n_pairs = n_vars * (n_vars - 1)
        nm1 = n_vars - 1
        result_matrix = np.zeros((n_vars, n_vars), dtype=np.float32)
        ent_2d = entropies.reshape(n_vars, nm1)

        # For target variable i: sources are [0..i-1, i+1..n-1]
        for i in range(n_vars):
            result_matrix[i, :i] = ent_2d[i, :i]       # src_local 0..i-1 -> src 0..i-1
            result_matrix[i, i+1:] = ent_2d[i, i:]     # src_local i..nm1-1 -> src i+1..n-1

        self._result_matrix = result_matrix.T  # [i,j] = TE(i→j)

        elapsed = time.time() - t0
        vprint(f"[TENEX] done -- {elapsed:.2f}s  ({n_pairs / elapsed:,.0f} pairs/s)")
        return self._wrap_result()

    def _wrap_result(self) -> TransferEntropyResult:
        """Wrap the result matrix with metadata into TransferEntropyResult."""
        info = getattr(self, '_last_run_info', {})
        bin_arrs = getattr(self, '_last_bin_arrs', None)
        n_per_var = getattr(self, '_last_n_per_var', None)
        if bin_arrs is not None and isinstance(bin_arrs, torch.Tensor):
            bin_arrs = bin_arrs.cpu().numpy()
        if n_per_var is not None and isinstance(n_per_var, torch.Tensor):
            n_per_var = n_per_var.cpu().numpy()

        return TransferEntropyResult(
            matrix=self._result_matrix,
            variable_names=self._variable_names,
            bin_arrs=bin_arrs,
            n_per_var=n_per_var,
            b_max=info.get('b_max', 0),
            tau=info.get('tau', 1),
            kernel=info.get('kernel', ''),
            timings=info.get('timings', None),
            sources=self._sources,
        )
