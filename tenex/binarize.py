"""Binarization algorithms for TENEX preprocessing.

A binarizer maps a continuous ``(n, T)`` expression matrix to a binary ``{0, 1}``
bin matrix (``b_max = 2``), which routes the compute to the GEMM-B2 kernel. This
is a two-state special case of discretization, exposed as a separate
``binarization=`` preprocessing option on the engine and the pipeline so a run
can opt into the binary approximation without hand-preparing the data.

Only GPU-vectorizable methods are provided. Each is a per-variable threshold
computed as a reduction over the time axis, so it runs entirely as elementwise
and reduction ops with no per-gene distribution fitting. The slower fitting-based
binarizers of scBoolSeq (bimodal Gaussian mixture, dip-test classifier) are not
included.

Each algorithm ships four interchangeable backends:

- ``numpy`` : CPU reference.
- ``torch`` : GPU via PyTorch (default when the device is CUDA).
- ``cupy``  : GPU via CuPy (optional dependency ``cupy-cuda1Xx``). The result is
  handed back to PyTorch through DLPack with no host round-trip or device copy.
- ``cuda``  : native CUDA kernels (tenex.kernels.binarize_kernel), fused and
  streaming so they never hit the sort-memory limit of torch.quantile, and with
  a single/multi-GPU row-split. Returns a torch tensor.

The dispatcher always returns the same 4-tuple as
:func:`tenex.preprocess.discretize` with a ``torch.Tensor`` bin matrix, so the
binarization path is interchangeable with the multi-bin binning path everywhere
downstream (kernel selection, dispatch, result assembly) regardless of backend.

Available methods
-----------------
nonzero
    Active where the value is nonzero, ``1[x > 0]``. The zero-inflated binarizer
    of scBoolSeq (Magana-Lopez et al., PLOS Comput Biol 2024).
mean
    Active above the per-variable mean, ``1[x > mean_g]``.
median
    Active above the per-variable median, ``1[x > median_g]``.
quantile
    Active above a per-variable quantile, ``1[x > Q_g(q)]`` (parameter ``q``,
    default 0.5).
"""
import numpy as np
import torch


# ── Algorithms, one implementation per backend ────────────────────────────────
# Each takes a (n, T) array and returns a (n, T) int8 {0, 1} array on the same
# backend. Per-variable thresholds reduce over the time axis (axis/dim = 1).

def _nonzero_numpy(arr, **_):
    return (arr > 0).astype(np.int8)


def _nonzero_torch(arr, **_):
    return (arr > 0).to(torch.int8)


def _nonzero_cupy(arr, **_):
    import cupy as cp
    return (cp.asarray(arr) > 0).astype(cp.int8)


def _mean_numpy(arr, **_):
    return (arr > arr.mean(axis=1, keepdims=True)).astype(np.int8)


def _mean_torch(arr, **_):
    return (arr > arr.mean(dim=1, keepdim=True)).to(torch.int8)


def _mean_cupy(arr, **_):
    import cupy as cp
    a = cp.asarray(arr)
    return (a > a.mean(axis=1, keepdims=True)).astype(cp.int8)


def _check_q(q):
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"quantile q must be in [0, 1] (got {q})")
    return float(q)


def _rows_per_chunk(n, T, free_bytes, safety=3.0, frac=0.5):
    """Rows per block so a per-row quantile sort fits in ``free_bytes``.

    ``torch.quantile`` / ``cp.quantile`` sort along the time axis and allocate
    roughly ``safety`` times the block size, so cap that at ``frac`` of the free
    memory (mirrors the adaptive-chunking used by discretize()). Returns ``n``
    (single pass) when the whole matrix fits.
    """
    per_row = T * 4 * safety
    rows = max(1, int(free_bytes * frac / per_row))
    return min(n, int(rows))


def _quantile_binarize_torch(arr, q):
    """Per-variable quantile threshold on the GPU, chunked over variables.

    Bounds the sort memory of ``torch.quantile`` so large time-point counts
    (e.g. 10^5 cells) do not exhaust device memory.
    """
    n, T = arr.shape
    if arr.is_cuda:
        chunk = _rows_per_chunk(n, T, torch.cuda.mem_get_info(arr.device)[0])
    else:
        chunk = n
    if chunk >= n:
        return (arr > torch.quantile(arr, q, dim=1, keepdim=True)).to(torch.int8)
    out = torch.empty((n, T), dtype=torch.int8, device=arr.device)
    for i in range(0, n, chunk):
        blk = arr[i:i + chunk]
        thr = torch.quantile(blk, q, dim=1, keepdim=True)
        out[i:i + chunk] = (blk > thr).to(torch.int8)
    return out


def _quantile_binarize_cupy(arr, q):
    import cupy as cp
    a = cp.asarray(arr)
    n, T = a.shape
    chunk = _rows_per_chunk(n, T, int(cp.cuda.Device().mem_info[0]))
    if chunk >= n:
        return (a > cp.quantile(a, q, axis=1, keepdims=True)).astype(cp.int8)
    out = cp.empty((n, T), dtype=cp.int8)
    for i in range(0, n, chunk):
        blk = a[i:i + chunk]
        out[i:i + chunk] = (blk > cp.quantile(blk, q, axis=1, keepdims=True)).astype(cp.int8)
    return out


def _median_numpy(arr, **_):
    return (arr > np.median(arr, axis=1, keepdims=True)).astype(np.int8)


def _median_torch(arr, **_):
    return _quantile_binarize_torch(arr, 0.5)      # linear-interp median


def _median_cupy(arr, **_):
    # cp.quantile(0.5) rather than cp.median: matches the torch backend's
    # linear-interpolation median and avoids a cp.median kernel that fails to
    # JIT-compile on CUDA 13.x.
    return _quantile_binarize_cupy(arr, 0.5)


def _quantile_numpy(arr, q=0.5, **_):
    q = _check_q(q)
    return (arr > np.quantile(arr, q, axis=1, keepdims=True)).astype(np.int8)


def _quantile_torch(arr, q=0.5, **_):
    return _quantile_binarize_torch(arr, _check_q(q))


def _quantile_cupy(arr, q=0.5, **_):
    return _quantile_binarize_cupy(arr, _check_q(q))


# name -> {backend: impl, ..., 'doc': str}
_BINARIZERS = {
    'nonzero': {
        'numpy': _nonzero_numpy, 'torch': _nonzero_torch,
        'cupy': _nonzero_cupy,
        'doc': 'active where nonzero (x > 0); scBoolSeq zero-inflated rule',
    },
    'mean': {
        'numpy': _mean_numpy, 'torch': _mean_torch, 'cupy': _mean_cupy,
        'doc': 'active above the per-variable mean (x > mean_g)',
    },
    'median': {
        'numpy': _median_numpy, 'torch': _median_torch, 'cupy': _median_cupy,
        'doc': 'active above the per-variable median (x > median_g)',
    },
    'quantile': {
        'numpy': _quantile_numpy, 'torch': _quantile_torch,
        'cupy': _quantile_cupy,
        'doc': 'active above a per-variable quantile (x > Q_g(q), q default 0.5)',
    },
}

_BACKENDS = ('numpy', 'torch', 'cupy', 'cuda')


def available_binarizers() -> list:
    """Return the sorted list of registered binarization method names."""
    return sorted(_BINARIZERS)


def binarizer_doc(method: str) -> str:
    """Return the one-line description of a binarization method."""
    return _BINARIZERS[method.lower()]['doc']


def cupy_available() -> bool:
    """Return True when the CuPy backend can be imported."""
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


# ── Backend helpers ───────────────────────────────────────────────────────────

def _resolve_backend(backend, use_numpy, device) -> str:
    """Pick a backend when the caller passes ``backend=None``.

    Preserves the earlier ``use_numpy`` behaviour (CPU numpy vs GPU torch) and
    forces numpy whenever the target device is not CUDA.
    """
    if backend is not None:
        backend = backend.lower()
        if backend not in _BACKENDS:
            raise ValueError(
                f"unknown backend '{backend}'. available: {list(_BACKENDS)}")
        return backend
    if device.type != 'cuda':
        return 'numpy'
    return 'numpy' if use_numpy else 'torch'


# ── Dispatcher ────────────────────────────────────────────────────────────────

def binarize(arr, method: str = 'nonzero', device: torch.device = None,
             use_numpy: bool = True, backend: str = None, **params) -> tuple:
    """Binarize a continuous ``(n, T)`` matrix into a ``{0, 1}`` bin array.

    Parameters
    ----------
    arr       : (n, T) float array (numpy or torch), already aligned.
    method    : binarization algorithm name (see :func:`available_binarizers`).
    device    : target torch device (default: cuda if available else cpu).
    use_numpy : when ``backend`` is None, choose CPU numpy (True) or GPU torch
                (False). Forced to numpy when the device is not CUDA.
    backend   : force a backend, one of 'numpy', 'torch', 'cupy'. Overrides
                ``use_numpy`` when set. 'cupy' needs the optional ``cupy``
                package and a CUDA device.
    **params  : method-specific parameters (e.g. ``q`` for 'quantile').

    Returns
    -------
    bin_arr   : (n, T, 1) int8 torch.Tensor on ``device``, values in {0, 1}.
    n_bins    : (n,) int32 torch.Tensor, distinct states per variable (1 or 2).
    b_max     : int, max distinct states across variables (2 unless a variable
                is constant across all time points).
    n_per_var : (n,) int32 torch.Tensor, same as ``n_bins`` for binary data.
    """
    key = method.lower()
    if key not in _BINARIZERS:
        raise ValueError(
            f"unknown binarization '{method}'. "
            f"available: {available_binarizers()}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resolved = _resolve_backend(backend, use_numpy, device)
    if resolved in ('cupy', 'cuda') and device.type != 'cuda':
        raise ValueError(f"the {resolved} backend requires a CUDA device")

    if resolved == 'cuda':
        from tenex.kernels.binarize_kernel import binarize_cuda
        bin2d = binarize_cuda(arr, method=key, device=device,
                              device_ids=params.get('device_ids'),
                              q=params.get('q', 0.5))     # (n, T) int8 torch
    elif resolved == 'numpy':
        fn = _BINARIZERS[key][resolved]
        a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) \
            else np.asarray(arr)
        a = np.ascontiguousarray(a, dtype=np.float32)
        bin2d = torch.from_numpy(np.ascontiguousarray(fn(a, **params))).to(device)
    elif resolved == 'torch':
        fn = _BINARIZERS[key][resolved]
        a = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)
        bin2d = fn(a.to(device).float(), **params)           # (n, T) int8
    else:  # cupy: compute on the target device, hand back via DLPack (no copy)
        fn = _BINARIZERS[key][resolved]
        import cupy as cp
        idx = device.index if device.index is not None else 0
        # Set the global current device (mirrors the torch.cuda.set_device()
        # pattern used across the engine) rather than a scoped context.
        cp.cuda.Device(idx).use()
        a = arr.detach().to(device).float() if isinstance(arr, torch.Tensor) \
            else arr
        # clone so the tensor owns its memory beyond the CuPy pool block
        bin2d = torch.from_dlpack(fn(a, **params)).clone()

    bin_arr = bin2d.unsqueeze(2)                              # (n, T, 1)

    # Distinct states per variable without a dense remap (GEMM-B2 reads the raw
    # {0, 1} indicators): for binary data this is 1 if constant, else 2.
    flat = bin_arr.reshape(bin_arr.shape[0], -1)
    n_per_var = (flat.amax(1) - flat.amin(1) + 1).to(torch.int32)
    b_max = int(n_per_var.max().item())
    return bin_arr, n_per_var.clone(), b_max, n_per_var
