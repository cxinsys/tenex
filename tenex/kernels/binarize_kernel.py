"""Native CUDA kernels for binarization (fourth backend, alongside numpy/torch/cupy).

Two kernels, both streaming and allocation-free beyond the (n, T) output:

- ``mean_binarize_kernel``  : one block per variable. Pass 1 reduces the row to
  its mean, pass 2 writes ``1[x > mean]``. No per-row temporary, so it never
  hits the sort-memory limit that ``torch.quantile`` does.
- ``threshold_binarize_kernel`` : elementwise ``1[x > thr[g]]`` for a supplied
  per-variable threshold vector. Used by 'nonzero' (thr = 0) and by
  'median' / 'quantile' (exact threshold computed by a chunked sort).

Binarization is one memory-bound pass, so it runs on a single GPU. Large inputs
that exceed device memory are streamed in adaptive row-chunks on that one GPU
(the strategy discretize() uses), which keeps the result on the target device
with no cross-device gather. Spreading the pass across GPUs is counterproductive
here. The saving is tiny and gathering the blocks back through host memory
(direct GPU-to-GPU copy is unreliable on this PCIe topology) costs far more, so
a single GPU is ~30x faster than a 4-GPU split plus gather on CeNGEN. The result
is always a torch int8 tensor.
"""
import threading
from typing import Optional

import torch

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// One block per variable: reduce the row to its mean, then write 1[x > mean].
__global__ void mean_binarize_kernel(
    const float* __restrict__ x,   // (n, T) row-major
    int8_t*      __restrict__ out, // (n, T)
    int n, long T)
{
    const int g = blockIdx.x;
    if (g >= n) return;
    const float* row  = x   + (long)g * T;
    int8_t*      orow = out + (long)g * T;
    const int tid = threadIdx.x, B = blockDim.x, W = B >> 5;
    extern __shared__ float sbuf[];             // W floats

    float s = 0.f;
    for (long t = tid; t < T; t += B) s += row[t];
    s = warp_reduce_sum(s);
    if ((tid & 31) == 0) sbuf[tid >> 5] = s;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < W) ? sbuf[tid] : 0.f;
        v = warp_reduce_sum(v);
        if (tid == 0) sbuf[0] = v / (float)T;   // mean
    }
    __syncthreads();
    const float thr = sbuf[0];
    for (long t = tid; t < T; t += B)
        orow[t] = (row[t] > thr) ? (int8_t)1 : (int8_t)0;
}

// Elementwise 1[x > thr[row]] with a supplied per-variable threshold.
__global__ void threshold_binarize_kernel(
    const float* __restrict__ x,   // (n, T)
    const float* __restrict__ thr, // (n,)
    int8_t*      __restrict__ out, // (n, T)
    int n, long T)
{
    const long total = (long)n * T;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (long)gridDim.x * blockDim.x) {
        const int g = (int)(i / T);
        out[i] = (x[i] > thr[g]) ? (int8_t)1 : (int8_t)0;
    }
}

torch::Tensor mean_binarize(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32,
                "mean_binarize expects a 2-D float32 CUDA tensor");
    const c10::cuda::CUDAGuard guard(x.device());
    x = x.contiguous();
    const int  n = (int)x.size(0);
    const long T = (long)x.size(1);
    auto out = torch::empty({x.size(0), x.size(1)},
                            torch::dtype(torch::kInt8).device(x.device()));
    const int B = 256, W = B / 32;
    mean_binarize_kernel<<<n, B, W * sizeof(float),
                           at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<int8_t>(), n, T);
    return out;
}

torch::Tensor threshold_binarize(torch::Tensor x, torch::Tensor thr) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32,
                "threshold_binarize expects a 2-D float32 CUDA tensor");
    const c10::cuda::CUDAGuard guard(x.device());
    x = x.contiguous();
    thr = thr.to(torch::kFloat32).contiguous();
    const int  n = (int)x.size(0);
    const long T = (long)x.size(1);
    auto out = torch::empty({x.size(0), x.size(1)},
                            torch::dtype(torch::kInt8).device(x.device()));
    const long total = (long)n * T;
    const int  B = 256;
    const int  grid = (int)std::min<long>((total + B - 1) / B, 65535L);
    threshold_binarize_kernel<<<grid, B, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), thr.data_ptr<float>(), out.data_ptr<int8_t>(), n, T);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_binarize", &mean_binarize, "fused per-variable mean binarize");
    m.def("threshold_binarize", &threshold_binarize, "per-variable threshold binarize");
}
"""

_module = None
_lock = threading.Lock()


def _load_module():
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is not None:
            return _module
        try:
            import tenex._ext.binarize_cuda as _mod          # AOT
            _module = _mod
            return _module
        except ImportError:
            pass
        from torch.utils.cpp_extension import load_inline     # JIT fallback
        _module = load_inline(
            name='binarize_cuda',
            cpp_sources='',
            cuda_sources=_CUDA_SRC,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
        return _module


def _rows_per_chunk(n, T, free_bytes, safety=3.0, frac=0.5):
    """Rows whose quantile sort fits in ``free_bytes`` (adaptive chunking)."""
    per_row = T * 4 * safety
    return min(n, max(1, int(free_bytes * frac / per_row)))


def _quantile_threshold(x, q):
    """Exact per-variable quantile threshold (n,), chunked to bound sort memory."""
    n, T = x.shape
    free = torch.cuda.mem_get_info(x.device)[0]
    chunk = _rows_per_chunk(n, T, free)
    if chunk >= n:
        return torch.quantile(x, q, dim=1)
    thr = torch.empty(n, device=x.device, dtype=torch.float32)
    for i in range(0, n, chunk):
        thr[i:i + chunk] = torch.quantile(x[i:i + chunk], q, dim=1)
    return thr


def _binarize_one_device(x, method, q):
    """Binarize a (n, T) float32 CUDA tensor on its current device -> int8 (n, T)."""
    mod = _load_module()
    x = x.contiguous()
    if method == 'mean':
        return mod.mean_binarize(x)
    if method == 'nonzero':
        thr = torch.zeros(x.size(0), device=x.device, dtype=torch.float32)
        return mod.threshold_binarize(x, thr)
    # median / quantile: exact threshold via chunked sort, then the compare kernel
    qq = 0.5 if method == 'median' else _check_q(q)
    thr = _quantile_threshold(x, qq)
    return mod.threshold_binarize(x, thr)


def _check_q(q):
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"quantile q must be in [0, 1] (got {q})")
    return float(q)


def _mem_safety(method):
    # mean/nonzero stream in place (~2x block); median/quantile sort (~5x block).
    return 5.0 if method in ('median', 'quantile') else 2.0


def _binarize_chunked(x, method, q, dev, out):
    """Binarize ``(m, T)`` ``x`` on GPU ``dev`` with adaptive row-chunking.

    Writes the int8 result into ``out`` (a preallocated ``(m, T)`` int8 tensor on
    any device). Rows are independent for every method, so the input is streamed
    in row-blocks sized to the free memory of ``dev``. ``out`` is either on
    ``dev`` (same-device write) or on the host (D2H); it is never on a different
    GPU, so no unreliable GPU-to-GPU copy is issued.
    """
    torch.cuda.set_device(dev)
    ddev = torch.device(f'cuda:{dev}')
    m, T = x.shape
    free = torch.cuda.mem_get_info(ddev)[0]
    per_row = T * 4 * _mem_safety(method)
    chunk = min(m, max(1, int(free * 0.5 / per_row)))
    for i in range(0, m, chunk):
        res = _binarize_one_device(x[i:i + chunk].to(ddev), method, q)
        out[i:i + chunk] = res if out.device == ddev else res.to(out.device)
    torch.cuda.synchronize(ddev)


def binarize_cuda(arr, method='nonzero', device=None, device_ids=None, q=0.5):
    """Binarize a continuous ``(n, T)`` matrix with the native CUDA kernels.

    Binarization is one memory-bound pass, so a single GPU is used whenever the
    ``(n, T)`` int8 result fits in device memory. Large inputs are streamed in
    adaptive row-chunks on that one GPU, so the result stays on the target device
    with no cross-device gather. Multi-GPU is engaged only when the result is too
    large for a single GPU and several devices are supplied. Then the variables
    (rows) are split across the devices, each block streamed on its own GPU, and
    the int8 blocks gathered on the host (the result cannot live on one GPU, and
    direct GPU-to-GPU copy is unreliable on this PCIe topology). For data that
    fits a single GPU that is ~30x faster than a split plus gather, so multi-GPU
    is reserved for the out-of-core case where it is actually required.

    Parameters
    ----------
    arr        : (n, T) float array (numpy or torch).
    method     : 'nonzero' | 'mean' | 'median' | 'quantile'.
    device     : CUDA device for the single-GPU compute and result (default:
                 current, or ``device_ids[0]`` when given).
    device_ids : GPUs available for the multi-GPU out-of-core path. Ignored while
                 the result fits a single GPU.
    q          : quantile for the 'quantile' method.

    Returns
    -------
    (n, T) int8 torch.Tensor. On ``device`` for the single-GPU path, on the host
    for the multi-GPU out-of-core path (the result exceeds one GPU).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("the cuda backend requires a CUDA device")
    if device_ids:
        device = torch.device(f'cuda:{device_ids[0]}')
    if device is None:
        device = torch.device(f'cuda:{torch.cuda.current_device()}')

    x = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr)
    x = x.float()
    n, T = x.shape

    free = torch.cuda.mem_get_info(device)[0]
    ids = list(device_ids) if device_ids else None

    # Multi-GPU only when the int8 result cannot fit a single GPU. The blocks are
    # gathered on the host because the whole result exceeds one GPU anyway.
    if ids and len(ids) > 1 and n * T > free * 0.6:
        import numpy as np
        bounds = np.linspace(0, n, len(ids) + 1).astype(int)
        ranges = [(int(bounds[k]), int(bounds[k + 1])) for k in range(len(ids))
                  if bounds[k + 1] > bounds[k]]
        out = torch.empty((n, T), dtype=torch.int8)          # host
        errors = []
        lock = threading.Lock()

        def _worker(dev, lo, hi):
            try:
                _binarize_chunked(x[lo:hi], method, q, dev, out[lo:hi])
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append((dev, e))

        threads = [threading.Thread(target=_worker, args=(ids[k], lo, hi))
                   for k, (lo, hi) in enumerate(ranges)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise RuntimeError(f"binarize_cuda multi-GPU errors: {errors}")
        return out

    # Single-GPU: result on `device`, adaptive input row-chunking, no host copy.
    out = torch.empty((n, T), dtype=torch.int8, device=device)
    _binarize_chunked(x, method, q, device.index if device.index is not None else 0, out)
    return out
