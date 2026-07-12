"""Device contract — an explicitly requested device must never silently fall back.

These run without a GPU (CI on Linux, macOS, and Windows). They assert that
asking for a device that is not available raises an error instead of quietly
computing on the CPU. ``accelerator='auto'`` is the only mode allowed to fall
back to the CPU, because it does not name an explicit device.
"""
import numpy as np
import pytest
import torch


def _engine(n_genes=12, n_time=80, seed=0):
    from tenex import TransferEntropyEngine

    rng = np.random.RandomState(seed)
    data = rng.randn(n_genes, n_time).astype(np.float32)
    names = np.array([f"g{i}" for i in range(n_genes)])
    return TransferEntropyEngine(data=data, variable_names=names)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CUDA-less host")
def test_explicit_gpu_without_cuda_raises():
    """accelerator='gpu' on a CUDA-less host must raise, not fall back to CPU."""
    eng = _engine()
    with pytest.raises((RuntimeError, ValueError)):
        eng.compute(accelerator="gpu", binning_method="FSBW-L", kp=0.5, tau=1)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CUDA-less host")
def test_explicit_gpu_device_index_without_cuda_raises():
    """Naming a device index still requires CUDA; it must raise when absent."""
    eng = _engine()
    with pytest.raises((RuntimeError, ValueError)):
        eng.compute(accelerator="gpu", devices=[0],
                    binning_method="FSBW-L", kp=0.5, tau=1)


def test_explicit_cpu_always_works():
    """accelerator='cpu' is supported on every platform."""
    eng = _engine()
    result = eng.compute(accelerator="cpu", binning_method="FSBW-L", kp=0.5, tau=1)
    assert result.shape == (12, 12)
    assert np.all(np.diag(result) == 0)
    assert not np.any(np.isnan(result))


def test_auto_may_use_cpu():
    """accelerator='auto' does not name a device, so a CPU result is allowed."""
    eng = _engine()
    result = eng.compute(accelerator="auto", binning_method="FSBW-L", kp=0.5, tau=1)
    assert result.shape == (12, 12)
    assert not np.any(np.isnan(result))


def test_unknown_accelerator_raises():
    """An unrecognized accelerator name is an error, never a silent fallback."""
    eng = _engine()
    with pytest.raises((ValueError, RuntimeError)):
        eng.compute(accelerator="npu", binning_method="FSBW-L", kp=0.5, tau=1)
