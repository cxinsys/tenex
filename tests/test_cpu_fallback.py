"""CPU-only smoke tests — runs on GitHub Actions (no GPU)."""
import numpy as np
import pytest
import torch


def test_import():
    from tenex import TransferEntropyEngine, __version__
    assert __version__ == "0.1.0"


def test_kernel_registry():
    from tenex.kernels import registered_kernels
    names = [k.name for k in registered_kernels()]
    assert "scatter_add" in names
    assert "Full-SMEM" in names


def test_scatter_add_cpu():
    """ScatterAdd kernel works on CPU with synthetic data."""
    from tenex import TransferEntropyEngine

    rng = np.random.RandomState(42)
    n_genes, n_time = 20, 100
    data = rng.randn(n_genes, n_time).astype(np.float32)
    node_name = np.array([f"gene_{i}" for i in range(n_genes)])

    w = TransferEntropyEngine(data=data, variable_names=node_name)
    result = w.compute(accelerator="cpu", binning_method='FSBW-L', kp=0.5, tau=1)

    assert result.shape == (n_genes, n_genes)
    assert np.all(np.diag(result) == 0)  # diagonal = 0
    assert not np.any(np.isnan(result))  # no NaNs


def test_clear_cache():
    from tenex import TransferEntropyEngine

    rng = np.random.RandomState(42)
    n_genes, n_time = 10, 50
    data = rng.randn(n_genes, n_time).astype(np.float32)
    node_name = np.array([f"g{i}" for i in range(n_genes)])

    w = TransferEntropyEngine(data=data, variable_names=node_name)
    w.compute(accelerator="cpu", binning_method='FSBW-L', kp=0.5, tau=1)
    w.clear_cache()
    assert len(w._bin_arrs_cache) == 0
