"""
Verify 4-way hash fusion produces bit-for-bit identical results
to the original per-job polynomial hash.

The algebraic identity:
  h(A,B) = h(A) * P^|B| + h(B)   (int64 wrapping)

allows us to compute h_yf and h_yp once, then derive all 4 hashes:
  h(Yf,Yp)      = h_yf * P^d + h_yp
  h(Yp)          = h_yp
  h(Yf,Xp,Yp)   = (h_yf * P + xp) * P^d + h_yp
  h(Xp,Yp)       = xp * P^d + h_yp
"""
import numpy as np
import pytest

HASH_P = 1000000007


def _horner_hash(values):
    """Python int64-wrapping Horner polynomial hash."""
    h = np.int64(0)
    for v in values:
        h = np.int64(h * np.int64(HASH_P) + np.int64(v))
    return h


def test_outte_4way_identity():
    """Verify OutTE 4-way hash matches independent per-job hashes."""
    rng = np.random.default_rng(42)

    for d in [1, 2, 5, 10, 50]:
        yf_vals = rng.integers(0, 28, size=d).astype(np.int64)
        yp_vals = rng.integers(0, 28, size=d).astype(np.int64)
        xp_val = np.int64(rng.integers(0, 28))

        # Original per-job hashes (Python int64 wrapping)
        h_job0 = _horner_hash(np.concatenate([yf_vals, yp_vals]))
        h_job1 = _horner_hash(yp_vals)
        h_job2 = _horner_hash(np.concatenate([yf_vals, [xp_val], yp_vals]))
        h_job3 = _horner_hash(np.concatenate([[xp_val], yp_vals]))

        # 4-way derived hashes
        h_yf = _horner_hash(yf_vals)
        h_yp = _horner_hash(yp_vals)
        P_d = np.int64(1)
        for _ in range(d):
            P_d = np.int64(P_d * np.int64(HASH_P))

        h4_0 = np.int64(h_yf * P_d + h_yp)
        h4_1 = h_yp
        h4_2 = np.int64(np.int64(h_yf * np.int64(HASH_P) + xp_val) * P_d + h_yp)
        h4_3 = np.int64(xp_val * P_d + h_yp)

        assert h4_0 == h_job0, f"d={d}: H(Yf,Yp) mismatch"
        assert h4_1 == h_job1, f"d={d}: H(Yp) mismatch"
        assert h4_2 == h_job2, f"d={d}: H(Yf,Xp,Yp) mismatch"
        assert h4_3 == h_job3, f"d={d}: H(Xp,Yp) mismatch"


def test_inte_4way_identity():
    """Verify InTE 4-way hash matches independent per-job hashes."""
    rng = np.random.default_rng(123)

    for d in [1, 2, 5, 10, 50]:
        yp_vals = rng.integers(0, 28, size=d).astype(np.int64)
        xf_val = np.int64(rng.integers(0, 28))
        xp_val = np.int64(rng.integers(0, 28))

        # Original per-job hashes
        h_job0 = _horner_hash([xf_val, xp_val])
        h_job1 = _horner_hash([xp_val])
        h_job2 = _horner_hash(np.concatenate([[xf_val, xp_val], yp_vals]))
        h_job3 = _horner_hash(np.concatenate([[xp_val], yp_vals]))

        # 4-way derived hashes
        h_yp = _horner_hash(yp_vals)
        h_xfxp = np.int64(np.int64(xf_val) * np.int64(HASH_P) + np.int64(xp_val))
        P_d = np.int64(1)
        for _ in range(d):
            P_d = np.int64(P_d * np.int64(HASH_P))

        h4_0 = h_xfxp
        h4_1 = np.int64(xp_val)
        h4_2 = np.int64(h_xfxp * P_d + h_yp)
        h4_3 = np.int64(np.int64(xp_val) * P_d + h_yp)

        assert h4_0 == h_job0, f"d={d}: H(Xf,Xp) mismatch"
        assert h4_1 == h_job1, f"d={d}: H(Xp) mismatch"
        assert h4_2 == h_job2, f"d={d}: H(Xf,Xp,Yp) mismatch"
        assert h4_3 == h_job3, f"d={d}: H(Xp,Yp) mismatch"


def test_4way_identity_many_timepoints():
    """Verify for L=1000 time points that all hashes match."""
    rng = np.random.default_rng(99)
    L = 1000
    d = 5

    data = rng.integers(0, 28, size=(d * 2 + 1, L)).astype(np.int64)
    yf_data = data[:d]      # (d, L)
    yp_data = data[d:2*d]   # (d, L)
    xp_data = data[2*d]     # (L,)

    for t in range(L):
        yf_vals = yf_data[:, t]
        yp_vals = yp_data[:, t]
        xp_val = xp_data[t]

        # Per-job
        h_job0 = _horner_hash(np.concatenate([yf_vals, yp_vals]))
        h_job3 = _horner_hash(np.concatenate([[xp_val], yp_vals]))

        # 4-way
        h_yf = _horner_hash(yf_vals)
        h_yp = _horner_hash(yp_vals)
        P_d = np.int64(1)
        for _ in range(d):
            P_d = np.int64(P_d * np.int64(HASH_P))

        h4_0 = np.int64(h_yf * P_d + h_yp)
        h4_3 = np.int64(np.int64(xp_val) * P_d + h_yp)

        assert h4_0 == h_job0, f"t={t}: H(Yf,Yp) mismatch"
        assert h4_3 == h_job3, f"t={t}: H(Xp,Yp) mismatch"


@pytest.mark.cuda
def test_cuda_4way_vs_generic():
    """End-to-end: 4-way CUDA pipeline matches generic pipeline."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tenex.inference.trace import (
        build_causal_network, TRACEMethod,
    )

    rng = np.random.default_rng(7)
    n, T = 50, 200
    bin_data = rng.integers(0, 10, size=(n, T)).astype(np.int32)

    # Fake TE matrix for causal network
    te_matrix = rng.random((n, n)).astype(np.float32) * 0.1
    for i in range(n):
        for j in rng.choice(n, size=5, replace=False):
            if i != j:
                te_matrix[i, j] = 0.5 + rng.random() * 0.5

    network = build_causal_network(te_matrix, n_surrogates=50, significance=1.5)
    outgoing = {i: [] for i in range(n)}
    for j, sources in network.items():
        for i in sources:
            outgoing[i].append(j)

    device = torch.device('cuda:0')
    method = TRACEMethod()

    # Run generic (fallback) pipeline
    import tenex.inference.trace as _pm
    mod = _pm._get_trace_sort_module()

    data_gpu = torch.from_numpy(bin_data.astype(np.int32)).to(device)
    data_flat = data_gpu.reshape(-1)

    outte_gen, inte_gen = method._compute_cuda_generic(
        mod, data_flat, outgoing, network, n, tau=1, T=T, L=T-1, device=device)

    # Run 4-way pipeline
    outte_4w, inte_4w = method._compute_cuda(
        bin_data, None, outgoing, network, n, tau=1, K=1, device=device)

    np.testing.assert_allclose(outte_4w, outte_gen, atol=1e-6,
                               err_msg="OutTE mismatch between 4-way and generic")
    np.testing.assert_allclose(inte_4w, inte_gen, atol=1e-6,
                               err_msg="InTE mismatch between 4-way and generic")
