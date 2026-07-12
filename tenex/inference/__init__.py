"""
TENEX inference — link inference algorithms from a TE matrix.

Each algorithm implements the InferenceMethod interface and is registered
in a method registry. NetWeaver dispatches to these methods.

Available methods:
  - 'fdr'             : z-score → FDR (Benjamini–Hochberg) → optional DPI trimming
  - 'clr'             : Context Likelihood of Relatedness (Faith et al., 2007)
  - 'nd'              : Network Deconvolution (Feizi et al., Nature Biotech 2013)
  - 'trace'           : TRACE — Threshold-Refined Aggregate Causal Entropy.
                        Fast marginal-TE inference with OutTE/InTE aggregation.
                        Descends from TENET (Kim 2021); borrows OutTE/InTE
                        definitions from Julian Lee 2025 but not POINT's
                        procedure.
  - 'point'           : POINT — Julian Lee 2025's exact procedure
                        (NotImplementedError for now; paper-exact CUDA
                        reimplementation planned).
  - 'surrogate_test'  : Time-axis shuffle surrogate test. Yields effective-TE
                        (bias-corrected TE value) and BH-FDR-thresholded GRN
                        from empirical null distribution.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from tenex.inference.grn import GRN


class InferenceMethod(ABC):
    """Abstract base class for link inference algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this method (e.g., 'fdr', 'clr', 'point')."""

    @abstractmethod
    def infer(
        self,
        te_matrix: np.ndarray,
        variable_names: np.ndarray,
        device: torch.device,
        **kwargs,
    ) -> GRN | tuple[GRN, GRN]:
        """
        Run link inference on a TE matrix.

        Parameters
        ----------
        te_matrix      : (n, n) float32 — pairwise TE matrix.
        variable_names : (n,) str array.
        device         : torch device.
        **kwargs       : method-specific parameters (fdr, links, etc.)

        Returns
        -------
        GRN or (GRN, trimmed_GRN)
        """


# ── Method registry ──────────────────────────────────────────────────────────

_REGISTRY: dict[str, InferenceMethod] = {}


def register(method: InferenceMethod):
    _REGISTRY[method.name] = method


def get_method(name: str) -> InferenceMethod:
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown inference method: {name!r}. Available: {available}")
    return _REGISTRY[name]


def available_methods() -> list[str]:
    return sorted(_REGISTRY.keys())


# ── Shared utilities ─────────────────────────────────────────────────────────

def build_pairs(variable_names: np.ndarray, sources: np.ndarray | list | None = None) -> np.ndarray:
    """Build directed variable pairs, optionally filtered by source list."""
    n = len(variable_names)
    if sources is not None:
        _, src_idx, _ = np.intersect1d(variable_names, sources, return_indices=True)
        all_idx = np.arange(n)
        src_rep = np.repeat(src_idx, n)
        tgt_rep = np.tile(all_idx, len(src_idx))
        mask = src_rep != tgt_rep
        pairs = np.stack((src_rep[mask], tgt_rep[mask]), axis=1)
    else:
        idx = np.arange(n)
        src = np.repeat(idx, n - 1)
        base = np.tile(np.arange(n - 1), n)
        src_expanded = np.repeat(idx, n - 1)
        tgt = base + (base >= src_expanded).astype(np.int64)
        pairs = np.stack((src, tgt), axis=1).astype(np.int32)
    return pairs


def make_grn(variable_names: np.ndarray, pairs: np.ndarray, te: np.ndarray) -> GRN:
    return GRN(
        source=variable_names[pairs[:, 0]],
        target=variable_names[pairs[:, 1]],
        te=te.astype(np.float32),
        pairs=pairs,
    )


def bh_fdr_gpu(pval: torch.Tensor, alpha: float) -> torch.Tensor:
    """Benjamini–Hochberg FDR correction (GPU-native)."""
    m = pval.shape[0]
    if m == 0:
        return pval
    sorted_pval, sort_idx = pval.sort()
    rank = torch.arange(1, m + 1, device=pval.device, dtype=pval.dtype)
    adjusted = (sorted_pval * m / rank).clamp(max=1.0)
    adjusted = adjusted.flip(0).cummin(0).values.flip(0)
    result = torch.empty_like(adjusted)
    result[sort_idx] = adjusted
    return result


def fdr_threshold_gpu(te: torch.Tensor, pairs: torch.Tensor, fdr: float):
    """z-score → p-value → BH-FDR on GPU. Returns (te_cut, pairs_cut)."""
    std = te.std()
    if std == 0:
        return torch.empty(0, device=te.device), pairs[:0]
    z = (te - te.mean()) / std
    pval = 0.5 * torch.erfc(z / 1.4142135623730951)
    fdr_pval = bh_fdr_gpu(pval, 0.05)
    mask = fdr_pval < fdr
    return te[mask], pairs[mask]


def fdr_threshold_cpu(te: np.ndarray, pairs: np.ndarray, fdr: float):
    """z-score → p-value → BH-FDR via scipy/statsmodels (CPU)."""
    import scipy.stats
    import statsmodels.stats.multitest as smm
    std = te.std()
    if std == 0:
        return te[:0], pairs[:0]
    z = (te - te.mean()) / std
    pval = 1.0 - scipy.stats.norm.cdf(z)
    _, fdr_pval, _, _ = smm.multipletests(pval, alpha=0.05, method='fdr_bh')
    mask = fdr_pval < fdr
    return te[mask], pairs[mask]


def threshold_pairs(te, pairs, device, fdr=0.01, links=0, sources=None):
    """Apply top-K or FDR thresholding. Returns (te_cut, pairs_cut) as numpy."""
    if isinstance(te, torch.Tensor):
        if links > 0:
            k = min(links, len(te))
            _, idx = te.topk(k)
            idx = idx[te[idx].argsort(descending=True)]
            return te[idx].cpu().numpy(), pairs[idx].cpu().numpy()
        else:
            if te.device.type == 'cuda':
                te_c, p_c = fdr_threshold_gpu(te, pairs, fdr)
            else:
                te_c, p_c = fdr_threshold_cpu(
                    te.cpu().numpy(), pairs.cpu().numpy(), fdr)
                return te_c, p_c
            return te_c.cpu().numpy(), p_c.cpu().numpy()
    else:
        if links > 0:
            k = min(links, len(te))
            idx = np.argpartition(te, -k)[-k:]
            idx = idx[np.argsort(te[idx])[::-1]]
            return te[idx], pairs[idx]
        else:
            return fdr_threshold_cpu(te, pairs, fdr)


# ── Auto-register all methods ────────────────────────────────────────────────

def _register_all():
    from tenex.inference.fdr import FDRMethod
    from tenex.inference.clr import CLRMethod
    from tenex.inference.nd import NDMethod
    from tenex.inference.trace import TRACEMethod
    from tenex.inference.point import POINTMethod
    from tenex.inference.surrogate_test import SurrogateTestMethod

    register(FDRMethod())
    register(CLRMethod())
    register(NDMethod())
    register(TRACEMethod())
    register(POINTMethod())  # placeholder: raises NotImplementedError on infer()
    register(SurrogateTestMethod())


_register_all()

# Re-export NetWeaver for convenience
from tenex.inference.netweaver import NetWeaver  # noqa: E402
