"""
Network Deconvolution.

Reference: Feizi et al., Nature Biotechnology 31, 726-733, 2013.

The observed TE matrix contains both direct and indirect effects:
    G_obs ≈ G_dir + G_dir² + G_dir³ + ... = G_dir (I - G_dir)^{-1}

Inverting via eigendecomposition:
    G_dir = Q diag(λ / (1 + λ)) Q^T
"""

import numpy as np
import torch

from tenex.inference import (
    InferenceMethod, GRN, build_pairs, make_grn, threshold_pairs,
)
from tenex._log import vprint


class NDMethod(InferenceMethod):

    @property
    def name(self) -> str:
        return 'nd'

    def infer(self, te_matrix, variable_names, device, **kwargs):
        fdr = kwargs.get('fdr', 0.01)
        links = kwargs.get('links', 0)
        sources = kwargs.get('sources', None)

        M = torch.from_numpy(np.asarray(te_matrix, dtype=np.float32)).to(device)
        n = M.shape[0]

        # Directed network deconvolution: keep the asymmetric TE matrix rather than
        # symmetrizing it, so opposite directions retain their distinct scores.
        # Model: G_obs = G_dir + G_dir^2 + ... = G_dir (I - G_dir)^{-1}, hence
        # G_dir = (I + G_obs)^{-1} G_obs, which stays valid for a nonsymmetric G_obs.
        G = M.clone()
        G.fill_diagonal_(0)
        # Scale so the spectral radius stays below 1 (the largest singular value
        # bounds it for a nonsymmetric matrix), keeping the series convergent.
        scale = torch.linalg.matrix_norm(G, ord=2)
        if scale > 0:
            G = G / (scale / 0.99)
        eye = torch.eye(n, device=device, dtype=G.dtype)
        G_dir = torch.linalg.solve(eye + G, G)
        G_dir.fill_diagonal_(0)

        # Keep only the directions present in the original TE matrix.
        G_dir = G_dir.clamp(min=0)
        G_dir[M <= 0] = 0

        # Threshold — move to CPU for large n to avoid OOM on pair indexing
        G_dir_np = G_dir.cpu().numpy()
        del G_dir
        pairs_np = build_pairs(variable_names, sources)
        scores = G_dir_np[pairs_np[:, 0], pairs_np[:, 1]]

        te_cut, pairs_cut = threshold_pairs(
            scores, pairs_np, torch.device('cpu'), fdr=fdr, links=links)
        grn = make_grn(variable_names, pairs_cut, te_cut)
        vprint(f"[TENEX] inference(nd): {len(grn)} links")
        return grn
