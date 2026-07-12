"""
CLR — Context Likelihood of Relatedness.

Reference: Faith et al., PLoS Computational Biology, 2007.

For each entry M[i,j], compute z-scores along both the row (i) and
the column (j), clamp negatives to zero, then combine:
    clr[i,j] = sqrt( max(z_row, 0)^2 + max(z_col, 0)^2 )
"""

import numpy as np
import torch

from tenex.inference import (
    InferenceMethod, GRN, build_pairs, make_grn, threshold_pairs,
)
from tenex._log import vprint


class CLRMethod(InferenceMethod):

    @property
    def name(self) -> str:
        return 'clr'

    def infer(self, te_matrix, variable_names, device, **kwargs):
        fdr = kwargs.get('fdr', 0.01)
        links = kwargs.get('links', 0)
        sources = kwargs.get('sources', None)

        M = torch.from_numpy(np.asarray(te_matrix, dtype=np.float32)).to(device)

        # Row z-scores → square in-place
        row_mean = M.mean(dim=1, keepdim=True)
        row_std = M.std(dim=1, keepdim=True).clamp(min=1e-12)
        z2 = ((M - row_mean) / row_std).clamp(min=0).square_()

        # Column z-scores → square and add
        col_mean = M.mean(dim=0, keepdim=True)
        col_std = M.std(dim=0, keepdim=True).clamp(min=1e-12)
        z2 += ((M - col_mean) / col_std).clamp(min=0).square_()

        clr = z2.sqrt_()
        clr.fill_diagonal_(0)

        # Threshold — move to CPU for large n to avoid OOM on pair indexing
        clr_np = clr.cpu().numpy()
        del clr
        pairs_np = build_pairs(variable_names, sources)
        scores = clr_np[pairs_np[:, 0], pairs_np[:, 1]]

        te_cut, pairs_cut = threshold_pairs(
            scores, pairs_np, torch.device('cpu'), fdr=fdr, links=links)
        grn = make_grn(variable_names, pairs_cut, te_cut)
        vprint(f"[TENEX] inference(clr): {len(grn)} links")
        return grn
