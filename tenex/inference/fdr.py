"""
FDR inference — z-score → Benjamini–Hochberg FDR → optional DPI trimming.

Original approach used in TENET (Kim et al., 2021) and FastTENET (Sung et al., 2024).
"""

import numpy as np
import torch

from tenex.inference import (
    InferenceMethod, GRN, build_pairs, make_grn, threshold_pairs,
)
from tenex._log import vprint


class FDRMethod(InferenceMethod):

    @property
    def name(self) -> str:
        return 'fdr'

    def infer(self, te_matrix, variable_names, device, **kwargs):
        fdr = kwargs.get('fdr', 0.01)
        links = kwargs.get('links', 0)
        sources = kwargs.get('sources', None)
        is_trimming = kwargs.get('is_trimming', True)
        trim_threshold = kwargs.get('trim_threshold', 0.0)
        batch_size = kwargs.get('batch_size', 512)

        pairs_np = build_pairs(variable_names, sources)
        n_pairs = len(pairs_np)
        te_matrix = np.asarray(te_matrix, dtype=np.float32)

        # For large datasets, GPU thresholding may OOM (pairs tensor too large).
        # Use GPU for small pair counts, CPU for large ones.
        gpu_pair_limit = 50_000_000  # ~50M pairs fits in 24GB VRAM
        use_gpu = (device.type == 'cuda' and n_pairs <= gpu_pair_limit)

        if use_gpu:
            pairs_t = torch.from_numpy(pairs_np).to(device, dtype=torch.long)
            M = torch.from_numpy(te_matrix).to(device)
            te_t = M[pairs_t[:, 0], pairs_t[:, 1]]
            te_cut, pairs_cut = threshold_pairs(te_t, pairs_t, device, fdr=fdr, links=links)
        else:
            te_flat = te_matrix[pairs_np[:, 0], pairs_np[:, 1]]
            te_cut, pairs_cut = threshold_pairs(
                te_flat, pairs_np, torch.device('cpu'), fdr=fdr, links=links)
        grn = make_grn(variable_names, pairs_cut, te_cut)
        vprint(f"[TENEX] inference(fdr): {len(grn)} links after thresholding")

        if not is_trimming:
            return grn

        trimmed_grn = _trim(te_matrix, variable_names, pairs_cut, te_cut,
                            device=device, batch_size=batch_size,
                            trim_threshold=trim_threshold)
        vprint(f"[TENEX] inference(fdr): {len(trimmed_grn)} links after trimming")
        return grn, trimmed_grn


def _trim(te_matrix, variable_names, pairs, te_values, device, batch_size, trim_threshold):
    """DPI trimming — remove indirect edges (GPU-accelerated, sparsity-aware)."""
    n = te_matrix.shape[0]
    arr_np = np.zeros((n, n), dtype=np.float32)
    arr_np[pairs[:, 0], pairs[:, 1]] = te_values.astype(np.float32)
    arr = torch.from_numpy(arr_np).to(device)

    has_outgoing = (arr.sum(dim=1) > 0)
    mediator_set = has_outgoing.nonzero(as_tuple=True)[0]
    n_mediators = len(mediator_set)

    if n_mediators == 0:
        return make_grn(variable_names, pairs, te_values)

    if device.type == 'cuda':
        free_mem, _ = torch.cuda.mem_get_info(device)
        mem_budget = int(free_mem * 0.4)
    else:
        mem_budget = 2 * 1024**3

    trimmed_pairs = []
    trimmed_tes = []

    for row_start in range(0, n, batch_size):
        row_end = min(row_start + batch_size, n)
        batch = arr[row_start:row_end]

        if not (batch != 0).any():
            continue

        batch_nonzero_cols = batch.nonzero(as_tuple=True)[1].unique()
        active_mask = torch.isin(batch_nonzero_cols, mediator_set)
        active_mediators = batch_nonzero_cols[active_mask]
        n_active = len(active_mediators)

        if n_active == 0:
            rows, cols = batch.nonzero(as_tuple=True)
            if len(rows) > 0:
                trimmed_pairs.append(
                    torch.stack((rows + row_start, cols), dim=1).cpu().numpy())
                trimmed_tes.append(batch[rows, cols].cpu().numpy())
            continue

        bs = row_end - row_start
        kc = max(1, mem_budget // (bs * n * 4))
        kc = min(kc, n_active)

        indirect = torch.zeros_like(batch)

        for k_start in range(0, n_active, kc):
            k_end = min(k_start + kc, n_active)
            k_idx = active_mediators[k_start:k_end]
            paths = torch.min(batch[:, k_idx].unsqueeze(2), arr[k_idx].unsqueeze(0))
            indirect = torch.max(indirect, paths.max(dim=1).values)

        if trim_threshold != 0.0:
            indirect += trim_threshold

        trimmed = torch.where(batch >= indirect, batch, torch.zeros_like(batch))
        rows, cols = torch.nonzero(trimmed, as_tuple=True)
        if len(rows) == 0:
            continue
        trimmed_pairs.append(
            torch.stack((rows + row_start, cols), dim=1).cpu().numpy())
        trimmed_tes.append(trimmed[rows, cols].cpu().numpy())

    if not trimmed_pairs:
        return make_grn(variable_names, np.empty((0, 2), dtype=np.int32),
                        np.empty(0, dtype=np.float32))

    all_pairs = np.concatenate(trimmed_pairs, axis=0)
    all_tes = np.concatenate(trimmed_tes, axis=0)
    return make_grn(variable_names, all_pairs, all_tes)
