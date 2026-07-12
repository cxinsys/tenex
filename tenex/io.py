"""
I/O utilities for loading domain-specific data for TENEX.

Each loader returns a domain-specific dataclass that the user unpacks
into TransferEntropyEngine's domain-neutral constructor.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tenex.preprocess import align_data
from tenex.utils import load_exp_data, load_time_data


@dataclass
class ScRnaData:
    """
    scRNA-seq data aligned for Transfer Entropy computation.

    Attributes
    ----------
    data       : (n_genes, T) float32 ndarray — expression matrix,
                 filtered by branch and ordered by ascending pseudotime.
    gene_names : (n_genes,) str ndarray — gene identifiers (row labels).
    sources    : (n_sources,) str ndarray or None — optional TF list;
                 when set, only TE from these variables is computed.

    Usage
    -----
    >>> scrna = load_scrna(...)
    >>> engine = TransferEntropyEngine(
    ...     data=scrna.data,
    ...     variable_names=scrna.gene_names,
    ...     sources=scrna.sources,
    ... )
    """
    data: np.ndarray
    gene_names: np.ndarray
    sources: Optional[np.ndarray] = None


def _load_str_with_npy_cache(path: str) -> np.ndarray:
    """Load a text file of strings with automatic .npy caching."""
    import os
    import os.path as osp
    _base, _ = osp.splitext(path)
    fpath_npy = _base + '.npy'
    if osp.isfile(fpath_npy):
        return np.load(fpath_npy, allow_pickle=True)
    data = np.loadtxt(path, dtype=str)
    if os.access(osp.dirname(osp.abspath(path)), os.W_OK):
        np.save(fpath_npy, data)
    return data


def load_scrna(
    expression,
    pseudotime,
    branch,
    gene_names=None,
    branch_id: int = 1,
    sources=None,
    make_binary: bool = False,
) -> ScRnaData:
    """
    Load scRNA-seq data and return an aligned ScRnaData object.

    Handles file paths and numpy arrays. Cells are filtered by branch and
    sorted by pseudotime.

    Parameters
    ----------
    expression : str or (n_genes, n_cells) ndarray
        Path to expression CSV, or pre-loaded expression matrix.
    pseudotime : str or (n_cells,) ndarray
        Path to pseudotime file, or pre-loaded pseudotime array.
    branch : str or (n_cells,) ndarray
        Path to branch-label file, or pre-loaded branch labels.
    gene_names : (n_genes,) ndarray, optional
        Gene names. Required when expression is an ndarray.
        Automatically extracted when expression is a CSV path.
    branch_id : int
        Which branch to select (default 1).
    sources : str or ndarray, optional
        Path to source variable (e.g. TF) name list, or ndarray of names.
        When set, only TE from these variables is computed.
    make_binary : bool
        Binarise expression values (> 0 -> 1).

    Returns
    -------
    ScRnaData
        Dataclass with aligned .data, .gene_names, and optional .sources.
    """
    # Load expression
    if isinstance(expression, str):
        names, exp_data = load_exp_data(expression, make_binary)
        if gene_names is None:
            gene_names = names
    else:
        exp_data = np.asarray(expression, dtype=np.float32)
        if gene_names is None:
            raise ValueError(
                "gene_names is required when expression is an ndarray"
            )
        if make_binary:
            exp_data = (exp_data > 0).astype(np.float32)

    # Load pseudotime
    if isinstance(pseudotime, str):
        pseudotime = load_time_data(pseudotime, dtype=np.float32)
    else:
        pseudotime = np.asarray(pseudotime, dtype=np.float32)

    # Load branch
    if isinstance(branch, str):
        branch = load_time_data(branch, dtype=np.int32)
    else:
        branch = np.asarray(branch, dtype=np.int32)

    # Load sources (TF list)
    if isinstance(sources, str):
        sources = _load_str_with_npy_cache(sources)
    elif sources is not None:
        sources = np.asarray(sources)

    # Align: filter by branch, sort by pseudotime
    aligned = align_data(exp_data, pseudotime, branch, branch_id=branch_id)

    return ScRnaData(
        data=aligned,
        gene_names=np.asarray(gene_names),
        sources=sources,
    )
