"""
Data loading utilities for TENEX.
Drop-in replacement for fasttenet.utils / mate.utils.
"""


import os.path as osp

import numpy as np
import torch


def load_exp_data(dpath: str, make_binary: bool = False):
    """
    Load raw expression matrix (NOT pseudotime-aligned).

    If a pre-saved binary (.npy) cache exists alongside the CSV, it is loaded
    directly for speed.  Otherwise the CSV is parsed and the cache is written.

    The returned array is **raw** (not sorted by pseudotime). Pass it to
    ``TransferEntropyEngine(exp_data=..., trajectory=..., branch=...)`` which handles
    alignment internally, or call ``align_data()`` explicitly.

    Returns
    -------
    node_name : (n_genes,) str array
    exp_data  : (n_genes, n_cells) float32 array  (raw, unaligned)
    """
    import hashlib, os
    _bin_suffix = '_binary' if make_binary else ''
    # Cache alongside CSV if writable; otherwise fall back to ~/.cache/tenex/
    _ddir = osp.dirname(osp.abspath(dpath))
    _base, _fext = osp.splitext(dpath)
    if os.access(_ddir, os.W_OK):
        # New naming: *_raw.npy to make it clear this is unaligned data
        fpath_bin  = _base + _bin_suffix + '_raw.npy'
        fpath_name = _base + _bin_suffix + '_raw_node_name.npy'
        # Fallback: read old-format cache (without _raw) for backward compat
        fpath_bin_legacy  = _base + _bin_suffix + '.npy'
        fpath_name_legacy = _base + _bin_suffix + '_node_name.npy'
    else:
        _cache_dir = osp.expanduser('~/.cache/tenex')
        os.makedirs(_cache_dir, exist_ok=True)
        _key = hashlib.md5((dpath + _bin_suffix).encode()).hexdigest()[:12]
        fpath_bin  = osp.join(_cache_dir, _key + '_raw.npy')
        fpath_name = osp.join(_cache_dir, _key + '_raw_node_name.npy')
        fpath_bin_legacy  = osp.join(_cache_dir, _key + '.npy')
        fpath_name_legacy = osp.join(_cache_dir, _key + '_node_name.npy')

    # Try new-format cache first
    if osp.isfile(fpath_bin) and osp.isfile(fpath_name):
        exp_data  = np.load(fpath_bin)
        node_name = np.load(fpath_name, allow_pickle=True)
        return node_name, exp_data

    # Fallback to legacy cache (without _raw suffix)
    if osp.isfile(fpath_bin_legacy) and osp.isfile(fpath_name_legacy):
        exp_data  = np.load(fpath_bin_legacy)
        node_name = np.load(fpath_name_legacy, allow_pickle=True)
        return node_name, exp_data

    raw       = np.loadtxt(dpath, delimiter=',', dtype=str)
    node_name = raw[0, 1:]
    exp_data  = raw[1:, 1:].T.astype(np.float32)   # (n_genes, n_cells)

    if make_binary:
        exp_data = (exp_data > 0).astype(np.float32)

    np.save(fpath_bin,  exp_data)
    np.save(fpath_name, node_name)
    return node_name, exp_data


def load_time_data(dpath: str, dtype=np.float32) -> np.ndarray:
    return _load_with_npy_cache(dpath, dtype)


def _load_with_npy_cache(dpath: str, dtype=np.float32) -> np.ndarray:
    """Load a 1-D text file with automatic .npy caching.

    The cache filename is keyed on dtype so the same source file can be
    loaded as float32 (pseudotime) and int32 (branch labels) without one
    overwriting the other or being returned with the wrong dtype.
    """
    import os
    _base, _ = osp.splitext(dpath)
    _dtype_tag = np.dtype(dtype).str.lstrip('<>=|').replace('<', '').replace('>', '')
    fpath_npy = f"{_base}_{_dtype_tag}.npy"

    if osp.isfile(fpath_npy):
        return np.load(fpath_npy)

    data = np.loadtxt(dpath, dtype=dtype)
    _ddir = osp.dirname(osp.abspath(dpath))
    if os.access(_ddir, os.W_OK):
        np.save(fpath_npy, data)
    return data


def get_device_list() -> list[int]:
    return list(range(torch.cuda.device_count()))
