"""
GPU-accelerated preprocessing for TENEX.

Moves discretization (binning) and optional smoothing to GPU using PyTorch,
eliminating the CPU numpy bottleneck of the original implementation.
"""

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Data alignment (CPU, done once)
# ─────────────────────────────────────────────────────────────────────────────

def align_data(exp_data: np.ndarray,
               trajectory: np.ndarray,
               branch: np.ndarray,
               branch_id: int = 1) -> np.ndarray:
    """
    Select cells on a given branch, sort by pseudotime.

    Parameters
    ----------
    exp_data : (n_genes, n_cells) float32
    trajectory : (n_cells,) float32  pseudotime
    branch : (n_cells,) int32
    branch_id : which branch to use

    Returns
    -------
    refined : (n_genes, n_selected_cells) float32
    """
    n_cells = exp_data.shape[1]
    if len(trajectory) != n_cells or len(branch) != n_cells:
        raise ValueError(
            f"Dimension mismatch: exp_data has {n_cells} cells, "
            f"trajectory has {len(trajectory)}, branch has {len(branch)}"
        )
    mask = branch == branch_id
    if mask.sum() == 0:
        raise ValueError(f"No cells found with branch_id={branch_id}")
    selected_trj = trajectory[mask]
    order = np.argsort(selected_trj)
    return exp_data[:, mask][:, order]


# ─────────────────────────────────────────────────────────────────────────────
# Discretizers (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _gene_stats(arr: torch.Tensor):
    """Return (std, min, max) per gene. arr: (n_genes, T)."""
    stds = arr.std(dim=1, correction=1)
    stds = torch.where((stds == 0) | torch.isnan(stds), torch.ones_like(stds), stds)
    mins = arr.min(dim=1).values
    maxs = arr.max(dim=1).values
    return stds, mins, maxs


def bin_fsbw_l_numpy(arr: np.ndarray, kp: float = 0.5):
    """
    FSBW-L discretization on CPU using numpy — exact match with original FastTENET/MATE.

    numpy.std(axis=1, ddof=1) uses a different internal summation algorithm
    (pairwise / compensated) than PyTorch, producing slightly different std values
    even for identical float32 input.  This function replicates the original MATE
    ShiftDiscretizer.binning() exactly.

    Parameters
    ----------
    arr : (n_genes, T) float32 numpy array (already pseudotime-aligned)
    kp  : bin width fraction (default 0.5, same as FastTENET default)

    Returns
    -------
    bin_arr : (n_genes, T, 1) int32 numpy array
    n_bins  : (n_genes,) int32 numpy array
    """
    stds = np.std(arr, axis=1, ddof=1)           # numpy pairwise summation
    stds = np.where((stds == 0) | np.isnan(stds), 1.0, stds)
    mins = np.min(arr, axis=1)
    maxs = np.max(arr, axis=1)
    n_bins = np.ceil((maxs - mins) / stds).astype(np.int32)
    shifted_min = mins - kp * stds
    # Original MATE: np.floor((arr.T - shifted_min) / stds).T.astype(int32)
    bin_arr = np.floor((arr.T - shifted_min) / stds).T.astype(np.int32)
    return bin_arr[:, :, np.newaxis], n_bins      # (n_genes, T, 1), (n_genes,)


def bin_fsbw(arr: torch.Tensor, kp: float = 0.5):
    """
    Basic FSBW (Fixed-Size Bin Width) discretization.
    bin = floor((x - min) / (kp * std))

    Returns
    -------
    bin_arr : (n_genes, T, 1) int32
    n_bins  : (n_genes,) int32
    """
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / (kp * stds)).to(torch.int32)
    bin_arr = torch.floor(
        (arr - mins.unsqueeze(1)) / (kp * stds.unsqueeze(1))
    ).to(torch.int32)
    return bin_arr.unsqueeze(-1), n_bins


def bin_fsbw_l(arr: torch.Tensor, kp: float = 0.5):
    """
    FSBW-L: shift left by kp*std before binning.
    bin = floor((x - (min - kp*std)) / std)

    This is the default method used in FastTENET.
    Always produces non-negative bin indices.
    """
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    shifted_min = mins - kp * stds
    bin_arr = torch.floor(
        (arr - shifted_min.unsqueeze(1)) / stds.unsqueeze(1)
    ).to(torch.int32)
    return bin_arr.unsqueeze(-1), n_bins


def bin_fsbw_r(arr: torch.Tensor, kp: float = 0.5):
    """FSBW-R: shift right by kp*std."""
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    shifted_min = mins + kp * stds
    bin_arr = torch.floor(
        (arr - shifted_min.unsqueeze(1)) / stds.unsqueeze(1)
    ).to(torch.int32)
    return bin_arr.unsqueeze(-1), n_bins


def bin_fsbw_b(arr: torch.Tensor, kp: float = 0.5):
    """
    FSBW-B: three-kernel binning (left, center, right).
    Returns (n_genes, T, 3) int32.
    """
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    kernels = []
    for i in range(3):
        if i % 2 == 1:  # odd -> push
            shift = mins + ((i // 2 + i % 2) * kp * stds)
        else:            # even -> pull
            shift = mins - (i // 2 * kp * stds)
        b = torch.floor(
            (arr - shift.unsqueeze(1)) / stds.unsqueeze(1)
        ).to(torch.int32)
        kernels.append(b)
    return torch.stack(kernels, dim=-1), n_bins


def bin_fsbw_i(arr: torch.Tensor, kp: float = 0.5):
    """
    FSBW-I: Interpolation discretization.
    Inserts midpoints between consecutive time points, then bins.
    Returns (n_genes, 2*T-1, 1) float32 (NOT int32 — uses continuous bins).
    """
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    # Continuous bin values (not floored)
    bin_arr = (arr - mins.unsqueeze(1)) / stds.unsqueeze(1)
    # Midpoints between consecutive time steps
    mid_arr = (bin_arr[:, :-1] + bin_arr[:, 1:]) / 2.0
    # Interleave: [b0, m0, b1, m1, ..., b_{T-2}, m_{T-2}, b_{T-1}]
    n_genes, T = arr.shape
    inter = torch.zeros(n_genes, 2 * T - 1, dtype=torch.float32, device=arr.device)
    inter[:, 0::2] = bin_arr
    inter[:, 1::2] = mid_arr
    # Floor to int for histogram-based TE
    inter = torch.floor(inter).to(torch.int32)
    return inter.unsqueeze(-1), n_bins


def bin_fsbw_t(arr: torch.Tensor, kp: float = 0.5):
    """
    FSBW-T: Tag discretization — 3 shifted kernels with tag-encoded bin offsets.
    Each kernel's bins are offset by a coefficient to make them distinguishable
    when stacked, enabling multi-kernel TE without ambiguity.
    Returns (n_genes, T, 3) int32.
    """
    stds, mins, maxs = _gene_stats(arr)
    n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    kernels = []
    for i in range(3):
        if i % 2 == 1:  # odd -> push
            shift = mins + ((i // 2 + i % 2) * kp * stds)
        else:            # even -> pull
            shift = mins - (i // 2 * kp * stds)
        b = torch.floor(
            (arr - shift.unsqueeze(1)) / stds.unsqueeze(1)
        ).to(torch.int32)
        # Tag offset: shift bin values to avoid overlap between kernels
        bin_maxs = b.max(dim=1).values  # (n_genes,)
        coeff = (i + 1) * (10 ** torch.ceil(torch.log10(bin_maxs.float().clamp(min=1)))).to(torch.int32)
        b = b + coeff.unsqueeze(1)
        kernels.append(b)
    return torch.stack(kernels, dim=-1), n_bins


def bin_fsbn(arr: torch.Tensor, kp: float = 0.5, num_bins: int = None):
    """
    FSBN: Fixed-width binning using uniform bin edges (np.digitize equivalent).
    If num_bins is not given, uses ceil((max-min)/std) per gene.
    Returns (n_genes, T, 1) int32.
    """
    stds, mins, maxs = _gene_stats(arr)
    if num_bins is not None:
        n_bins = torch.full((arr.shape[0],), num_bins, dtype=torch.int32, device=arr.device)
    else:
        n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    n_bins = n_bins.clamp(min=2)
    # Per-gene digitize: bin_idx = floor((x - min) / (max - min) * n_bins)
    ranges = (maxs - mins).clamp(min=1e-8)
    normalized = (arr - mins.unsqueeze(1)) / ranges.unsqueeze(1)  # [0, 1]
    bin_arr = torch.floor(normalized * n_bins.unsqueeze(1).float()).to(torch.int32)
    # Clamp to [0, n_bins-1]
    bin_arr = bin_arr.clamp(min=0)
    for g in range(arr.shape[0]):
        bin_arr[g] = bin_arr[g].clamp(max=int(n_bins[g]) - 1)
    return bin_arr.unsqueeze(-1), n_bins


def bin_fsbq(arr: torch.Tensor, kp: float = 0.5, num_bins: int = None):
    """
    FSBQ: Quantile-based binning. Each gene's values are divided into
    equal-frequency bins using quantile boundaries.
    Returns (n_genes, T, 1) int32.
    """
    stds, mins, maxs = _gene_stats(arr)
    if num_bins is not None:
        n_bins = torch.full((arr.shape[0],), num_bins, dtype=torch.int32, device=arr.device)
    else:
        n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    n_bins = n_bins.clamp(min=2)

    n_genes, T = arr.shape
    bin_arr = torch.zeros(n_genes, T, dtype=torch.int32, device=arr.device)
    for g in range(n_genes):
        nb = int(n_bins[g])
        # Quantile boundaries
        quantiles = torch.linspace(0, 1, nb + 1, device=arr.device)
        boundaries = torch.quantile(arr[g].float(), quantiles)
        # searchsorted: find bin for each value
        idx = torch.searchsorted(boundaries[1:-1].contiguous(), arr[g].float().contiguous())
        bin_arr[g] = idx.to(torch.int32)
    return bin_arr.unsqueeze(-1), n_bins


def bin_kmeans(arr: torch.Tensor, kp: float = 0.5, num_bins: int = None):
    """
    K-means binning. Clusters each gene's expression values using K-means.
    Falls back to CPU numpy/sklearn for the clustering step.
    Returns (n_genes, T, 1) int32.
    """
    try:
        from sklearn.cluster import KMeans as _KMeans
    except ImportError as exc:
        raise ImportError(
            "k-means binning requires scikit-learn, which is an optional "
            "dependency. Install it with 'pip install tnx[kmeans]' or "
            "'pip install scikit-learn'."
        ) from exc

    stds, mins, maxs = _gene_stats(arr)
    if num_bins is not None:
        n_bins = torch.full((arr.shape[0],), num_bins, dtype=torch.int32, device=arr.device)
    else:
        n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    n_bins = n_bins.clamp(min=2)

    arr_np = arr.cpu().numpy()
    n_genes, T = arr.shape
    bin_arr = np.zeros((n_genes, T), dtype=np.int32)
    for g in range(n_genes):
        nb = int(n_bins[g])
        km = _KMeans(n_clusters=nb, random_state=0, n_init='auto')
        bin_arr[g] = km.fit_predict(arr_np[g].reshape(-1, 1))

    bin_t = torch.from_numpy(bin_arr).to(device=arr.device)
    return bin_t.unsqueeze(-1), n_bins


def bin_log(arr: torch.Tensor, kp: float = 0.5, num_bins: int = None):
    """
    Log-scale binning. Creates logarithmically spaced bin edges.
    Values must be positive. Returns (n_genes, T, 1) int32.
    """
    stds, mins, maxs = _gene_stats(arr)
    if num_bins is not None:
        n_bins = torch.full((arr.shape[0],), num_bins, dtype=torch.int32, device=arr.device)
    else:
        n_bins = torch.ceil((maxs - mins) / stds).to(torch.int32)
    n_bins = n_bins.clamp(min=2)

    n_genes, T = arr.shape
    # Clamp to positive for log
    arr_pos = arr.clamp(min=1e-10)
    bin_arr = torch.zeros(n_genes, T, dtype=torch.int32, device=arr.device)

    for g in range(n_genes):
        nb = int(n_bins[g])
        log_min = torch.floor(torch.log10(arr_pos[g].min()))
        log_max = torch.ceil(torch.log10(arr_pos[g].max()))
        if log_min == log_max:
            log_max = log_min + 1
        log_edges = torch.logspace(log_min.item(), log_max.item(), nb, device=arr.device)
        idx = torch.searchsorted(log_edges.contiguous(), arr_pos[g].contiguous())
        bin_arr[g] = idx.clamp(max=nb - 1).to(torch.int32)

    return bin_arr.unsqueeze(-1), n_bins


# Registry for easy lookup

BINNING_METHODS = {
    'fsbw':    bin_fsbw,
    'fsbw-l':  bin_fsbw_l,
    'fsbw-r':  bin_fsbw_r,
    'fsbw-b':  bin_fsbw_b,
    'fsbw-i':  bin_fsbw_i,
    'fsbw-t':  bin_fsbw_t,
    'fsbn':    bin_fsbn,
    'fsbq':    bin_fsbq,
    'k-means': bin_kmeans,
    'log':     bin_log,
    'default': bin_fsbw_l,
}


def _compact_dtype(bin_arr: torch.Tensor, b_max: int) -> torch.Tensor:
    """
    Auto type decision: choose the smallest integer dtype that can represent
    all bin values (0..b_max-1).  Reduces GPU memory by up to 4×.

    Bin values after remap_bins are always 0..b_max-1.
    After coarsen_bins, b_max ≤ smem_bin_limit (typically 28).
    Typical b_max: mESC≈11, skin≈23, zebrafish≈32, cengen≈55.

    Decision table:
        b_max ≤ 127   → int8   (4× savings vs int32)
        b_max ≤ 32767 → int16  (2× savings vs int32)
        otherwise        → int32  (no change)
    """
    if bin_arr.dtype == torch.int32:
        if b_max <= 127:
            return bin_arr.to(torch.int8)
        elif b_max <= 32767:
            return bin_arr.to(torch.int16)
    return bin_arr


def remap_bins_multi_gpu(bin_arr: torch.Tensor, device_ids: list) -> tuple:
    """
    Multi-GPU version of remap_bins: split genes across GPUs, remap in parallel.

    bin_arr must be on CPU. Each GPU processes a chunk of genes independently.
    Results are gathered back to CPU and concatenated.
    Produces bit-for-bit identical results to single-GPU remap_bins.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import math

    n_genes, T, K = bin_arr.shape
    n_gpus = len(device_ids)
    chunk = math.ceil(n_genes / n_gpus)

    results = [None] * n_gpus

    def _remap_chunk(rank):
        dev_id = device_ids[rank]
        torch.cuda.set_device(dev_id)
        dev = torch.device(f'cuda:{dev_id}')
        beg = rank * chunk
        end = min(beg + chunk, n_genes)
        if beg >= n_genes:
            return rank, torch.empty(0, T, K, dtype=torch.int32), \
                   torch.empty(0, dtype=torch.int32), 0

        chunk_d = bin_arr[beg:end].to(dev)
        remapped_d, npg_d, mb = remap_bins(chunk_d)
        return rank, remapped_d.cpu(), npg_d.cpu(), mb

    with ThreadPoolExecutor(max_workers=n_gpus) as pool:
        futures = {pool.submit(_remap_chunk, i): i for i in range(n_gpus)}
        for fut in as_completed(futures):
            rank, rem, npg, mb = fut.result()
            results[rank] = (rem, npg, mb)

    all_rem = [r[0] for r in results if r[0].shape[0] > 0]
    all_npg = [r[1] for r in results if r[1].shape[0] > 0]
    b_max = max(r[2] for r in results)

    return torch.cat(all_rem), torch.cat(all_npg), b_max


def remap_bins(bin_arr: torch.Tensor) -> tuple:
    """
    Remap each gene's bin indices to a dense 0..n_unique-1 range.

    The original FSBW-L formula assigns bin indices based on each gene's own
    std/min, so the *values* in bin_arr can be arbitrary non-negative integers.
    For example, binary data (make_binary=True) maps expression 0 → bin 0 and
    expression 1 → bin k_i where k_i = floor(1/std_i + 0.5).  With 1960 skin
    genes the global maximum k_i may reach 44, giving b_max=45 even though
    each gene uses only 2 distinct bins.

    This function performs a per-gene injective relabelling so that every gene's
    bin values are the consecutive integers 0, 1, ..., n_unique_i - 1.

    Mathematical note
    -----------------
    TE(Y→X) = Σ p(xt+1, xt, yt) log [ p(xt+1,xt,yt)·p(xt) / (p(xt+1,xt)·p(xt,yt)) ]

    TE depends only on the *joint distribution* p(xt+1, xt, yt), not on the
    specific integer labels used for the bins.  An injective relabelling leaves
    the distribution unchanged:  every (xt+1, xt, yt) combination that occurred
    before the remapping still occurs with exactly the same count after, just
    under a different (smaller) label.  TE values are therefore bit-for-bit
    identical before and after remapping.

    Performance impact
    ------------------
    For binary data (skin dataset):  b_max 44 → 2, b_max³ 85184 → 8.
      - Full-SMEM kernel becomes applicable  (smem: 345 KB → 0.1 KB)
      - scatter_add batch tensors shrink 10 000×
      - OOM on formula broadcast eliminated
    For continuous data: b_max may stay the same if the gene with the
    globally largest bin already uses all bins 0..b_max-1, but intermediate
    genes get tighter bounds, reducing average histogram fill fraction.

    Parameters
    ----------
    bin_arr : (n_genes, T, K) int32 tensor (on CPU or GPU)

    Returns
    -------
    remapped   : (n_genes, T, K) int32 tensor — dense bin indices
    n_per_var : (n_genes,) int32 tensor — unique bin count per gene
    b_max   : int — max unique bins across all genes
    """
    n_genes, T, K = bin_arr.shape
    flat = bin_arr.view(n_genes, T * K)       # (n_genes, T*K)
    n_per_var = torch.empty(n_genes, dtype=torch.int32, device=bin_arr.device)

    # In-place remapping: write inverse indices back into the input tensor.
    # This avoids allocating a full copy (saves ~8.5 GB for CeNGEN).
    # Safe because torch.unique computes its result before we overwrite.
    for i in range(n_genes):
        _, inv = torch.unique(flat[i], return_inverse=True)
        flat[i] = inv.to(torch.int32)
        n_per_var[i] = int(_.shape[0])

    b_max = int(n_per_var.max())
    return bin_arr, n_per_var, b_max


def coarsen_bins(bin_arr: torch.Tensor,
                 n_per_var: torch.Tensor,
                 smem_bin_limit: int) -> tuple:
    """
    Bin coarsening: reduce each gene's bin count to smem_bin_limit via uniform merging.

    For genes where n_per_var[g] > smem_bin_limit, map old bins to new bins:
        new_bin = old_bin * smem_bin_limit // n_per_var[g]

    This maps [0, ng) → [0, smem_bin_limit) with all smem_bin_limit values present
    when ng > smem_bin_limit, because ng/smem_bin_limit > 1 ensures each new bin
    receives at least one old bin.

    After coarsening, n_per_var[g] = smem_bin_limit for affected genes (conservative:
    actual unique count equals smem_bin_limit for ng > smem_bin_limit, so the
    Adaptive-SMEM kernel allocates exactly the right SMEM and all pairs are
    guaranteed to fit within smem_optin).

    Accuracy tradeoff
    -----------------
    Coarsening reduces histogram resolution for high-bin genes.
    Impact is data-dependent:
      - Datasets with b_max ≤ smem_bin_limit: no coarsening applied → no accuracy loss.
      - CeNGEN (b_max=55, smem_bin_limit=28): corr 1.000→0.997, Top-10K 100%→98.8%.
      - Binary/low-density data (b_max ≤ 2): never affected.

    Vectorised: operates on GPU or CPU tensors without a Python loop over genes.

    Parameters
    ----------
    bin_arr        : (n_genes, T, K) int32
    n_per_var     : (n_genes,) int32
    smem_bin_limit : maximum bins per gene derived from device SMEM opt-in

    Returns
    -------
    bin_arr_coarsened : (n_genes, T, K) int32
    n_per_var_new    : (n_genes,) int32  — n_per_var_new[g] ≤ smem_bin_limit for all g
    b_max_new      : int               — max(n_per_var_new)
    """
    n_genes, T, K = bin_arr.shape
    needs_coarsen = (n_per_var > smem_bin_limit)          # (n_genes,) bool

    if not needs_coarsen.any():
        return bin_arr, n_per_var, int(n_per_var.max())

    n_coarsen = int(needs_coarsen.sum().item())

    # In-place coarsening: new_bin = old_bin * limit // n_per_var[g]
    idx = needs_coarsen.nonzero(as_tuple=True)[0]

    if bin_arr.device.type == 'cuda' and bin_arr.dtype == torch.int8:
        # CUDA kernel: zero extra memory, pure in-place
        # Ensure contiguous layout (dtype conversion may produce non-contiguous)
        bin_arr = bin_arr.contiguous()
        _use_cuda_kernel = True
        try:
            from tenex._ext import te_adaptive_smem
            te_adaptive_smem.coarsen_bins_cuda(bin_arr, n_per_var, smem_bin_limit)
            torch.cuda.synchronize(bin_arr.device)
        except (ImportError, AttributeError):
            _use_cuda_kernel = False

        if not _use_cuda_kernel:
            # PyTorch fallback with int16 chunking
            bytes_per_gene = 8 * bin_arr.shape[1] * bin_arr.shape[2]
            free_mem = torch.cuda.mem_get_info(bin_arr.device)[0]
            chunk_sz = max(1, min(n_coarsen, int(free_mem * 0.3 // bytes_per_gene)))
            for s in range(0, n_coarsen, chunk_sz):
                e = min(s + chunk_sz, n_coarsen)
                ci = idx[s:e]
                ng = n_per_var[ci].to(torch.int16).view(-1, 1, 1)
                c = bin_arr[ci].to(torch.int16)
                bin_arr[ci] = ((c * smem_bin_limit) // ng).to(bin_arr.dtype)
                del c, ng
    else:
        # CPU: simple vectorized (no memory constraint)
        ng_sub = n_per_var[idx].view(-1, 1, 1)
        bin_arr[idx] = (bin_arr[idx].to(torch.int32) * smem_bin_limit // ng_sub).to(bin_arr.dtype)

    # Update n_per_var: coarsened genes now have exactly smem_bin_limit distinct values
    # (proof: for ng > smem_bin_limit, the mapping i → i*smem_bin_limit//ng hits every
    # value 0..smem_bin_limit-1 at least once since ng/smem_bin_limit > 1).
    n_per_var[needs_coarsen] = smem_bin_limit
    new_max = int(n_per_var.max())

    # Re-compact dtype: after coarsening, b_max ≤ smem_bin_limit (typically 28)
    bin_arr = _compact_dtype(bin_arr, new_max)

    return bin_arr, n_per_var, new_max


def discretize(arr: np.ndarray,
               method: str = 'fsbw-l',
               kp: float = 0.5,
               device: torch.device = None,
               use_numpy_bins: bool = True) -> tuple:
    """
    Full preprocessing pipeline: numpy → GPU tensor → binned integer array.

    Parameters
    ----------
    arr            : (n_genes, T) float32 numpy array (already aligned)
    method         : binning method name
    kp             : bin width fraction
    device         : target torch device
    use_numpy_bins : if True (default), compute bins on CPU numpy using the same
                     algorithm as original FastTENET/MATE, then transfer to GPU.
                     Eliminates the GPU vs CPU std-accumulation difference.
                     Only applies to 'fsbw-l'.
                     Set False for pure GPU pipeline (slightly faster).

    Returns
    -------
    bin_arr    : (n_genes, T, K) int32 torch.Tensor on device
    n_bins     : (n_genes,)     int32 torch.Tensor on device (FSBW formula bins)
    b_max   : int  (max unique bins per gene after remap, used for kernel selection)
    n_per_var : (n_genes,) int32 torch.Tensor on device (unique bins after remap)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if use_numpy_bins and method.lower() in ('fsbw-l', 'default'):
        # CPU numpy binning: matches original FastTENET/MATE exactly.
        # numpy.std uses pairwise summation which differs from PyTorch's Welford,
        # even for identical float32 input.  Using numpy here ensures corr→1.0.
        bin_arr_np, n_bins_np = bin_fsbw_l_numpy(arr, kp=kp)
        n_bins = torch.from_numpy(n_bins_np.copy()).to(device)

        n_genes, T_np = arr.shape
        K_np = bin_arr_np.shape[2]  # 1 for FSBW-L

        if device.type == 'cuda':
            # Memory budget: bin_arr on GPU = n_genes * T * K * 4 bytes.
            # remap_bins is now in-place, so no extra copy needed.
            # But we still need to fit the full tensor on GPU.
            total_bytes = n_genes * T_np * K_np * 4
            free_mem = torch.cuda.mem_get_info(device)[0]

            if total_bytes > free_mem * 0.9:
                # Too large for GPU all at once — chunk: remap on GPU per chunk,
                # store results on CPU, then transfer final array to GPU.
                # Per chunk: input + unique temporaries ≈ 3 * chunk_genes * T * K * 4
                bytes_per_gene = 3 * T_np * K_np * 4
                chunk_size = max(1, int(free_mem // bytes_per_gene))
                npg_list = []
                bin_arr_cpu = torch.from_numpy(bin_arr_np.copy()).clamp(min=0)
                for i in range(0, n_genes, chunk_size):
                    end = min(i + chunk_size, n_genes)
                    chunk_gpu = bin_arr_cpu[i:end].to(device)
                    chunk_gpu, npg_chunk, _ = remap_bins(chunk_gpu)
                    bin_arr_cpu[i:end] = chunk_gpu.cpu()
                    npg_list.append(npg_chunk.cpu())
                    del chunk_gpu
                    torch.cuda.empty_cache()
                n_per_var = torch.cat(npg_list).to(device)
                b_max = int(n_per_var.max())
                bin_arr = _compact_dtype(bin_arr_cpu, b_max).to(device)
                return bin_arr, n_bins, b_max, n_per_var

        bin_arr = torch.from_numpy(bin_arr_np.copy()).clamp(min=0).to(device)
    else:
        fn = BINNING_METHODS.get(method.lower())
        if fn is None:
            raise ValueError(f"Unknown binning method '{method}'. "
                             f"Choose from {list(BINNING_METHODS)}")

        # Chunked GPU binning: gene-level independence allows arbitrary chunking.
        # Peak memory per gene: input(T*4) + 3 intermediates(T*4 each) = 16*T bytes
        n_genes, T = arr.shape
        bytes_per_gene = 16 * T    # peak GPU bytes per gene during binning
        if device.type == 'cuda':
            free_mem = torch.cuda.mem_get_info(device)[0]
            max_genes = max(1, int(free_mem // bytes_per_gene))
            if max_genes < n_genes:
                # Must chunk: not enough GPU memory for all genes at once
                chunk_size = max_genes
                bin_chunks = []
                nbins_chunks = []
                for i in range(0, n_genes, chunk_size):
                    chunk = torch.from_numpy(arr[i:i+chunk_size]).to(
                        dtype=torch.float32, device=device)
                    b, nb = fn(chunk, kp=kp)
                    bin_chunks.append(b.cpu())
                    nbins_chunks.append(nb.cpu())
                    del chunk, b, nb
                    torch.cuda.empty_cache()
                bin_arr_cpu = torch.cat(bin_chunks, dim=0).clamp(min=0)
                n_bins = torch.cat(nbins_chunks, dim=0).to(device)

                # Chunked remap: same memory-aware approach as numpy path
                n_genes_total = bin_arr_cpu.shape[0]
                T_r, K_r = bin_arr_cpu.shape[1], bin_arr_cpu.shape[2]
                remap_bytes = 3 * T_r * K_r * 4  # per-gene remap memory
                free_r = torch.cuda.mem_get_info(device)[0]
                remap_chunk = max(1, int(free_r // remap_bytes))
                if remap_chunk < n_genes_total:
                    npg_list = []
                    for ri in range(0, n_genes_total, remap_chunk):
                        re = min(ri + remap_chunk, n_genes_total)
                        cg = bin_arr_cpu[ri:re].to(device)
                        cg, npg_c, _ = remap_bins(cg)
                        bin_arr_cpu[ri:re] = cg.cpu()
                        npg_list.append(npg_c.cpu())
                        del cg
                        torch.cuda.empty_cache()
                    n_per_var = torch.cat(npg_list).to(device)
                    b_max = int(n_per_var.max())
                    bin_arr = _compact_dtype(bin_arr_cpu, b_max).to(device)
                else:
                    bin_arr = bin_arr_cpu.to(device)
                    bin_arr, n_per_var, b_max = remap_bins(bin_arr)
                    bin_arr = _compact_dtype(bin_arr, b_max)
                return bin_arr, n_bins, b_max, n_per_var

        arr_t = torch.from_numpy(arr).to(dtype=torch.float32, device=device)
        bin_arr, n_bins = fn(arr_t, kp=kp)
        # Clamp negative values to 0 (can appear in edge cases with kp > 0.5)
        bin_arr = bin_arr.clamp(min=0)

    # Per-gene dense remapping: eliminate sparse histograms.
    # For binary data this reduces b_max from ~44 to 2 (8000× fewer histogram cells).
    # TE values are mathematically identical — only bin labels change.
    bin_arr, n_per_var, b_max = remap_bins(bin_arr)

    bin_arr = _compact_dtype(bin_arr, b_max)
    return bin_arr, n_bins, b_max, n_per_var


def discretize_multi_gpu(arr: np.ndarray,
                         method: str,
                         kp: float,
                         device_ids: list,
                         use_numpy_bins: bool = True) -> tuple:
    """
    Multi-GPU discretization: CPU binning + parallel remap_bins across GPUs.

    The binning step (FSBW-L) runs on CPU numpy (fast, exact match with original).
    The remap_bins step (per-gene torch.unique) is distributed across GPUs.
    Returns CPU tensors ready for per-GPU transfer in the compute phase.
    """
    # Step 1: CPU numpy binning (unchanged, per-gene independent)
    if use_numpy_bins and method.lower() in ('fsbw-l', 'default'):
        bin_arr_np, n_bins_np = bin_fsbw_l_numpy(arr, kp=kp)
        bin_arr = torch.from_numpy(bin_arr_np.copy()).clamp(min=0)
        n_bins = torch.from_numpy(n_bins_np.copy())
    else:
        arr_t = torch.from_numpy(arr).to(dtype=torch.float32)
        fn = BINNING_METHODS.get(method.lower())
        if fn is None:
            raise ValueError(f"Unknown binning method '{method}'. "
                             f"Choose from {list(BINNING_METHODS)}")
        bin_arr, n_bins = fn(arr_t, kp=kp)
        bin_arr = bin_arr.clamp(min=0)

    # Step 2: Multi-GPU remap_bins (distribute genes across GPUs)
    bin_arr, n_per_var, b_max = remap_bins_multi_gpu(bin_arr, device_ids)

    return bin_arr, n_bins, b_max, n_per_var
