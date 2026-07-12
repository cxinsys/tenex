# Computing Transfer Entropy

This page covers the workflow most users run every day. It loads a dataset,
computes the pairwise transfer-entropy (TE) matrix, and inspects or saves it.
Every snippet is runnable as-is once you point it at your own files.

## The basic workflow

```python
import tenex as tnx

# 1. Load and align the data (filter by branch, order by pseudotime).
scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
)

# 2. Compute the n x n TE matrix. The kernel and device are chosen automatically.
engine = tnx.TransferEntropyEngine(
    data=scrna.data,
    variable_names=scrna.gene_names,
)
result = engine.compute()

# 3. Use the result.
te = result.matrix          # (n_genes, n_genes) float32, te[i, j] = TE(i -> j)
print(te.shape, result.kernel)
```

`result` is a `TransferEntropyResult`. Its `.matrix` is a NumPy-compatible array,
and it also carries the metadata that the inference step reuses.

## What `compute()` does

A single call runs the whole pipeline:

1. **Discretization and dense remapping.** Continuous expression is binned with
   the bandwidth-based FSBW-L scheme, then the used bins of each gene are
   renumbered to consecutive integers. The largest count across genes is
   `result.b_max`.
2. **Kernel selection.** TENEX picks the fastest CUDA kernel that fits the data
   and the device, or the CPU fallback when no GPU is present. The choice is
   deterministic, and the selected kernel is reported as `result.kernel`. See
   [Kernels and Performance](kernels.md).
3. **TE computation.** The kernel counts the joint histograms and evaluates the
   TE formula for every ordered gene pair.

You do not need to configure any of this for a normal run. The options below are
there when you want to control the device, the lag, or the output.

## Choosing the device

By default `accelerator="auto"` uses the GPU when one is available and falls back
to the CPU otherwise. `devices` accepts a list of GPU indices, an integer count,
or `"auto"`/`-1` for all GPUs:

```python
# A specific GPU, by index
result = engine.compute(
    accelerator="gpu",
    devices=[0],
)

# Several specific GPUs (work is split across them)
result = engine.compute(
    accelerator="gpu",
    devices=[0, 1, 2, 3],
)

# The first N GPUs (an integer is a count, not an index)
result = engine.compute(
    accelerator="gpu",
    devices=2,
)

# CPU only
result = engine.compute(accelerator="cpu")
```

For large datasets, multiple GPUs shorten the compute phase roughly linearly.
The preprocessing phases are unaffected, because they run before the work is
distributed across devices, so the speedup is largest when the compute phase
dominates (the atlas-scale datasets).

## Setting the time lag

The lag `tau` controls how far ahead the future of a gene is read. The default
of `1` matches FastTENET:

```python
result = engine.compute(tau=2)
```

## Binary data

When every gene is binary after discretization (`b_max == 2`), TENEX selects the
GEMM-B2 kernel, which recasts the counting as matrix multiplication and is the
fastest path. You can force binarization at load time:

```python
scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
    make_binary=True,
)
```

## Numerical reproducibility

TENEX reproduces FastTENET TE values within float32 precision. The one knob that
affects exactness is where the discretization runs, controlled by
`use_numpy_bins`:

```python
# Default (None): follow the device.
#   On CUDA  -> GPU binning, about 5x faster, correlation ~1.0 but not bit-for-bit.
#   On CPU   -> NumPy binning, bit-for-bit identical to FastTENET.
result = engine.compute(use_numpy_bins=None)

# Force FastTENET-exact binning even on the GPU:
result = engine.compute(use_numpy_bins=True)
```

The only other source of difference is bin coarsening, which is applied
automatically to the largest datasets when a histogram would not fit in GPU
shared memory (see [Kernels and Performance](kernels.md)).

## Understanding the result

`TransferEntropyResult` is more than the matrix. The downstream inference and
the surrogate test reuse its discretized bins, so keep the object around rather
than extracting only `.matrix`:

| Attribute | Meaning |
|-----------|---------|
| `.matrix` | `(n_genes, n_genes)` float32, `matrix[i, j] = TE(i -> j)` |
| `.variable_names` | gene names aligned to the matrix |
| `.kernel`, `.b_max` | the kernel that ran and the global max bin count |
| `.tau` | the lag used |
| `.bin_arrs`, `.n_per_var` | discretized bins consumed by the surrogate test and TRACE |
| `.timings` | per-phase durations (populated when `profile=True`) |

The object is NumPy-compatible, so `result.shape`, indexing, and
`np.asarray(result)` all work directly on the TE matrix.

## Inspecting the matrix

```python
import numpy as np

te = result.matrix
names = result.variable_names
n = te.shape[0]

# The strongest directed relationships (ignoring the zero diagonal).
order = np.argsort(te.ravel())[::-1]
for fi in order[:10]:
    i, j = divmod(int(fi), n)
    if i != j:
        print(f"{names[i]} -> {names[j]}  TE={te[i, j]:.6f}")
```

## Saving the matrix

`result.matrix` is a plain array, so use any NumPy writer. The labelled-table
example below also needs pandas, which is not a TENEX dependency (install it
separately with `pip install pandas`):

```python
import numpy as np
import pandas as pd

# Compact binary form.
np.save("te_matrix.npy", result.matrix)

# Labelled table, convenient for sharing or loading elsewhere.
pd.DataFrame(
    result.matrix,
    index=result.variable_names,
    columns=result.variable_names,
).to_csv("te_matrix.csv")
```

## Restricting to transcription factors

To compute TE only **from** a known set of regulators, pass them as `sources`.
This is faster and focuses the network on candidate regulators:

```python
tfs = ["GATA1", "TAL1", "KLF1"]
scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
    sources=tfs,
)
engine = tnx.TransferEntropyEngine(
    data=scrna.data,
    variable_names=scrna.gene_names,
    sources=tfs,
)
result = engine.compute()
```

## Profiling the run

Pass `profile=True` to record per-phase timings on the result. This shows where
time goes, namely preprocessing for small datasets and the compute phase for
large ones. Timings are collected on GPU runs. On the CPU path `result.timings`
is `None`:

```python
result = engine.compute(profile=True)
for phase, seconds in result.timings.items():
    print(f"{phase:18s} {seconds:.4f} s")
```

## Selecting a kernel

You normally do not choose a kernel. `compute()` selects one automatically from
the data (the bin count `b_max`, whether it is binary, whether a source filter is
active) and the hardware (whether a GPU is present and how much shared memory it
exposes). The same dataset can therefore run on a different kernel on different
machines, which is expected behaviour, not a misconfiguration.

You can request a specific kernel by name, but the request is honoured only when
that kernel supports the current data and device. An incompatible request raises
`ValueError` rather than silently running, so this is mainly useful for
benchmarking and debugging, not everyday use:

```python
result = engine.compute(kernel="Full-SMEM")   # names are case-insensitive
```

See [Kernels and Performance](kernels.md) for the conditions under which each
kernel is selected.

## End to end: from data to a network

The common case chains the compute step into GRN inference. The TE matrix is
computed once and reused:

```python
import tenex as tnx

scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
)
engine = tnx.TransferEntropyEngine(
    data=scrna.data,
    variable_names=scrna.gene_names,
)
result = engine.compute()

# Keep only statistically significant edges, and remove indirect ones.
nw = tnx.NetWeaver(
    result,
    fdr=0.01,
    is_trimming=True,
)
grn, trimmed = nw.infer(method="fdr")
print(f"{len(grn)} edges (after trimming: {len(trimmed)})")
```

See [Inferring Gene Networks](inference.md) for the inference methods and how to
export the result.
