# API Reference

The public API is exposed at the top level of the `tenex` package. Signatures
and defaults below match the implementation.

## Data loading

### `tenex.load_scrna(expression, pseudotime, branch, gene_names=None, branch_id=1, sources=None, make_binary=False)`

Load scRNA-seq data and return an aligned `ScRnaData`. Cells are filtered by the
selected branch and ordered along pseudotime.

| Argument | Description |
|----------|-------------|
| `expression` | path to the expression matrix (CSV) or a NumPy array (`n_genes x n_cells`) |
| `pseudotime` | path to the pseudotime vector, or the array itself |
| `branch` | path to the branch / cell-selection labels, or the array itself |
| `gene_names` | gene names; **required** when `expression` is a NumPy array |
| `branch_id` | branch to keep (default `1`) |
| `sources` | optional source genes (transcription factors), as a list or a path to a names file. This is recorded on the returned `ScRnaData`; to actually restrict the TE computation, pass the same `sources` to `TransferEntropyEngine` (see [Link Inference](inference.md#restricting-to-transcription-factors)) |
| `make_binary` | binarize expression before discretization (default `False`) |

Returns an `ScRnaData` with `.data` (`n_genes x n_cells`), `.gene_names`, and
`.sources`.

## Transfer entropy

### `tenex.TransferEntropyEngine(data, variable_names, sources=None)`

Orchestrates discretization, kernel selection, and the pairwise TE computation.

**`compute(accelerator="auto", devices="auto", binning_method="FSBW-L", kp=0.5, tau=1, batch_size=None, autotune=False, kernel=None, coarsening=None, use_numpy_bins=None, profile=False)`**

Computes the `n x n` TE matrix and returns a `TransferEntropyResult`.

| Argument | Description |
|----------|-------------|
| `accelerator` | `"auto"`, `"gpu"`, or `"cpu"` (default `"auto"`) |
| `devices` | a list of GPU indices like `[0, 2]`, an integer count (the first N GPUs), or `"auto"`/`-1` for all GPUs |
| `tau` | time lag (default `1`) |
| `kernel` | Forces a specific kernel by name. Valid names are `"GEMM-B2"`, `"Full-SMEM"`, `"Adaptive-SMEM"`, and `"scatter_add"` (case-insensitive). The default `None` lets TENEX auto-select the best kernel for the data and device. |
| `coarsening` | `True`/`False`/`None` to force or disable bin coarsening (default `None`, automatic) |
| `batch_size`, `autotune` | tuning knobs for the per-pair kernels |
| `use_numpy_bins` | `None` follows the device (GPU binning on CUDA, NumPy binning on CPU); `True`/`False` overrides |
| `profile` | when `True`, fills `result.timings` with per-phase durations (GPU runs; on the CPU path `result.timings` is `None`) |

### `tenex.TransferEntropyResult`

Holds the computed matrix and the metadata that downstream inference consumes.

- `.matrix` — `(n_genes, n_genes)` float32 array, `matrix[i, j] = TE(i -> j)`.
- `.variable_names`, `.tau`, `.kernel`, `.b_max`, `.timings`.
- `.bin_arrs`, `.n_per_var` — discretized bins reused by the surrogate test and TRACE.
- NumPy-compatible: `.shape`, `.dtype`, indexing, and `np.asarray(result)`.

## Link inference

### `tenex.NetWeaver(result, sources=None, fdr=0.01, links=0, is_trimming=True, trim_threshold=0.0)`

Infers a directed GRN from a `TransferEntropyResult`.

**`infer(method="fdr", device=None, **kwargs)`** (`device=None` auto-detects
`cuda:0` or CPU)

- `method="fdr"` — z-score / Benjamini-Hochberg FDR thresholding; returns
  `(grn, trimmed_grn)`.
- `method="surrogate_test"` — effective TE and a per-pair test against a
  surrogate null; returns a `SurrogateTestResult`. Kwargs: `n_surrogates`,
  `shuffle_method` (`"block"`/`"random"`), `block_length`, `p_method`
  (`"parametric"`/`"mc"`), `fused`, `seed`, `devices`.
- `method="trace"` — marginal key-driver inference (OutTE/InTE); returns a
  `TRACEResult`. Kwargs: `n_surrogates`, `significance`, `devices`.
- `method="clr"` / `method="nd"` — matrix-based CLR or Network Deconvolution.
- `method="point"` — reserved placeholder; currently raises `NotImplementedError`.

`tenex.available_methods()` returns `['clr', 'fdr', 'nd', 'point',
'surrogate_test', 'trace']`.

### `tenex.GRN`

A directed gene regulatory network.

- `.source`, `.target`, `.te`, `.pairs`.
- `to_sif()` -> `(n_edges, 3)` array of `[source, TE, target]`.
- `to_edge_list()` -> `[(source, target, score), ...]`.

### `tenex.SurrogateTestResult`

- `.effective_te` — observed TE minus the mean surrogate TE (bias-corrected).
- `.observed_te`, `.mean_surrogate_te`, `.std_surrogate_te`.
- `.p_values`, `.grn` — significant edges (BH-FDR, positive effective TE).
- `.n_surrogates`, `.shuffle_method`, `.block_length`, `.p_method`, `.fdr`.

### `tenex.TRACEResult`

- `.outte`, `.inte` — `(n,)` outgoing / incoming marginal TE per gene.
- `.network`, `.grn`.
- `top_drivers(k=10)` / `top_receivers(k=10)` -> `[(gene, score), ...]`.

## Pipeline

### `tenex.Pipeline(engine=None, **defaults)`

Computes the TE matrix once and reuses it across inference methods.

**`run(data=None, variable_names=None, sources=None, methods=None, method_kwargs=None)`**
returns a `PipelineResult` whose attributes mirror the requested methods (for
example `.fdr`, `.surrogate_test`) plus `.matrix` for the cached TE matrix.

The Pipeline auto-fills the extra inputs only for the matrix-based methods
(`fdr`, `clr`, `nd`) and the fused methods (`surrogate_test`, `point`). TRACE is
not dispatched through the Pipeline because it needs the per-gene bin arrays and
lag directly. Call it on its own instead:

```python
from tenex.inference.trace import TRACEMethod

result = engine.compute()
trace = TRACEMethod().infer(
    result.matrix, result.variable_names, device="cuda:0",
    bin_data=result.bin_arrs, tau=result.tau,
)
```

```python
pipe = tnx.Pipeline(engine, fdr=0.05)
pr = pipe.run(methods=["fdr", "surrogate_test"],
              method_kwargs={"surrogate_test": {"n_surrogates": 100}})
```

A `Pipeline` can also be built without an engine and fed data through `run()`:

```python
pr = (tnx.Pipeline(fdr=0.05)
        .configure(binning_method="FSBW-L", tau=1)
        .run(data=X, variable_names=names, methods=["fdr"]))
```

## Kernels

- `tenex.registered_kernels()` -> priority-ordered list of `TEKernel`. Their
  `.name` values are `GEMM-B2`, `Full-SMEM`, `Adaptive-SMEM`, `scatter_add`.
- `tenex.auto_select(b_max, on_cuda, smem_optin, smem_bytes, n_per_var, source_filter)`
  -> the highest-priority kernel whose `supports()` predicate holds.
- `tenex.get_kernel(name)` -> a registered kernel by (case-insensitive) name.
- `tenex.TEKernel` — abstract base class (see [Supported Kernels](kernels.md)).
- `tenex.__version__` — package version string.
