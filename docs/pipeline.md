# Running a Pipeline

`Pipeline` is the convenience layer that ties the two steps together. It computes
the transfer-entropy (TE) matrix once and reuses it across several inference
methods, so you do not recompute the expensive part for every method.

Use it when you want more than one view of the same data, for example an
FDR-thresholded network and a surrogate test side by side.

## The basic workflow

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

pipe = tnx.Pipeline(engine, fdr=0.05)
pr = pipe.run(
    methods=["fdr", "surrogate_test"],
    method_kwargs={"surrogate_test": {"n_surrogates": 100}},
)
```

The TE matrix is computed on the first method and cached, so the second method
reuses it. The keyword defaults passed at construction (here `fdr=0.05`) apply to
every method unless a per-method override is given.

## Reading the result

`run()` returns a `PipelineResult`. Each requested method is available as an
attribute named after the method, and the shared TE matrix is on `.matrix`:

```python
grn, trimmed = pr.fdr           # (grn, trimmed): FDR + DPI-trimmed edges
st = pr.surrogate_test          # SurrogateTestResult: effective TE, p-values, edges
te = pr.matrix                  # the cached (n_genes, n_genes) TE matrix

print(f"{len(grn)} edges, {len(st.grn)} significant under the surrogate test")
```

`pr.get("fdr")` is the explicit form of `pr.fdr`, and `pr.te_result` is the full
`TransferEntropyResult` behind `pr.matrix`.

## Choosing the methods

`methods` accepts any registered inference method. The matrix-based methods read
only the TE matrix, while the surrogate test reuses the discretized bins that the
pipeline already holds:

```python
pr = pipe.run(methods=["fdr", "clr", "nd", "surrogate_test"])
```

Per-method parameters go in `method_kwargs`, keyed by method name:

```python
pr = pipe.run(
    methods=["fdr", "surrogate_test"],
    method_kwargs={
        "fdr": {"is_trimming": True},
        "surrogate_test": {"n_surrogates": 200, "shuffle_method": "block"},
    },
)
```

See [Inferring Gene Networks](inference.md) for what each method returns.

## Feeding data through run()

You do not have to build the engine yourself. A `Pipeline` can take the data
directly in `run()`, and `configure()` sets the keywords forwarded to the compute
step. This is the recommended form for a one-shot analysis:

```python
pr = (
    tnx.Pipeline(fdr=0.05)
    .configure(binning_method="FSBW-L", tau=1)
    .run(
        data=scrna.data,
        variable_names=scrna.gene_names,
        methods=["fdr"],
    )
)
```

Calling `run()` again with a different data array rebuilds the engine and drops
the cached matrix. Changing a `configure()` value also invalidates the cache, so
the next access recomputes with the new settings.

## When to use the engine directly

If you only need one network, the engine plus `NetWeaver` is simpler and just as
fast. Reach for `Pipeline` when you want the TE matrix reused across several
inference methods without recomputing it. The two styles share the same compute
path, so the resulting matrix is identical.
