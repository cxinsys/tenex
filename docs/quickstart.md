# Quick Start

## Compute the transfer entropy matrix

`load_scrna` aligns the expression matrix, pseudotime, and branch labels into an
`ScRnaData` object. `TransferEntropyEngine` then computes the `n x n` pairwise TE
matrix, automatically selecting the fastest kernel for the data and the detected
GPU.

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

result = engine.compute(accelerator="gpu", devices="auto")
# result.matrix: (n_genes, n_genes) float32, result.matrix[i, j] = TE(i -> j)
```

## Infer a gene regulatory network

`NetWeaver` turns the TE matrix into a directed GRN by keeping statistically
significant edges and optionally removing indirect ones.

```python
nw = tnx.NetWeaver(
    result,
    fdr=0.01,           # target false discovery rate
    is_trimming=True,   # remove indirect (transitive) edges via the DPI
)
grn, trimmed_grn = nw.infer(method="fdr", device="cuda:0")
# grn.to_sif() -> (n_edges, 3) array of [source, TE, target]
```

## Surrogate-based statistical test

The surrogate test compares each observed TE against an empirical null built by
shuffling the time axis, then reports a bias-corrected effective TE and per-pair
`p`-values.

```python
sur = nw.infer(method="surrogate_test", n_surrogates=100)
sur.effective_te   # (n, n) bias-corrected TE
sur.p_values       # (n, n) p-values
sur.grn            # BH-FDR-thresholded edges
```

## One-line pipeline

`Pipeline` computes the TE matrix once and reuses it across several inference
methods.

```python
pipe = tnx.Pipeline(engine, fdr=0.05)

pr = pipe.run(
    methods=["fdr", "surrogate_test"],
    method_kwargs={"surrogate_test": {"n_surrogates": 100}},
)
pr.fdr             # (grn, trimmed): FDR + DPI-trimmed edges
pr.surrogate_test  # effective TE, p-values, significant edges
pr.matrix          # cached pairwise TE matrix
```
