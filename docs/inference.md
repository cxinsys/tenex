# Link Inference

The pairwise TE matrix assigns a directed regulatory strength to every ordered
gene pair, but most of these values reflect noise rather than genuine
regulation. Reconstructing a GRN therefore means keeping only the pairs whose TE
is too large to have arisen by chance. `NetWeaver` provides two strategies for
this decision.

## Significance thresholding with FDR

The default workflow standardizes the TE values across all candidate pairs,
converts the resulting `z`-scores to one-sided `p`-values under a Gaussian null,
and keeps edges at a target false discovery rate (FDR) with the
Benjamini-Hochberg procedure. Because the transform is monotone in TE, this is
equivalent to a single data-driven TE cutoff expressed as a controlled error
rate.

```python
nw = tnx.NetWeaver(result, fdr=0.01, is_trimming=True)
grn, trimmed = nw.infer(method="fdr")
```

### Indirect-edge trimming (DPI)

A regulatory chain such as `i -> k -> j` can make gene `i` appear to act directly
on gene `j` even when it does not. With `is_trimming=True`, TENEX removes these
indirect artifacts by a transitive reduction based on the data processing
inequality. A direct edge `i -> j` is pruned when a stronger mediating path
`i -> k -> j` exists.

## Inspecting and exporting the network

A `GRN` holds the retained directed edges. Inspect it, rank the edges, or write
it to a file:

```python
print(f"{len(grn)} edges (after trimming: {len(trimmed)})")

# Edges as tuples, in the order the GRN stores them.
edges = grn.to_edge_list()              # [(source, target, score), ...]
for source, target, score in edges[:10]:
    print(f"{source} -> {target}  {score:.4f}")

# Write a Cytoscape-style SIF table (source  TE  target).
import numpy as np
np.savetxt("grn.sif", grn.to_sif(), fmt="%s", delimiter="\t")
```

## Surrogate-based statistical test

As an alternative to the global FDR threshold, the surrogate test asks a
pair-specific question. For each candidate edge it compares the observed TE
against an empirical null obtained after the temporal relationship between the
two genes has been deliberately broken by shuffling along the time axis. Two
schemes are available, a block shuffle that retains short-range autocorrelation
and a random shuffle that applies a full permutation to each gene.

```python
sur = nw.infer(method="surrogate_test", n_surrogates=100)
sur.effective_te   # observed TE minus the mean surrogate TE (bias-corrected)
sur.p_values       # per-pair p-values
sur.grn            # significant edges (BH-FDR, positive effective TE)
```

The histogram TE estimator is slightly positive even with no directed
relationship, because a finite number of cells produces random co-occurrences.
The **effective TE** subtracts the mean surrogate TE to remove this
finite-sample bias. A directed edge is retained only when its BH-adjusted
`p`-value is below the target FDR and its effective TE is positive.

## Other matrix-based methods

`NetWeaver` also exposes CLR and Network Deconvolution, which operate directly on
the TE matrix:

```python
grn = nw.infer(method="clr")
grn = nw.infer(method="nd")
```

## Restricting to transcription factors

By default TENEX computes TE for every ordered gene pair. To restrict the
computation to a known set of regulators, pass them as `sources` to
`TransferEntropyEngine`. TE is then computed only **from** those genes, which is
faster and focuses the network on candidate regulators. `load_scrna` also
accepts `sources`, but only records the list on the returned `ScRnaData`; the
engine is what applies the filter, so pass `sources` there as well.

```python
tfs = ["GATA1", "TAL1", "KLF1"]

scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
    sources=tfs,
)
engine = tnx.TransferEntropyEngine(
    data=scrna.data, variable_names=scrna.gene_names, sources=tfs,
)
result = engine.compute()
```

The GEMM-B2 kernel computes the full `n x n` matrix and is therefore not used
when a source filter is active.

## Key-driver analysis (TRACE)

TRACE ranks genes as global regulators or targets using marginal transfer
entropy, summarizing each gene by its total outgoing (OutTE) and incoming
(InTE) information flow. It descends from TENET and uses the OutTE/InTE
formulation of Julian Lee, 2025.

```python
trace = nw.infer(method="trace", n_surrogates=100, significance=2.0)
trace.top_drivers(10)     # [(gene, OutTE), ...]  strongest regulators
trace.top_receivers(10)   # [(gene, InTE), ...]   most-regulated genes
trace.outte               # (n,) outgoing TE per gene
trace.inte                # (n,) incoming TE per gene
```

TRACE consumes the discretized bins carried by the `TransferEntropyResult`, so
run it on a result produced with the default single-lag binning.

## POINT (reserved)

`method="point"` is registered and appears in `available_methods()`, but it is a
reserved placeholder for a paper-exact key-driver procedure and currently raises
`NotImplementedError`.
