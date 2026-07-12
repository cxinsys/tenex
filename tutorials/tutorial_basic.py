"""
TENEX basic usage tutorial.

Computes the pairwise transfer entropy (TE) matrix for an scRNA-seq dataset and
prints the strongest directed gene-gene relationships.

Usage:
    python tutorial_basic.py [dataset_dir]

With no argument it runs on the mESC dataset bundled with the repository
(data/mesc). Pass a directory that contains expression.csv, pseudotime.txt, and
branch.txt to use your own dataset.
"""

import os
import sys

import numpy as np

import tenex as tnx

# ── Load the data ────────────────────────────────────────────────────────────
# `load_scrna` filters cells by branch and orders them along pseudotime.
# Default to the bundled mESC dataset so the tutorial runs out of the box.
default_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mesc"
)
data_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir

scrna = tnx.load_scrna(
    expression=f"{data_dir}/expression.csv",
    pseudotime=f"{data_dir}/pseudotime.txt",
    branch=f"{data_dir}/branch.txt",
)

print(f"genes: {scrna.data.shape[0]:,}")
print(f"cells: {scrna.data.shape[1]:,}")

# ── Compute the transfer entropy matrix ──────────────────────────────────────
engine = tnx.TransferEntropyEngine(
    data=scrna.data,
    variable_names=scrna.gene_names,
)

result = engine.compute(
    accelerator="auto",   # "auto" | "gpu" | "cpu"
    devices="auto",       # "auto" (all GPUs), an int, or a list of indices
    tau=1,                # time lag
)

te = result.matrix        # (n_genes, n_genes), te[i, j] = TE(i -> j)
print(f"\nTE matrix shape: {te.shape}")
print(f"TE range: [{te.min():.6f}, {te.max():.6f}]")

# ── Top directed relationships ───────────────────────────────────────────────
names = scrna.gene_names
n = te.shape[0]
order = np.argsort(te.ravel())[::-1]

print("\nTop 10 transfer entropies (source -> target):")
shown = 0
for fi in order:
    i, j = divmod(int(fi), n)
    if i != j:
        print(f"  {names[i]} -> {names[j]}  TE={te[i, j]:.6f}")
        shown += 1
        if shown == 10:
            break

# ── Optional: infer a gene regulatory network ────────────────────────────────
# nw = tnx.NetWeaver(result, fdr=0.01, is_trimming=True)
# grn, trimmed = nw.infer(method="fdr")
# print(f"\nGRN edges: {len(grn)}  (after DPI trimming: {len(trimmed)})")
