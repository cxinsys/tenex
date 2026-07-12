"""
Dataset registry for the TENEX benchmarks.

Each entry points at three files:

    expression  an (n_genes x n_cells) matrix, as a .csv path or a .npy path
    pseudotime  the pseudotime ordering of the cells
    branch      branch / cell-selection labels (which cells to keep)

For a .npy expression you must also give `gene_names` (a .npy or .txt of names),
because names cannot be read from a bare array.

The four entries below are the datasets used in the TENEX paper. The sizes are
listed for reference only — set the paths to wherever you downloaded each
dataset, or skip this file entirely and pass --expression/--pseudotime/--branch
directly on the command line.
"""

import os.path as osp

# Where the datasets live. This is a plain configuration variable: edit it here.
# By default it points at the data bundled with the repository (``<repo>/data``),
# which ships the small mESC dataset so the benchmarks and tutorials run out of
# the box. The larger datasets (skin, zebrafish, CeNGEN) are too big to bundle.
# Download them, place each under ``<DATA_ROOT>/<name>/`` following the file
# names below, and set DATA_ROOT to that directory (or edit the per-entry paths
# in DATASETS). See ``data/README.md``.
DATA_ROOT = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "data")

# To use a different location, replace the line above with an explicit path, e.g.:
#   DATA_ROOT = "/path/to/your/scdatasets"


def _p(*parts):
    return osp.join(DATA_ROOT, *parts)


DATASETS = {
    "mesc": {
        "expression": _p("mesc", "expression.csv"),
        "pseudotime": _p("mesc", "pseudotime.txt"),
        "branch":     _p("mesc", "branch.txt"),
        "description": "mESC, mouse embryonic stem cells (~3,281 genes x 459 cells)",
    },
    "skin": {
        "expression": _p("skin", "expression.csv"),
        "pseudotime": _p("skin", "pseudotime.txt"),
        "branch":     _p("skin", "branch.txt"),
        "description": "skin squamous cell carcinoma (~1,960 genes x 7,490 cells)",
    },
    "zebrafish": {
        "expression": _p("zebrafish", "expression.npy"),
        "gene_names": _p("zebrafish", "gene_names.npy"),
        "pseudotime": _p("zebrafish", "pseudotime.txt"),
        "branch":     _p("zebrafish", "branch.txt"),
        "description": "zebrafish embryogenesis (~25,258 genes x 26,022 cells)",
    },
    "cengen": {
        "expression": _p("cengen", "expression.npy"),
        "gene_names": _p("cengen", "gene_names.npy"),
        "pseudotime": _p("cengen", "pseudotime.txt"),
        "branch":     _p("cengen", "branch.txt"),
        "description": "CeNGEN, C. elegans neuronal atlas (~22,469 genes x 100,955 cells)",
    },
}
