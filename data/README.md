# Datasets

This directory holds the scRNA-seq datasets used by the tutorials and the
benchmarks. Dataset locations are configured in Python rather than through
environment variables. The benchmark registry lives in `benchmarks/datasets.py`,
where `DATA_ROOT` defaults to this directory. Edit `DATA_ROOT` or the paths for
each entry there to point at your own copies.

## Datasets included in this repository

The mESC dataset (`data/mesc/`) contains expression profiles of mouse embryonic
stem cells, 3,281 genes by 459 cells (Tuck et al.). It is small enough to
distribute with the repository, so the tutorials and the mESC benchmark run out
of the box. The directory contains four files.

- `expression.csv` holds the expression matrix.
- `pseudotime.txt` holds the pseudotime ordering of the cells.
- `branch.txt` holds the branch and cell selection labels.
- `tfs.txt` holds the transcription factor list, which is optional and used when
  inference is restricted to transcription factors.

Run the basic tutorial on it directly.

```bash
python tutorials/tutorial_basic.py
```

## Datasets retrieved from the original studies

The other three benchmark datasets are too large to distribute with the
repository. Download each one from its original study, place the files under
`data/<name>/` using the file names below, and they will be picked up
automatically. Alternatively, set `DATA_ROOT` in `benchmarks/datasets.py` to the
directory where you keep them.

- **skin** contains squamous cell carcinoma, 1,960 genes by 7,490 cells
  (`expression.csv`, `pseudotime.txt`, `branch.txt`).
- **zebrafish** contains a zebrafish embryogenesis map, 25,258 genes by 26,022
  cells (`expression.npy`, `gene_names.npy`, `pseudotime.txt`, `branch.txt`).
- **cengen** contains the C. elegans neuronal atlas (CeNGEN), 22,469 genes by
  100,955 cells (`expression.npy`, `gene_names.npy`, `pseudotime.txt`,
  `branch.txt`).

For a `.npy` expression matrix, a matching `gene_names.npy` (or `.txt`) is
required because gene names cannot be recovered from a bare array.

## Data sources

The benchmark datasets are public. They were assembled following the FastTENET
benchmark (Sung et al., *Bioinformatics*, 2024, doi:10.1093/bioinformatics/btae699).

- **mESC**: Tuck et al., *Life Science Alliance*, 2018
  (doi:10.26508/lsa.201800124). This dataset was first used for transfer entropy
  analysis in TENET (Kim et al., *Nucleic Acids Research*, 2021,
  doi:10.1093/nar/gkaa1014). The processed matrix is the copy distributed under
  `data/mesc/`.
- **skin**: integrated from four Gene Expression Omnibus studies. Squamous cell
  carcinoma GSE144236 (Ji et al., *Cell*, 2020, doi:10.1016/j.cell.2020.05.039),
  cutaneous T cell lymphoma GSE147944 (Gaydosik et al., *Blood*, 2020,
  doi:10.1182/blood.2019004725), Th17 inflammatory skin GSE179162 (Godsel et al.,
  *J. Clin. Invest.*, 2022, doi:10.1172/JCI144363), and the melanoma sample
  GSM5551114 of GSE143791 (Kfoury et al., *Cancer Cell*, 2021,
  doi:10.1016/j.ccell.2021.09.005). Only the melanoma sample was taken from
  GSE143791, which is otherwise a prostate cancer bone metastasis series.
- **zebrafish**: GSE112294 (Wagner et al., *Science*, 2018,
  doi:10.1126/science.aar4362).
- **cengen**: GSE136049 (Hammarlund et al., *Neuron*, 2018,
  doi:10.1016/j.neuron.2018.07.042), also available from the project portal at
  https://www.cengen.org/.

Each GEO accession resolves at
`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<ID>`.
