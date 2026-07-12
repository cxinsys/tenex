<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tenex/assets/tenex_logo_darkmode.png" />
    <source media="(prefers-color-scheme: light)" srcset="tenex/assets/tenex_logo_lightmode.png" />
    <img alt="TENEX" src="tenex/assets/tenex_logo_lightmode.png" width="400" />
  </picture>
</p>

<h3 align="center">TENET eXtremely optimized</h3>

<p align="center">
  GPU-accelerated TENET algorithm for gene regulatory network inference from single-cell RNA-seq data
</p>

<p align="center">
  <a href="https://github.com/cxinsys/tenex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/cxinsys/tenex/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://pypi.org/project/tnx/"><img alt="PyPI" src="https://img.shields.io/pypi/v/tnx.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg" />
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.6%20%7C%2012.8%20%7C%2013.2-76B900.svg" />
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg" />
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Non--Commercial-green.svg" /></a>
</p>

<p align="center">
  <a href="https://cxinsys.github.io/tenex/"><b>Documentation</b></a> ·
  <a href="INSTALL.md"><b>Installation</b></a> ·
  <a href="https://github.com/cxinsys/tenex/releases"><b>Releases</b></a>
</p>

---

**TENEX** computes pairwise transfer entropy (TE) on GPU to infer gene regulatory networks (GRNs) from scRNA-seq data. It is a high-performance reimplementation of [FastTENET](https://github.com/cxinsys/fasttenet), achieving up to **2,203x speedup**.

## Features

- **Adaptive kernel selection**: automatic selection of the best CUDA kernel for the data and hardware characteristics
- **Multi-GPU support**: thread-based parallelism with no spawn overhead
- **GPU-native preprocessing**: discretization and bin remapping entirely on GPU
- **Multiple inference methods**: FDR, CLR, Network Deconvolution, TRACE key-driver analysis, and a surrogate-based statistical test

## Quick Start

```python
import tenex as tnx

# Load scRNA-seq data (aligned by pseudotime within selected branch)
scrna = tnx.load_scrna(
    expression="expression_data.csv",
    pseudotime="pseudotime.txt",
    branch="branch.txt",
)

# Compute pairwise TE matrix
engine = tnx.TransferEntropyEngine(
    data=scrna.data,
    variable_names=scrna.gene_names,
)
result = engine.compute(accelerator="gpu")
nw = tnx.NetWeaver(result, fdr=0.01)
```

Infer GRN links (FDR-based with DPI trimming):

```python
grn, trimmed = nw.infer(method="fdr")
```

Surrogate test (effective TE + per-pair Gaussian z-test against the
empirical null distribution from time-axis block shuffles):

```python
sur_result = nw.infer(method="surrogate_test", n_surrogates=100)
sur_result.effective_te        # (n, n) bias-corrected TE
sur_result.p_values            # (n, n) p-values
sur_result.grn                 # BH-FDR-thresholded edges
```

Or use the `Pipeline` for one-line end-to-end compute + multiple inferences
(TE matrix is computed once and reused across methods):

```python
pipe = tnx.Pipeline(engine, fdr=0.05)

pr = pipe.run(
    methods=["fdr", "surrogate_test"],
    method_kwargs={"surrogate_test": {"n_surrogates": 100}},
)
pr.fdr                         # (grn, trimmed): FDR + DPI-trimmed edges
pr.surrogate_test              # effective TE, p-values, significant edges
pr.matrix                      # cached pairwise TE matrix
```

## Installation

Install PyTorch first (it is a prerequisite, not a TENEX dependency), then
install TENEX. Keeping PyPI as the primary index lets numpy, scipy, and the
other runtime dependencies resolve normally, while the TENEX wheel index is
added as an extra source.

```bash
# Step 1: install PyTorch matching your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu132

# Step 2: install TENEX from the matching CUDA sub-index
pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/cu132/
```

Use the same CUDA line in both URLs. Pre-built GPU wheels are published for
Linux (`cu132`, `cu128`) and Windows (`cu128`, `cu126`); on macOS or for
CPU-only use, `pip install tnx` from PyPI installs the universal CPU wheel.

For build-from-source, JIT fallback, CPU-only mode, and troubleshooting, see [INSTALL.md](INSTALL.md).

## Currently supported kernels

The counting of the 3-D joint histogram is the bottleneck of TE computation,
so TENEX provides several kernels and `auto_select()` picks one from `b_max`
(the largest bin count of any gene after dense remapping) and the available GPU
shared memory.

| Kernel | Selected when | How it optimizes |
|--------|---------------|------------------|
| **GEMM-B2** | all genes are binary after remapping (`b_max = 2`), CUDA available, no TF-gene filter | Recasts the counting for all `n(n-1)` pairs as three matrix multiplications, run on Tensor Cores through cuBLAS, followed by one fused Triton kernel. No per-pair histogram is built. |
| **Full-SMEM** | the full 3-D joint histogram fits on chip (`b_max^3 <= 65,536` and within the shared-memory capacity) | Holds the entire joint-count histogram in shared memory, one gene pair per CUDA block, using on-chip atomics that are roughly 80x faster than global-memory atomics. |
| **Adaptive-SMEM** | `b_max` is too large for a uniform shared-memory histogram (CUDA available) | Sizes the shared-memory histogram per pair (`b_i * b_i * b_j`) instead of the global maximum, and coarsens bins for the few high-cardinality genes whose histogram would still overflow the on-chip capacity. |
| **scatter_add** | no CUDA device is available | CPU fallback that accumulates the histograms directly in host memory. |

## Performance

Matched single-GPU comparison on NVIDIA PRO 6000 Blackwell (median of 3 runs):

| Dataset | Genes | FastTENET | TENEX | Speedup |
|---------|------:|----------:|------:|--------:|
| mESC | 3,281 | 33.28 s | 0.334 s | **100x** |
| Skin | 1,960 | 115.76 s | 0.216 s | **536x** |
| Zebrafish | 25,258 | 18.06 h | 42.92 s | **1,515x** |
| CeNGEN | 22,469 | 52.99 h | 86.60 s | **2,203x** |

### Surrogate test

Wall time for `nw.infer(method="surrogate_test", n_surrogates=100)` on
NVIDIA PRO 6000 Blackwell. The dispatch auto-selects between the loop
path (GPU-accumulator) and the fused CUDA kernel based on `L = T - dt`.

#### 1-GPU dispatch

The dispatch picks the fused CUDA kernel for short series and the loop path
(GPU-side accumulator) for long ones, using the heuristic `L < 1500` (the
fused path is available only for the Full-SMEM and Adaptive-SMEM backends; the
`fused` kwarg overrides the choice). Numerical agreement between the two paths
stays within float precision: `Δ mean_surrogate_te ≤ 2e-9` on every dataset
listed below.

| Dataset | n | L | Kernel | Loop | Fused | Selected |
|---------|------:|--------:|:---------------|------------:|---------------:|:-------:|
| mESC | 3,281 |    458 | Full-SMEM     |     10.8 s |       **5.6 s** | fused |
| Skin | 1,960 |  7,489 | Full-SMEM     |  **28.5 s** |         31.9 s   |  loop |
| Zebrafish | 25,258 | 26,021 | Adaptive-SMEM | **1.2 h**   | 2.3 h         |  loop |
| CeNGEN | 22,469 | 100,954 | Adaptive-SMEM | **2.3 h**  | 5.6 h         |  loop |

#### 2-GPU scaling

The 2-GPU runs use the same auto-dispatched path on each GPU, with the
surrogate workload partitioned between devices. The resulting
accumulators are combined at the end of the run and produce results
that are bit-for-bit identical to the 1-GPU run (we verified
`|Δ mean_surrogate_te| = 0.0` and identical significant-edge counts on
every dataset).

| Dataset | 1-GPU | 2-GPU | Speedup |
|---------|------:|------:|--------:|
| mESC | 10.8 s | **5.6 s** | 1.94× |
| Skin | 28.5 s | **14.3 s** | 1.99× |
| Zebrafish | 1.2 h | **39.7 min** | 1.82× |
| CeNGEN | 2.3 h | **1.2 h** | 1.93× |

mESC and Skin numbers are direct measurements at `n_surrogates=100`.
Zebrafish and CeNGEN are extrapolated from `n_surrogates=10` runs. The
per-iteration cost is constant once the per-process setup is amortized,
so the extrapolation is tight.

## Citation

If you use TENEX in your work, cite the software using the metadata in
[CITATION.cff](CITATION.cff). A paper describing TENEX is in preparation, and
its citation will be added upon publication.

## License

TENEX is released under the **TENEX Non-Commercial License** (see
[LICENSE](LICENSE)). The source code is openly available and free to use,
modify, and redistribute for non-commercial purposes, provided the license is
reproduced in copies and derivative works. For commercial use, contact Daewon
Lee (dwlee@cau.ac.kr) for a commercial license.
