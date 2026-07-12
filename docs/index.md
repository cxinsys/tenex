# TENEX

**TENEX** (TENET eXtremely optimized) computes pairwise transfer entropy (TE) on
the GPU to infer gene regulatory networks (GRNs) from single-cell RNA-seq data.
It is a high-performance reimplementation of
[FastTENET](https://github.com/cxinsys/fasttenet), achieving up to a
**2,203x speedup** while preserving numerical accuracy.

## Why TENEX

Transfer entropy quantifies directed information flow between genes along
pseudotime. Estimating it for every ordered gene pair requires counting a
3-D joint histogram over discretized expression values, and this counting is the
bottleneck. TENEX accelerates it by

- packing each histogram triplet into a single integer address and counting
  directly on the GPU, replacing the sort-based counting of FastTENET, and
- selecting, per dataset and device, the fastest of several CUDA kernels.

## Performance

Matched single-GPU comparison on NVIDIA PRO 6000 Blackwell (median of 3 runs).

| Dataset | Genes | FastTENET | TENEX | Speedup |
|---------|------:|----------:|------:|--------:|
| mESC | 3,281 | 33.28 s | 0.334 s | **100x** |
| Skin | 1,960 | 115.76 s | 0.216 s | **536x** |
| Zebrafish | 25,258 | 18.06 h | 42.92 s | **1,515x** |
| CeNGEN | 22,469 | 52.99 h | 86.60 s | **2,203x** |

Across all datasets TENEX reproduces FastTENET TE values within float32
precision when no bin coarsening is applied, with Pearson correlation
`r >= 0.9999`.

## Features

- **Adaptive kernel selection**: automatic selection of the best CUDA kernel for
  the data and hardware characteristics.
- **Multi-GPU support**: thread-based parallelism with no spawn overhead.
- **GPU-native preprocessing**: discretization and bin remapping entirely on GPU.
- **Multiple inference methods**: FDR, CLR, Network Deconvolution, and a
  surrogate-based statistical test.

## Where to next

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Supported Kernels](kernels.md)
- [Link Inference](inference.md)
- [API Reference](api.md)

## License

TENEX is released under the **TENEX Non-Commercial License**. Use is permitted
for non-commercial purposes only, and the license must be reproduced in copies
and derivative works. For commercial licensing, contact Daewon Lee
(dwlee@cau.ac.kr).
