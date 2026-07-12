# Supported Kernels

Counting the 3-D joint histogram is the bottleneck of TE computation, so TENEX
provides several CUDA kernels and `auto_select()` picks one from `b_max` (the
largest bin count of any gene after dense remapping) and the available GPU shared
memory.

| Kernel | Selected when | How it optimizes |
|--------|---------------|------------------|
| **GEMM-B2** | all genes are binary after remapping (`b_max = 2`), CUDA available, no TF-gene filter | Recasts the counting for all `n(n-1)` pairs as three matrix multiplications, run on Tensor Cores through cuBLAS, followed by one fused Triton kernel. No per-pair histogram is built. |
| **Full-SMEM** | the full 3-D joint histogram fits on chip (`b_max^3 <= 65,536` and within the shared-memory capacity) | Holds the entire joint-count histogram in shared memory, one gene pair per CUDA block, using on-chip atomics that are roughly 80x faster than global-memory atomics. |
| **Adaptive-SMEM** | `b_max` is too large for a uniform shared-memory histogram (CUDA available) | Sizes the shared-memory histogram per pair (`b_i * b_i * b_j`) instead of the global maximum, and coarsens bins for the few high-cardinality genes whose histogram would still overflow the on-chip capacity. |
| **scatter_add** | no CUDA device is available | CPU fallback that accumulates the histograms directly in host memory. |

## Discretization and remapping

Continuous expression is first discretized with the bandwidth-based FSBW-L
scheme. Because most genes use only a few of the possible bins, TENEX then
renumbers the bins that each gene actually uses to consecutive integers
`0, 1, 2, ...`. The largest count across all genes, `b_max`, drives kernel
selection.

## Bin coarsening

A few highly variable genes can produce histograms too large for shared memory.
When `b_max` exceeds the largest count the device shared memory can hold
(`b_safe`, derived from the opt-in shared-memory capacity), TENEX coarsens those
genes once, before computation, by uniformly merging adjacent bins so that every
pair fits on chip. Coarsening is lossy but its accuracy impact is small.

## How the kernel is selected

You do not choose a kernel directly. `auto_select()` walks the kernels in
priority order and picks the first one whose `supports()` predicate holds for the
current data and device. Because those predicates depend on `b_max`, whether the
data is binary, whether a source filter is active, GPU availability, and the
device shared-memory capacity, the same dataset can select a different kernel on
different hardware. A histogram that fits in shared memory on one GPU, for
example, may exceed it on another and fall through from Full-SMEM to
Adaptive-SMEM.

You can request a specific kernel by name. The request is checked against the
same `supports()` predicate, so a kernel that does not fit the data or device
raises `ValueError` instead of running. This path is intended for benchmarking
and debugging, not normal use:

```python
result = engine.compute(accelerator="gpu", kernel="Full-SMEM")
```

Available names are reported by `tenex.registered_kernels()` (`GEMM-B2`,
`Full-SMEM`, `Adaptive-SMEM`, `scatter_add`). Names are case-insensitive.
