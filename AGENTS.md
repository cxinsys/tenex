# TENEX for AI coding agents

This guide is written for AI coding agents such as Claude Code, Codex (GPT),
Gemini, Grok, and Kimi that help researchers **use** or **extend** TENEX. It captures the architecture, the public API, the invariants that must
hold, and the workflow for changing the code. Read it before proposing edits or
generating example code.

(The same content lives in `AGENTS.md` at the repository root, which most agents
discover automatically.)

---

## 1. What TENEX is

**TENEX** (TENET eXtremely optimized) computes pairwise **transfer entropy (TE)**
on the GPU to infer **gene regulatory networks (GRNs)** from single-cell RNA-seq
data. It is a high-performance, numerically faithful reimplementation of
[FastTENET](https://github.com/cxinsys/fasttenet) (itself a fast version of
TENET).

The core idea: estimating TE for every ordered gene pair reduces to counting a
**3-D joint histogram** over three discretized values per cell, namely the
future level of the target gene, its own present level, and the present level
of the candidate regulator. This counting is the bottleneck. TENEX accelerates it by

1. packing each `(future, past-self, past-other)` bin triplet into a single
   integer address and counting directly on the GPU (no sorting), and
2. auto-selecting, per dataset and device, the fastest of several CUDA kernels.

TENEX reproduces FastTENET TE values within float32 precision when no bin
coarsening is applied.

---

## 2. Repository layout

```
tenex/
├── __init__.py            public API surface (see §4)
├── io.py                  load_scrna, ScRnaData
├── transferentropy.py     TransferEntropyEngine (orchestrator)
├── preprocess.py          FSBW-L discretization, dense remap, bin coarsening
├── pipeline.py            Pipeline / PipelineResult (compute once, infer many)
├── result.py              TransferEntropyResult
├── utils.py               helpers (data loading utilities)
├── kernels/               TE compute kernels
│   ├── __init__.py          TEKernel ABC, registry, auto_select()
│   ├── gemm_b2.py           GEMM kernel (binary data, b_max == 2)
│   ├── full_smem.py         Full-SMEM kernel (small b_max)
│   ├── adaptive_smem.py     Adaptive-SMEM kernel (large b_max, per-pair sizing)
│   ├── scatter_add.py       CPU/GPU fallback
│   └── *_surrogate_test.py  fused surrogate-test variants
├── inference/             link inference
│   ├── netweaver.py         NetWeaver (orchestrates inference methods)
│   ├── fdr.py / clr.py / nd.py   matrix-based methods
│   ├── surrogate_test.py    surrogate-based statistical test
│   ├── trace.py             TRACE marginal key-driver inference (OutTE/InTE)
│   ├── point.py             POINT placeholder (raises NotImplementedError)
│   └── grn.py               GRN container
├── csrc/                  CUDA kernel sources (*.cu), compiled into tenex/_ext
└── _ext/                  AOT-compiled extension modules (built at install)

tests/        pytest suite        tutorials/  runnable examples
docs/         mkdocs site         .github/    CI (wheels, docs, index)
```

---

## 3. Installation and environment

- Python >= 3.10, Linux x86_64, an NVIDIA GPU for the CUDA path (a CPU fallback
  exists). PyTorch is required at **runtime** (`import tenex` imports `torch`),
  but is intentionally not in `install_requires` so the user installs the CUDA
  build they want; install it separately before TENEX.

```bash
# Install PyTorch first, then TENEX (PyPI stays primary, TENEX wheels are extra)
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/
```

Change `cu132` to match the user's CUDA toolkit. See `INSTALL.md` for the wheel
matrix, build-from-source, and the JIT fallback.

---

## 4. Public API (top-level `tenex`)

Signatures and defaults below are authoritative, so do not invent kwargs.

```python
import tenex as tnx

# Load + align (filter by branch, order by pseudotime)
scrna = tnx.load_scrna(
    expression,
    pseudotime,
    branch,
    gene_names=None,
    branch_id=1,
    sources=None,
    make_binary=False,
)
# -> ScRnaData with .data (n_genes x n_cells), .gene_names, .sources

# Compute the n x n TE matrix
engine = tnx.TransferEntropyEngine(
    data,
    variable_names,
    sources=None,
)
result = engine.compute(
    accelerator="auto",
    devices="auto",        # list of GPU indices, an int count, or "auto"/-1 for all
    binning_method="FSBW-L",
    kp=0.5,
    tau=1,
    batch_size=None,
    autotune=False,
    kernel=None,
    coarsening=None,
    use_numpy_bins=None,
    profile=False,
)
# -> TransferEntropyResult: .matrix[i, j] = TE(i -> j), plus .variable_names,
#    .bin_arrs, .n_per_var, .b_max, .tau, .kernel, .timings (numpy-compatible)

# Infer a GRN
nw = tnx.NetWeaver(
    result,
    sources=None,
    fdr=0.01,
    links=0,
    is_trimming=True,
    trim_threshold=0.0,
)
grn, trimmed = nw.infer(method="fdr", device=None)   # device=None auto-detects
```

- `tnx.available_methods()` -> `['clr', 'fdr', 'nd', 'point', 'surrogate_test', 'trace']`.
  `point` is a reserved placeholder and raises `NotImplementedError`.
- `surrogate_test` returns a `SurrogateTestResult` (`.effective_te`, `.p_values`,
  `.grn`, ...). Kwargs: `n_surrogates`, `shuffle_method` (`block`/`random`),
  `block_length`, `p_method` (`parametric`/`mc`), `fused`, `seed`, `devices`.
- `trace` returns a `TRACEResult` (`.outte`, `.inte`, `top_drivers(k)`,
  `top_receivers(k)`). Kwargs: `n_surrogates`, `significance`, `devices`.
- `GRN`: `.source`, `.target`, `.te`, `to_sif()`, `to_edge_list()`.
- `tnx.Pipeline(engine, fdr=0.05).run(methods=[...], method_kwargs={...})` computes
  the TE matrix once and reuses it across methods.
- Kernels: `tnx.registered_kernels()` -> kernels named `GEMM-B2`, `Full-SMEM`,
  `Adaptive-SMEM`, `scatter_add`. `tnx.auto_select(...)`, `tnx.get_kernel(name)`
  (case-insensitive), `tnx.TEKernel`.

To restrict the TE computation to a set of regulators (transcription factors),
pass them as `sources` to `TransferEntropyEngine`; TE is then computed only
**from** those genes (GEMM-B2 is excluded in this mode). `load_scrna(sources=...)`
only records the list on the `ScRnaData`, and `NetWeaver(sources=...)` forwards
them to the inference step; neither one filters the TE computation itself.

---

## 5. How it works (3-stage pipeline)

**Stage 1. Discretization and dense remapping** (`preprocess.py`). Continuous
expression is binned with the bandwidth-based **FSBW-L** scheme (`kp = 0.5`).
Because most genes use only a few of the available bins, TENEX renumbers the
used bins of each gene to consecutive integers `0..b_g-1`. The global maximum
`b_max = max_g b_g` drives kernel selection.

**Stage 2. Kernel selection and TE computation** (`kernels/`). `auto_select`
returns the first supported kernel in priority order:

| Kernel | Selected when | Mechanism |
|--------|---------------|-----------|
| `GEMM-B2` | `b_max == 2`, CUDA, no source/TF filter | binary counts become 3 matrix multiplications + 1 fused Triton kernel; no per-pair histogram |
| `Full-SMEM` | `b_max**3 <= 65536` and the histogram fits the device shared-memory capacity | full 3-D histogram in on-chip shared memory (SMEM), one gene pair per CUDA block, fast SMEM atomics |
| `Adaptive-SMEM` | CUDA available, larger `b_max` | per-pair histogram sized `b_i*b_i*b_j` instead of the global max; high-cardinality genes are coarsened to fit |
| `scatter_add` | no CUDA device | CPU fallback (also a universal fallback) |

Key sizing facts (for reasoning about kernel choice and memory):

- Full-SMEM per-block size: `SMEM(b_max, W) = [b_max*(b_max+1)^2 + W] * 4` bytes
  (`W` = warps per block; count arrays are `int32`, hence `* 4`).
- `S_opt-in` is the device opt-in shared memory, queried at runtime via the CUDA
  attribute `cudaDevAttrMaxSharedMemoryPerBlockOptin`. **It is device-dependent.**
  On the workstation GPUs used for the paper benchmarks (RTX A5000, RTX 4090,
  RTX PRO 6000 Blackwell Max-Q, all compute capability with 99 KB opt-in) it is
  101,376 bytes (99 KB), giving `b_safe = 28`. Data-center cards differ. Never
  hard-code a single value; derive it from the device.
- `b_safe` = largest `b` with `[b*(b+1)^2 + W] * 4 <= S_opt-in`. When
  `b_max > b_safe`, **bin coarsening** uniformly merges adjacent bins down to
  `b_safe` so every pair fits on chip. Coarsening is lossy but its accuracy
  impact is small; it is the only step that breaks bit-for-bit equivalence with
  FastTENET.

**Stage 3. Integration and GRN inference** (`inference/`). Per-pair (and
per-GPU) TE values are assembled into the `n x n` matrix; edges are selected by
significance thresholding (z-score + Benjamini-Hochberg FDR) and optionally
trimmed by the data processing inequality (DPI) to remove indirect edges.

Multi-GPU: the SMEM kernels split gene pairs across devices via a thread pool;
GEMM-B2 splits output rows.

---

## 6. Notation (used across code, docs, and the paper)

- `n`: number of genes
- `g`: gene index (`g = 1..n`), the subscript in `max_g` and `b_g`
- `i, j`: gene-pair indices, used in `T[i, j]`, `b_i`, `b_j`
- `l`: number of cells along pseudotime
- `l_eff`: `l - tau`, the number of valid lagged time points
- `tau`: time lag (default 1, FastTENET's `dt`)
- `kp`: FSBW-L bandwidth left-shift parameter (kappa, default 0.5)
- `b_g`: bin count after dense remapping (bins are `0..b_g-1` for gene `g`)
- `b_max`: global max bin count, `b_max = max_g b_g` (kernel-selection criterion)
- `b_i, b_j`: per-pair bin counts
    - Adaptive-SMEM sizes each pair histogram as `b_i*b_i*b_j`
- `b_safe`: max bin count fitting the device SMEM capacity (coarsening trigger)
- `W`: warps per CUDA block (appears in the SMEM size formula)
- `S_opt-in`: device opt-in shared-memory capacity, in bytes
    - queried at runtime
    - sets `b_safe`
- `X_r`: raw binned matrix (FSBW-L output)
- `X_d`: dense-remapped binned matrix
- `X_c`: coarsened binned matrix
- `T`: TE matrix
    - internal layout: `T[i, j] = TE(j -> i)`
    - public layout: `result.matrix[i, j] = TE(i -> j)`

---

## 7. Build, test, and develop

- **Build the AOT wheel** (compiles `csrc/*.cu` into `tenex/_ext`):
  ```bash
  FORCE_CUDA=1 python setup.py bdist_wheel
  ```
  `setup.py` imports `torch` at build time and lists the CUDA extensions; the
  `_ext` `.so` files are gitignored and produced by the build.
- **Run tests:** `pytest` (suite under `tests/`; CUDA tests are marked `cuda`).
- **Docs:** `mkdocs serve` (config in `mkdocs.yml`, sources under `docs/`).
- **CI:** `.github/workflows/build-wheels.yml` builds the wheel matrix
  (PyTorch x CUDA x Python); `docs.yml` deploys the docs; `publish-index.yml`
  publishes the wheel index.

### Adding a kernel
Subclass `TEKernel` (`kernels/__init__.py`), implement `supports(...)` (the
selection predicate) and the compute entry points, then `register(...)` it. The
registry is priority-ordered; `auto_select` returns the first whose `supports`
is true. If it needs a CUDA source, add the `.cu` to `csrc/` and a matching
`CUDAExtension` in `setup.py`.

### Adding an inference method
Register it in `inference/__init__.py` so it appears in `available_methods()` and
is reachable through `NetWeaver.infer(method=...)`.

---

## 8. Invariants and guardrails (important for AI edits)

These are hard constraints. Violating them is a regression.

1. **Numerical fidelity.** TENEX must reproduce FastTENET TE values (float32
   precision) on every code path except where bin coarsening is explicitly
   applied. Never introduce overflow, NaN, or silent precision loss. Count
   arrays are `int32` (assumes `l_eff < 2^31`); bin indices are `int8` when
   `b_max <= 127`.
2. **No data-specific hyperparameters.** The pipeline must work across all data
   scales. Do not tune thresholds, batch sizes, or `b_safe` to one dataset;
   derive device-dependent quantities at runtime.
3. **Rebuild after kernel/source changes.** Editing `csrc/*.cu` or anything that
   affects the compiled extensions requires rebuilding and reinstalling the AOT
   wheel; the installed `_ext/*.so` will otherwise be stale.
4. **CPU/GPU parity.** The CPU fallback and the GPU kernels must agree
   numerically. The `auto_select` decision is deterministic from `(b_max,
   device, source_filter)`.
5. **`use_numpy_bins` semantics.** `None` follows the device. On CUDA it uses
   GPU binning, which is about 5x faster and correlates at ~1.0 but is not
   bit-for-bit. On CPU it uses NumPy binning, which is FastTENET-exact. `True`
   or `False` overrides this default, which should be preserved.
6. **Match CUDA versions.** A TENEX wheel is tied to a specific PyTorch x CUDA
   build; never assume a wheel works across CUDA majors.

---

## 9. Common pitfalls

- Passing a NumPy array to `load_scrna` requires `gene_names`.
- `result.matrix[i, j]` is `TE(i -> j)` (the public layout is transposed from the
  internal `T[i, j] = TE(j -> i)`).
- TRACE and the surrogate test consume the discretized bins on the
  `TransferEntropyResult`; run them on a result produced with the default
  single-lag binning.
- Forcing `kernel="GEMM-B2"` fails unless the data is binary (`b_max == 2`) and
  no source filter is active.

---

## 10. References

- Paper: a manuscript describing TENEX (citation will be added on publication).
- Upstream: FastTENET (Sung et al.) and TENET.
- License: TENEX Non-Commercial License (see `LICENSE`). Non-commercial use only.
