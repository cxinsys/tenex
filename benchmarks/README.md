# TENEX benchmarks

Reproduce the end-to-end timing reported in the TENEX paper, on your own
hardware and data. These scripts use only the public TENEX API and assume TENEX
is already installed.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/
```

(Change `cu132` to match your CUDA toolkit. See the project `INSTALL.md`.)

## What is measured

`run_benchmark.py` times the TE computation — discretize, select a kernel,
compute the `n x n` transfer-entropy matrix, and assemble the result — using the
paper's protocol. Loading and aligning the data happens once, before the timed
runs, so it is not included in the reported time:

- a few **warm-up** runs are discarded (they absorb JIT and cache warm-up),
- several **timed** runs follow, and the **median** is reported,
- the GPU is synchronized around each run so the time covers all asynchronous
  kernel work, not just the launch.

It reports the selected kernel, `b_max`, median wall-clock time, throughput
(pairs/s), and peak GPU memory.

## Run

Point a dataset at your files by editing `DATA_ROOT` (or the per-entry paths) in `datasets.py`, then:

```bash
# A registered dataset
python run_benchmark.py --dataset mesc

# Your own files
python run_benchmark.py \
    --expression expr.csv --pseudotime pseudotime.txt --branch branch.txt

# Multi-GPU, more repeats, with the per-phase breakdown
python run_benchmark.py --dataset cengen --n-gpus 2 --repeats 3 --profile
```

### Compare against FastTENET

To reproduce the speedup and the numerical-agreement (Pearson correlation)
numbers, install [FastTENET](https://github.com/cxinsys/fasttenet) and add
`--compare-fasttenet`:

```bash
pip install fasttenet
python run_benchmark.py --dataset skin --compare-fasttenet
```

The comparison runs both engines on the same aligned data and prints the speedup
and the Pearson correlation between the two TE matrices. FastTENET's API can vary
by version; if the run fails, adapt `benchmark_fasttenet()` in `run_benchmark.py`
to your installed version.

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | timing harness and CLI |
| `datasets.py` | dataset registry (edit the paths) |

## Notes

- The datasets used in the paper are public; download them from their original
  sources and convert to a `(genes x cells)` matrix plus pseudotime and branch
  files. The registry lists the expected sizes for reference.
- TENEX selects the kernel automatically from `b_max` and the device. To force
  one, pass `kernel=...` to `engine.compute()` (see the API reference).
- For datasets whose `b_max` exceeds the device shared-memory budget, TENEX
  applies bin coarsening; this is the only step that breaks bit-for-bit
  equivalence with FastTENET (correlation stays very high).
