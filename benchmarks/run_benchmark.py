#!/usr/bin/env python
"""
TENEX end-to-end benchmark.

Measures the wall-clock time to compute the full pairwise transfer-entropy (TE)
matrix for a dataset, and optionally compares against FastTENET for speedup and
numerical agreement. This is the same measurement protocol used in the TENEX
paper: a few warm-up runs (to absorb JIT / cache warm-up) followed by several
timed runs, reported as the median.

Assumes TENEX is already installed:

    pip install torch --index-url https://download.pytorch.org/whl/cu132
    pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/

Examples
--------
# Time TENEX on a registered dataset (edit the paths in datasets.py first)
python run_benchmark.py --dataset mesc

# Time TENEX on your own files, on 2 GPUs, with 3 timed repeats
python run_benchmark.py \
    --expression expr.csv --pseudotime ptime.txt --branch branch.txt \
    --n-gpus 2 --repeats 3

# Print the per-phase breakdown (discretization, compute, assemble)
python run_benchmark.py --dataset cengen --profile

# Also run FastTENET and report speedup + Pearson correlation
# (requires `pip install fasttenet`)
python run_benchmark.py --dataset skin --compare-fasttenet
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import tenex as tnx

from datasets import DATASETS


# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _torch():
    """Return the torch module if available, else None (CPU-only timing)."""
    try:
        import torch
        return torch
    except Exception:
        return None


def _sync(devices):
    """Block until all GPU work is finished. Required for accurate timing
    because CUDA kernels launch asynchronously."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        for d in devices or range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)


def time_call(fn, devices, warmup=1, repeats=3):
    """Run ``fn`` ``warmup + repeats`` times and return (median, all_times).

    The GPU is synchronized before and after each call so the measured duration
    covers the full asynchronous workload, not just the kernel launch. The first
    ``warmup`` runs are discarded.
    """
    times = []
    for i in range(warmup + repeats):
        _sync(devices)
        t0 = time.perf_counter()
        fn()
        _sync(devices)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    return statistics.median(times), times


def peak_memory_gb():
    """Peak GPU memory (GB) allocated since the last reset, summed over GPUs."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return sum(
        torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())
    ) / 1024**3


def reset_peak_memory():
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)


def fmt_time(sec):
    """Human-readable duration."""
    if sec < 1e-3:
        return f"{sec * 1e6:.0f} us"
    if sec < 1:
        return f"{sec * 1e3:.1f} ms"
    if sec < 60:
        return f"{sec:.2f} s"
    if sec < 3600:
        return f"{sec / 60:.1f} min"
    return f"{sec / 3600:.2f} h"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(expression, pseudotime, branch, gene_names=None):
    """Load a dataset into an ``ScRnaData`` via ``tenex.load_scrna``.

    ``expression`` may be a CSV path (read directly by ``load_scrna``) or a
    ``.npy`` path holding an ``(n_genes, n_cells)`` array. For the ``.npy`` case
    a ``gene_names`` file (``.npy`` or ``.txt``) is also loaded, since names are
    required when the expression is passed as an array.
    """
    if isinstance(expression, str) and expression.endswith(".npy"):
        arr = np.load(expression, allow_pickle=True)
        names = None
        if gene_names:
            names = (np.load(gene_names, allow_pickle=True)
                     if gene_names.endswith(".npy")
                     else np.loadtxt(gene_names, dtype=str))
        return tnx.load_scrna(expression=arr, pseudotime=pseudotime,
                             branch=branch, gene_names=names)
    return tnx.load_scrna(expression=expression, pseudotime=pseudotime, branch=branch)


# ─────────────────────────────────────────────────────────────────────────────
# TENEX benchmark
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_tenex(scrna, accelerator, devices, tau, warmup, repeats, profile):
    """Time the end-to-end TE computation and return a result dict."""
    n_genes = scrna.data.shape[0]
    n_pairs = n_genes * (n_genes - 1)

    # A fresh engine per call keeps each timed run independent (no caching).
    def one_run():
        engine = tnx.TransferEntropyEngine(
            data=scrna.data, variable_names=scrna.gene_names,
        )
        return engine.compute(
            accelerator=accelerator, devices=devices,
            binning_method="FSBW-L", kp=0.5, tau=tau,
            profile=profile,
        )

    reset_peak_memory()
    median, all_times = time_call(one_run, devices, warmup=warmup, repeats=repeats)

    # One more run to capture the result object (matrix, kernel, phase timings).
    result = one_run()

    out = {
        "median_s": median,
        "all_times_s": all_times,
        "n_genes": n_genes,
        "n_pairs": n_pairs,
        "throughput_pairs_s": n_pairs / median if median else float("nan"),
        "kernel": getattr(result, "kernel", None),
        "b_max": getattr(result, "b_max", None),
        "peak_memory_gb": peak_memory_gb(),
        "matrix": result.matrix,
        "phase_timings": getattr(result, "timings", None) if profile else None,
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Optional FastTENET comparison
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_fasttenet(expression, pseudotime, branch, gene_names, n_gpus, warmup, repeats):
    """Time FastTENET end-to-end. Requires the ``fasttenet`` package.

    Returns (median_seconds, te_matrix) or raises if FastTENET is unavailable.
    The exact FastTENET API can change between versions; adapt this function if
    your installed version differs.
    """
    import fasttenet as fte  # optional dependency

    # Load the expression + names the same way as the TENEX side: FastTENET's
    # load_exp_data parses a CSV, while a .npy is loaded directly (with names).
    if isinstance(expression, str) and expression.endswith(".npy"):
        exp_data = np.load(expression, allow_pickle=True)
        node_name = None
        if gene_names:
            node_name = (np.load(gene_names, allow_pickle=True)
                         if gene_names.endswith(".npy")
                         else np.loadtxt(gene_names, dtype=str))
    else:
        node_name, exp_data = fte.load_exp_data(expression)

    trj = np.loadtxt(pseudotime)
    branch_arr = np.loadtxt(branch, dtype=int)
    aligned = fte.align_data(data=exp_data, trj=trj, branch=branch_arr)

    def one_run():
        worker = fte.FastTENET(aligned_data=aligned, node_name=node_name)
        return worker.run(device_ids=n_gpus, kp=0.5, binning_method="FSBW-L", dt=1)

    median, _ = time_call(one_run, devices=list(range(n_gpus)),
                          warmup=warmup, repeats=repeats)
    te = one_run()
    return median, np.asarray(te)


def pearson(a, b):
    """Pearson correlation between two TE matrices (all entries; the diagonal is
    zero in both, so it does not affect the result)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Benchmark TENEX (and optionally FastTENET) on an scRNA-seq dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_argument_group("dataset (use --dataset OR the three path flags)")
    src.add_argument("--dataset", choices=sorted(DATASETS), help="a dataset registered in datasets.py")
    src.add_argument("--expression", help="expression matrix (.csv or .npy)")
    src.add_argument("--pseudotime", help="pseudotime vector file")
    src.add_argument("--branch", help="branch / cell-selection file")
    src.add_argument("--gene-names", help="gene-name file (needed when expression is .npy)")

    run = p.add_argument_group("run configuration")
    run.add_argument("--accelerator", default="auto", choices=["auto", "gpu", "cpu"])
    run.add_argument("--n-gpus", type=int, default=None,
                     help="number of GPUs (default: all available)")
    run.add_argument("--tau", type=int, default=1, help="time lag")
    run.add_argument("--warmup", type=int, default=1, help="warm-up runs (discarded)")
    run.add_argument("--repeats", type=int, default=3, help="timed runs (median reported)")
    run.add_argument("--profile", action="store_true", help="print the per-phase breakdown")
    run.add_argument("--compare-fasttenet", action="store_true",
                     help="also run FastTENET and report speedup + correlation")
    args = p.parse_args()

    # Resolve the dataset paths.
    if args.dataset:
        d = DATASETS[args.dataset]
        expression = d["expression"]
        pseudotime = d["pseudotime"]
        branch = d["branch"]
        gene_names = d.get("gene_names")
        desc = d.get("description", args.dataset)
    elif args.expression and args.pseudotime and args.branch:
        expression, pseudotime, branch = args.expression, args.pseudotime, args.branch
        gene_names = args.gene_names
        desc = expression
    else:
        p.error("provide --dataset, or all of --expression/--pseudotime/--branch")

    devices = list(range(args.n_gpus)) if args.n_gpus else None

    print(f"Dataset:     {desc}")
    print("Loading ...", flush=True)
    scrna = load_dataset(expression, pseudotime, branch, gene_names)
    n_genes, n_cells = scrna.data.shape
    print(f"  genes={n_genes:,}  cells={n_cells:,}  pairs={n_genes * (n_genes - 1):,}")

    # ── TENEX ────────────────────────────────────────────────────────────────
    print(f"\nBenchmarking TENEX ({args.warmup} warm-up + {args.repeats} timed runs) ...",
          flush=True)
    tnx = benchmark_tenex(scrna, args.accelerator, devices, args.tau,
                          args.warmup, args.repeats, args.profile)

    print(f"\n── TENEX ──")
    print(f"  kernel:       {tnx['kernel']}  (b_max={tnx['b_max']})")
    print(f"  median time:  {fmt_time(tnx['median_s'])}")
    print(f"  throughput:   {tnx['throughput_pairs_s']:,.0f} pairs/s")
    if tnx["peak_memory_gb"] is not None:
        print(f"  peak GPU mem: {tnx['peak_memory_gb']:.2f} GB")
    if args.profile and tnx["phase_timings"]:
        print("  phases (s):")
        for name, sec in tnx["phase_timings"].items():
            if isinstance(sec, (int, float)):
                print(f"    {name:18s} {sec:10.4f}")

    # ── FastTENET comparison (optional) ──────────────────────────────────────
    if args.compare_fasttenet:
        n_gpus = args.n_gpus or 1
        print(f"\nBenchmarking FastTENET ({n_gpus} GPU(s)) ...", flush=True)
        try:
            ft_median, ft_matrix = benchmark_fasttenet(
                expression, pseudotime, branch, gene_names, n_gpus, args.warmup, args.repeats)
        except ImportError:
            print("  FastTENET is not installed (`pip install fasttenet`); skipping.")
        except Exception as exc:  # version / API differences
            print(f"  FastTENET run failed ({exc}); skipping comparison.")
        else:
            r = pearson(tnx["matrix"], ft_matrix)
            speedup = ft_median / tnx["median_s"] if tnx["median_s"] else float("nan")
            print(f"\n── FastTENET ──")
            print(f"  median time:  {fmt_time(ft_median)}")
            print(f"\n── Comparison ──")
            print(f"  speedup (FastTENET / TENEX): {speedup:.1f}x")
            print(f"  Pearson correlation:         {r:.6f}")


if __name__ == "__main__":
    main()
