# Contributing to TENEX

Thanks for your interest in improving TENEX. This guide covers a quick
development setup, running the tests, and the licensing terms for contributions.

## Development setup

PyTorch is a prerequisite and is installed separately so you can match your
CUDA build. Once PyTorch is present, install TENEX in editable mode.

```bash
git clone https://github.com/cxinsys/tenex.git
cd tenex

# Install PyTorch first, matching your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu132

# Editable install with the development and k-means extras
pip install -e ".[dev,kmeans]" --no-build-isolation
```

`--no-build-isolation` lets the CUDA extensions build against the PyTorch you
just installed. If no CUDA toolkit is available, the kernels fall back to JIT
compilation on first use, or you can build a CPU-only tree with
`TENEX_CPU_ONLY=1 pip install -e . --no-build-isolation`.

## Running the tests

```bash
pytest -k "not cuda"      # CPU-only tests (no GPU required)
pytest -m cuda            # GPU tests (require a CUDA device)
```

Please make sure the CPU test selection passes before opening a pull request.

## Coding conventions

- Python identifiers use `snake_case` for variables, functions, and attributes.
- Keep numerical results consistent with the reference FastTENET implementation
  (no overflow or NaN, matching values within float32 precision).

## Licensing of contributions

TENEX is source-available under the **TENEX Non-Commercial License** (see
[LICENSE](LICENSE)). By submitting a contribution you agree that it is
licensed under the same Non-Commercial terms. For commercial licensing or any
questions, contact Daewon Lee (dwlee@cau.ac.kr).
