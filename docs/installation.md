# Installation

TENEX ships pre-built wheels with AOT-compiled CUDA kernels, so no compiler
toolchain is needed on the target machine. Build-from-source and a JIT fallback
are also available.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.0 (with CUDA support) |
| CUDA toolkit | 11.8, 12.x, or 13.x |
| OS | Linux x86_64 |

## Pre-built wheels (recommended)

PyTorch is required at runtime but is intentionally not a hard dependency, so
that you control which CUDA build is installed. Install it first, then install
TENEX with PyPI kept as the primary index (so NumPy, SciPy, and the other
runtime dependencies resolve normally) and the TENEX wheel index added as an
extra source.

```bash
# Step 1: install PyTorch for your CUDA line
pip install torch --index-url https://download.pytorch.org/whl/cu132
# Step 2: install TENEX from the matching CUDA sub-index
pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/cu132/
```

Use the same CUDA line in both URLs. Pre-built GPU wheels are published for
Linux (`cu132`, `cu128`) and Windows (`cu128`, `cu126`). The sub-index keeps
each CUDA line separate, so pip installs the wheel that matches your PyTorch
alongside NumPy and SciPy. On macOS, or for CPU-only use on any platform,
`pip install tnx` from PyPI installs the universal CPU wheel.

## How the wheels are distributed

The wheels are built by GitHub Actions on a version tag and land in three
places. The Linux and Windows CUDA wheels and the universal CPU wheel are
attached to the GitHub Release as downloadable `.whl` files. The universal CPU
wheel and the source distribution are also uploaded to PyPI, so
`pip install tnx` works with no extra index. The CUDA wheels carry local
version labels (for example `0.1.0+pt213cu132`) that PyPI does not accept, so
they are served from a self-hosted index on GitHub Pages
(`https://cxinsys.github.io/tenex/whl/<cuda-line>/`) whose entries link back to
the files on the GitHub Release. Selecting the sub-index for your CUDA line is
what makes pip fetch the matching wheel.

## Verify the installation

```python
import tenex as tnx

print([k.name for k in tnx.registered_kernels()])
# ['GEMM-B2', 'Full-SMEM', 'Adaptive-SMEM', 'scatter_add']
```

## Build from source and troubleshooting

For build-from-source, the JIT fallback, CPU-only mode, the full wheel matrix,
and troubleshooting, see
[`INSTALL.md`](https://github.com/cxinsys/tenex/blob/main/INSTALL.md) in the
repository.
