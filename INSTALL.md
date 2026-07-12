# Installing TENEX

TENEX provides pre-built wheels with AOT-compiled CUDA kernels for the best
out-of-the-box experience. If a pre-built wheel is not available for your
environment, you can build from source or rely on JIT compilation as a fallback.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | >= 3.10 |
| PyTorch | >= 2.0 (with CUDA support) |
| CUDA toolkit | 11.8, 12.x, or 13.x |
| OS | Linux x86_64 |

---

## 1. Pre-built Wheels (Recommended)

Pre-built wheels ship with AOT-compiled CUDA extensions — no compiler toolchain
needed on the user machine.

### Install

PyTorch is a prerequisite rather than a TENEX dependency, so install it first
with the CUDA build you run. TENEX is then installed with PyPI kept as the
primary index (so numpy, scipy, and the other runtime dependencies resolve
normally) and the TENEX wheel index added as an extra source.

```bash
# Step 1: Install PyTorch with your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu132

# Step 2: Install TENEX from the matching CUDA sub-index
pip install tnx --extra-index-url https://cxinsys.github.io/tenex/whl/cu132/
```

Use the same CUDA line in both URLs. Prebuilt GPU wheels are published for
Linux (`cu132`, `cu128`) and Windows (`cu128`, `cu126`); the sub-index URL keeps
each CUDA line separate so pip picks the wheel that matches your PyTorch. Step 2
installs:

- **tenex** (with pre-compiled `.so` kernels)
- **numpy**, **scipy**, and the other runtime dependencies from PyPI

### Available wheel matrix

GPU wheels are built for these platform and CUDA lines, each across Python
3.10, 3.11, 3.12, and 3.13:

| Platform | CUDA line | PyTorch | Sub-index |
|----------|-----------|---------|-----------|
| Linux x86_64 | `cu132` | 2.13.0 | `.../whl/cu132/` |
| Linux x86_64 | `cu128` | 2.11.0 | `.../whl/cu128/` |
| Windows x86_64 | `cu128` | 2.8.0 | `.../whl/cu128/` |
| Windows x86_64 | `cu126` | 2.8.0 | `.../whl/cu126/` |

> Install the PyTorch version listed for your CUDA line, then install TENEX from
> the matching sub-index. The universal CPU wheel (`py3-none-any`, on PyPI)
> covers macOS, Windows, and Linux when no GPU is used.

Wheel naming convention:
```
tnx-0.1.0+pt213cu132-cp312-cp312-linux_x86_64.whl
       │     │   │      │
       │     │   │      └─ Python 3.12
       │     │   └─ CUDA 13.2
       │     └─ PyTorch 2.13
       └─ version
```

### Verify installation

```python
import tenex as tnx

print([k.name for k in tnx.registered_kernels()])
# ['GEMM-B2', 'Full-SMEM', 'Adaptive-SMEM', 'scatter_add']
```

If AOT extensions loaded successfully, `GEMM-B2`, `Full-SMEM`, `Adaptive-SMEM`,
and `scatter_add` will appear without triggering JIT compilation.

---

## 2. Build from Source

Build TENEX yourself when you need a CUDA version or architecture not covered
by the pre-built wheels.

### Prerequisites

- CUDA toolkit with `nvcc` (set `CUDA_HOME` if not in default path)
- PyTorch with CUDA support
- C++ compiler (GCC >= 9)

### Install

```bash
git clone https://github.com/cxinsys/tenex.git
cd tenex

# Install PyTorch first (match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu132

# Build and install with AOT-compiled CUDA extensions
pip install .
```

### Build options

| Environment variable | Description | Example |
|---------------------|-------------|---------|
| `CUDA_HOME` | Path to CUDA toolkit | `/usr/local/cuda-12.8` |
| `TORCH_CUDA_ARCH_LIST` | Target GPU architectures | `"8.0;8.6;9.0"` |
| `FORCE_CUDA` | Force CUDA build (skip detection) | `1` |
| `TENEX_VERSION` | Override package version | `0.2.0` |

### Example: targeting specific GPU architectures

```bash
# Build for Ampere (A100) and Hopper (H100) only
TORCH_CUDA_ARCH_LIST="8.0;9.0" pip install .
```

### Build a wheel

```bash
FORCE_CUDA=1 python setup.py bdist_wheel
# Output: dist/tnx-0.1.0+cu132-cp312-cp312-linux_x86_64.whl
```

---

## 3. JIT Compilation (Automatic Fallback)

If no AOT-compiled extension is found at import time, TENEX automatically falls
back to JIT (Just-In-Time) compilation using `torch.utils.cpp_extension.load_inline`.
This requires `nvcc` on the user machine.

### When JIT is used

- Installing from a pure-Python (CPU) wheel
- Installing via `pip install` from source without CUDA toolkit at build time
- Using a CUDA version not covered by the pre-built wheels

### Prerequisites for JIT

- CUDA toolkit with `nvcc`
- Set `CUDA_HOME` if nvcc is not on `PATH`:
  ```bash
  export CUDA_HOME=/usr/local/cuda-12.8
  ```

### Behavior

The first kernel call triggers compilation (may take 30–60 seconds). Compiled
modules are cached in `~/.cache/torch_extensions/` and reused in subsequent runs.

> **Note:** JIT compilation is a convenience fallback. For production use, prefer
> pre-built wheels or building from source to avoid first-run latency and the
> need for a compiler toolchain on every machine.

---

## 4. CPU-only (No GPU)

TENEX can run on CPU for small datasets or testing, though GPU acceleration is
strongly recommended for any real workload.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tnx
```

The universal CPU wheel is on PyPI, so no extra index is needed here. This same
wheel installs on macOS, Windows, and Linux.

CPU mode uses the `scatter_add` kernel backend (pure PyTorch, no CUDA needed).

---

## 5. Installation for Development

```bash
git clone https://github.com/cxinsys/tenex.git
cd tenex

pip install torch --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[dev]"
```

This installs in editable mode with test dependencies (`pytest`, `triton`).
CUDA extensions are JIT-compiled on first use unless you run `python setup.py
build_ext --inplace` to AOT-compile them into the source tree.

### Run tests

```bash
pytest                    # CPU tests only
pytest -m cuda            # GPU tests
pytest -m "not cuda"      # skip GPU tests
```

---

## Troubleshooting

### `RuntimeError: No CUDA runtime detected`

Your PyTorch was installed without CUDA support. Reinstall with the correct index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu132
```

### JIT compilation fails with `nvcc not found`

Set `CUDA_HOME` to your CUDA toolkit path:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
```

### Wheel version mismatch

TENEX wheels are tagged with PyTorch and CUDA versions (e.g., `+pt212cu132`).
If you upgrade PyTorch, reinstall TENEX with a matching wheel:

```bash
pip install --force-reinstall tnx --extra-index-url https://cxinsys.github.io/tenex/whl/cu132/
```

### `ImportError` after install

Verify the extension was loaded:

```python
try:
    import tenex._ext.te_smem
    print("AOT extension loaded")
except ImportError:
    print("Falling back to JIT")
```
