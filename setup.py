"""
TENEX build script.

Builds the CUDA extensions ahead of time (AOT) into the wheel. This needs
PyTorch with CUDA and nvcc, so install PyTorch first and build with
``pip install . --no-build-isolation`` so the build sees your torch. Set
``TENEX_CPU_ONLY=1`` to build a pure-Python wheel with no CUDA extensions
(torch is then not required at build time).
"""
import os
from setuptools import setup, find_packages

# The PyPI release uses the plain "0.1.0". The self-hosted wheel-index CI sets
# TENEX_VERSION to a local-tagged string (for example "0.1.0+pt212cu126"). No
# CUDA tag is appended here, so the version is never doubled.
VERSION = os.environ.get("TENEX_VERSION", "0.1.0")

CPU_ONLY = os.environ.get("TENEX_CPU_ONLY", "").lower() in ("1", "true", "yes")


def cuda_ext_modules():
    """Return the CUDA extension list, importing torch lazily so a CPU-only
    build never requires torch or CUDA."""
    import torch
    from torch.utils.cpp_extension import CUDAExtension

    # A CI wheel builder has no GPU, so allow the build when TORCH_CUDA_ARCH_LIST
    # or FORCE_CUDA is set (the CUDA toolkit / nvcc is present in the build image).
    if (not torch.cuda.is_available()
            and not os.environ.get("TORCH_CUDA_ARCH_LIST")
            and not os.environ.get("FORCE_CUDA")):
        raise RuntimeError(
            "No CUDA runtime detected and neither TORCH_CUDA_ARCH_LIST nor "
            "FORCE_CUDA is set. Install the CUDA toolkit, set TORCH_CUDA_ARCH_LIST, "
            "or set TENEX_CPU_ONLY=1 to build without the CUDA extensions."
        )

    cuda_flags = ["-O3", "--use_fast_math"]
    trace_flags = ["-O3"]  # TRACE kernels use log2f, so no --use_fast_math

    def ext(name, source, flags):
        return CUDAExtension(
            name=f"tenex._ext.{name}",
            sources=[f"tenex/csrc/{source}"],
            extra_compile_args={"nvcc": flags},
        )

    return [
        ext("te_smem", "full_smem.cu", cuda_flags),
        ext("te_adaptive_smem", "adaptive_smem.cu", cuda_flags),
        ext("te_gmem", "gmem.cu", cuda_flags),
        ext("binarize_cuda", "binarize_cuda.cu", cuda_flags),
        ext("full_smem_surrogate_test", "full_smem_surrogate_test.cu", cuda_flags),
        ext("adaptive_smem_surrogate_test", "adaptive_smem_surrogate_test.cu", cuda_flags),
        ext("trace_sort_entropy", "trace_sort_entropy.cu", trace_flags),
    ]


if CPU_ONLY:
    ext_modules, cmdclass = [], {}
else:
    from torch.utils.cpp_extension import BuildExtension

    ext_modules = cuda_ext_modules()
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="tnx",
    version=VERSION,
    packages=find_packages(include=["tenex", "tenex.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    package_data={"tenex": ["csrc/*.cu", "assets/*.png"]},
)
