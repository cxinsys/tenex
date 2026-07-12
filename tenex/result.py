"""
TransferEntropyResult — rich result object from TransferEntropyEngine.compute().

Wraps the (n, n) TE matrix with metadata needed for downstream analysis
(NetWeaver, POINT, etc.). Behaves like a numpy array for backward
compatibility via __array__, __getitem__, and attribute delegation.
"""

import numpy as np


class TransferEntropyResult:
    """
    Result of TransferEntropyEngine.compute().

    Attributes
    ----------
    matrix         : (n, n) float32 ndarray — pairwise TE values.
    variable_names : (n,) str ndarray — variable (e.g. gene) names.
    bin_arrs       : (n, T, K) ndarray — discretized bin arrays (uint8/int16/int32).
    n_per_var      : (n,) int32 ndarray — number of unique bins per variable.
    b_max       : int — global maximum bin count after remapping.
    tau             : int — time lag used for TE computation.
    kernel         : str — name of the kernel used.
    timings        : dict | None — phase-level timing breakdown (if profile=True).
    """

    __slots__ = (
        'matrix', 'variable_names', 'bin_arrs', 'n_per_var',
        'b_max', 'tau', 'kernel', 'timings', 'sources',
    )

    def __init__(
        self,
        matrix: np.ndarray,
        variable_names: np.ndarray,
        bin_arrs: np.ndarray | None = None,
        n_per_var: np.ndarray | None = None,
        b_max: int = 0,
        tau: int = 1,
        kernel: str = '',
        timings: dict | None = None,
        sources: np.ndarray | None = None,
    ):
        self.matrix = matrix
        self.variable_names = np.asarray(variable_names)
        self.bin_arrs = bin_arrs
        self.n_per_var = n_per_var
        self.b_max = b_max
        self.tau = tau
        self.kernel = kernel
        self.timings = timings
        # Source (TF) filter used for this computation, if any. Only the
        # (source -> target) entries were computed; the rest of the matrix is
        # structurally zero. Downstream inference uses this to restrict its
        # null distribution and BH correction to the computed pairs.
        self.sources = sources

    # ── numpy compatibility ──────────────────────────────────────────────

    def __array__(self, dtype=None, copy=None):
        """Allow np.array(result), np.save(result), etc."""
        arr = self.matrix if dtype is None else self.matrix.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key):
        """Allow result[i, j] indexing."""
        return self.matrix[key]

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def dtype(self):
        return self.matrix.dtype

    @property
    def ndim(self):
        return self.matrix.ndim

    @property
    def size(self):
        return self.matrix.size

    @property
    def T(self):
        return self.matrix.T

    def __len__(self):
        return len(self.matrix)

    # ── Convenience ──────────────────────────────────────────────────────

    @property
    def n_vars(self) -> int:
        return self.matrix.shape[0]

    def __repr__(self):
        return (
            f"TransferEntropyResult(n_vars={self.n_vars}, "
            f"kernel='{self.kernel}', tau={self.tau}, b_max={self.b_max})"
        )
