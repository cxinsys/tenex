"""
High-level Pipeline orchestrating TE computation and link inference.

Design pattern: task-oriented dispatcher (cf. HuggingFace ``diffusers``,
``transformers.pipeline``). Users construct a Pipeline, optionally
``configure()`` compute kwargs, then call ``run(data, variable_names,
methods=[...])`` to obtain inference results. The Pipeline constructs
the underlying engine lazily and caches the TE matrix so multiple
inference calls reuse a single compute.

The Pipeline transparently routes:

- **Matrix-based methods** (``"fdr"``, ``"clr"``, ``"nd"``): compute TE
  once, then apply the chosen inference on the matrix.
- **Fused methods** (``"surrogate_test"``, ``"point"``): reuse the cached
  TE matrix, bin arrays, and kernel identifier so the specialized kernel
  can skip redundant work.

Three layers of abstraction are offered:

- **Layer 1**: raw ``TransferEntropyEngine`` + ``NetWeaver`` (unchanged).
- **Layer 2**: ``Pipeline`` (this module) — recommended default.

The classical ``engine.compute() → NetWeaver.infer()`` flow remains fully
supported; ``Pipeline`` is a convenience wrapper, not a replacement.
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch

from tenex.inference import get_method
from tenex.result import TransferEntropyResult
from tenex.transferentropy import TransferEntropyEngine
from tenex._log import set_verbose
from tenex._log import vprint


_MATRIX_METHODS = frozenset({"fdr", "clr", "nd"})
_FUSED_METHODS = frozenset({"surrogate_test", "point"})


@dataclass
class PipelineResult:
    """Composite result of :meth:`Pipeline.run`.

    Attributes
    ----------
    te_result         : shared :class:`TransferEntropyResult` (cached compute).
    inference_results : mapping method-name → method-specific result
                        (e.g. ``(grn, trimmed)`` for FDR,
                        :class:`SurrogateTestResult` for surrogate_test).
    """

    te_result: TransferEntropyResult
    inference_results: dict[str, Any] = field(default_factory=dict)

    @property
    def matrix(self) -> np.ndarray:
        return self.te_result.matrix

    def get(self, method: str) -> Any:
        return self.inference_results[method]

    def __getattr__(self, name: str) -> Any:
        # Allow ``pr.fdr``, ``pr.surrogate_test`` etc. Fall back to the
        # dataclass attributes; avoid recursion on our own slots.
        inf = self.__dict__.get('inference_results', {})
        if name in inf:
            return inf[name]
        raise AttributeError(
            f"{type(self).__name__!r} has no attribute {name!r}"
        )


class Pipeline:
    """End-to-end TE + link-inference pipeline.

    Parameters
    ----------
    engine : optional :class:`TransferEntropyEngine`.  When omitted, the
        engine is constructed lazily on :meth:`run` from the provided
        ``data`` / ``variable_names`` / ``sources`` arguments.  Passing a
        pre-built engine remains supported for low-level callers.
    **defaults : inference kwargs applied to every call unless overridden
        (e.g. ``fdr=0.05``, ``is_trimming=True``, ``links=0``).

    Examples
    --------
    >>> # Recommended: data flows through run()
    >>> pipe = tnx.Pipeline(fdr=0.05).configure(binning_method="FSBW-L", tau=1)
    >>> pr = pipe.run(data=X, variable_names=names, methods=["fdr"])

    >>> # Low-level: pre-built engine
    >>> pipe = tnx.Pipeline(engine, fdr=0.05)
    >>> pr = pipe.run(methods=["fdr", "clr", "surrogate_test"])
    """

    def __init__(
        self,
        engine: TransferEntropyEngine | None = None,
        **defaults,
    ):
        self.engine = engine
        self._defaults = dict(defaults)
        self._te_result: TransferEntropyResult | None = None
        self._compute_kwargs: dict[str, Any] = {}
        self._data_fingerprint: tuple | None = None
        if engine is not None:
            self._data_fingerprint = self._fingerprint(
                engine._data, engine._variable_names, engine._sources,
            )

    @staticmethod
    def _fingerprint(data, variable_names, sources) -> tuple:
        """Identity key for (data, variable_names, sources).

        Uses ndarray memory address + shape + dtype: cheap, avoids hashing
        large arrays, and reliably detects when the caller swaps in a
        different buffer.
        """
        def _key(arr):
            if arr is None:
                return None
            a = np.asarray(arr)
            return (a.ctypes.data, a.shape, str(a.dtype))
        return (_key(data), _key(variable_names), _key(sources))

    def configure(self, **kwargs) -> "Pipeline":
        """Set kwargs forwarded to ``engine.compute()`` on first access.

        Callable multiple times; later calls update earlier ones.  Any
        change to the compute configuration invalidates the cached
        ``te_result``.  Returns ``self`` for chaining.
        """
        changed = any(
            k not in self._compute_kwargs or self._compute_kwargs[k] != v
            for k, v in kwargs.items()
        )
        self._compute_kwargs.update(kwargs)
        if changed and self._te_result is not None:
            self._te_result = None
        return self

    def _ensure_engine(
        self,
        data: np.ndarray | None,
        variable_names: np.ndarray | None,
        sources: np.ndarray | None,
    ) -> None:
        """Build or reuse the underlying engine for the given data.

        When the incoming ``(data, variable_names, sources)`` matches the
        current engine's buffers, reuse — preserving bin/pair caches.
        Otherwise construct a fresh engine and drop the cached TE result.
        """
        if data is None:
            if self.engine is None:
                raise ValueError(
                    "Pipeline has no engine; pass data and variable_names "
                    "to run() (or construct Pipeline with a pre-built engine)."
                )
            return

        if variable_names is None:
            raise ValueError(
                "variable_names is required when data is provided."
            )

        fp = self._fingerprint(data, variable_names, sources)
        if self.engine is not None and fp == self._data_fingerprint:
            return

        self.engine = TransferEntropyEngine(
            data=data,
            variable_names=variable_names,
            sources=sources,
        )
        self._data_fingerprint = fp
        self._te_result = None

    @property
    def te_result(self) -> TransferEntropyResult:
        """Cached TE matrix; lazily computed on first access."""
        if self._te_result is None:
            self._te_result = self.engine.compute(**self._compute_kwargs)
        return self._te_result

    def clear_cache(self) -> None:
        """Discard the cached TE matrix; next access will recompute."""
        self._te_result = None

    def infer(self, method: str, taus: list[int] | None = None, lag_combine: str = "max", **kwargs) -> Any:
        """Run a single inference method on the cached TE matrix.

        Parameters
        ----------
        method : one of ``"fdr"``, ``"clr"``, ``"nd"``, ``"surrogate_test"``,
            ``"point"``.
        taus : list[int] or None
            If given, run inference at each tau and combine per-pair
            (see :meth:`_infer_multi_lag`).  The caller must choose the
            lag values explicitly; no automatic selection is performed,
            and only ``surrogate_test`` is currently supported.
        lag_combine : str
            Combination strategy when ``taus`` is given: ``"max"`` (default),
            ``"adaptive_select"``, or ``"weighted_sum"``.  All strategies
            apply Bonferroni correction across the K lags.
        **kwargs : method-specific parameters. Any key set at ``Pipeline``
            construction as a default is inherited unless overridden here.
        """
        if taus is not None:
            return self._infer_multi_lag(
                taus=taus, combine=lag_combine, inference_method=method, **kwargs
            )

        params = {**self._defaults, **kwargs}
        method_obj = get_method(method)
        te = self.te_result

        device = params.pop("device", None)
        if device is None:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        # Auto-fill params expected by fused methods. The matrix-based
        # methods (fdr/clr/nd) only consume the TE matrix itself.
        if method in _FUSED_METHODS:
            params.setdefault("bin_data", te.bin_arrs)
            params.setdefault("n_per_var", te.n_per_var)
            params.setdefault("b_max", te.b_max)
            params.setdefault("kernel", te.kernel)
            params.setdefault("tau", te.tau)

        # Forward the engine's source filter so the inference methods restrict
        # thresholding and BH correction to the computed source-target pairs
        # rather than treating the uncomputed (structurally zero) entries as data.
        _engine_sources = getattr(self.engine, "_sources", None)
        if _engine_sources is not None:
            params.setdefault("sources", _engine_sources)

        return method_obj.infer(
            te.matrix, te.variable_names, device, **params,
        )

    def _infer_multi_lag(
        self,
        taus: list[int],
        combine: str = "max",
        inference_method: str = "surrogate_test",
        **kwargs,
    ) -> Any:
        """Internal: run inference across multiple time-lags and combine per-pair.

        For each ``tau`` in ``taus``, the TE matrix is recomputed and the
        chosen inference method is run.  The per-tau results are then
        combined into a single :class:`SurrogateTestResult`.

        Combination strategies
        ----------------------
        max (default)
            Per pair, pick the lag with the largest ``effective_te``.
            Empirically the most robust choice on synthetic benchmarks.
        adaptive_select
            Per pair, pick the lag with the smallest p-value.  Can be
            aggressive: a marginally smaller p may come with weaker TE.
        weighted_sum
            Literal weighted sum of ``effective_te`` across lags,
            weights ``-log(p)`` truncated to zero above p≥0.05.
            Pairs with no significant lag contribute zero and are
            dropped by the final ``effective_te > 0`` filter; the whole
            call raises if no pair is significant at any lag.

        All three strategies report a **Bonferroni-corrected** p-value
        (``K * min p``) so the multi-lag selection does not inflate the
        FDR budget relative to a single-lag run.
        """
        if not taus:
            raise ValueError("taus must be a non-empty list of time-lags")
        if combine not in ("adaptive_select", "weighted_sum", "max"):
            raise ValueError(
                f"combine must be 'adaptive_select', 'weighted_sum', or 'max' "
                f"(got {combine!r})"
            )
        if inference_method != "surrogate_test":
            raise NotImplementedError(
                f"multi_lag combine for {inference_method!r} is not supported; "
                f"only 'surrogate_test' is implemented"
            )

        per_tau_results: list[Any] = []
        original_tau = self._compute_kwargs.get("tau", 1)

        for tau in taus:
            self.configure(tau=tau)
            self.clear_cache()
            vprint(f"[TENEX] multi_lag: computing tau={tau}")
            per_tau_results.append(self.infer(inference_method, **kwargs))

        # Restore original tau (clears the cache too, via configure)
        self.configure(tau=original_tau)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return self._combine_surrogate_multi_lag(
            per_tau_results, taus, combine,
            variable_names=self.engine.variable_names,
            device=device,
        )

    @staticmethod
    def _combine_surrogate_multi_lag(
        results: list[Any],
        taus: list[int],
        combine: str,
        variable_names: np.ndarray,
        device: torch.device,
    ) -> "SurrogateTestResult":
        """Combine per-tau SurrogateTestResults into one SurrogateTestResult."""
        from tenex.inference import bh_fdr_gpu, make_grn
        from tenex.inference.surrogate_test import SurrogateTestResult

        n_vars = results[0].observed_te.shape[0]
        variable_names = np.asarray(variable_names)
        K = len(taus)

        # Stack per-tau matrices: (K, n, n)
        obs_stack = np.stack([r.observed_te for r in results], axis=0)
        eff_stack = np.stack([r.effective_te for r in results], axis=0)
        mean_stack = np.stack([r.mean_surrogate_te for r in results], axis=0)
        std_stack = np.stack([r.std_surrogate_te for r in results], axis=0)
        pval_stack = np.stack([r.p_values for r in results], axis=0)

        if combine == "max":
            sel_idx = np.argmax(eff_stack, axis=0, keepdims=True)
            effective_te = np.take_along_axis(eff_stack, sel_idx, axis=0)[0]
            observed_te = np.take_along_axis(obs_stack, sel_idx, axis=0)[0]
            mean_surr = np.take_along_axis(mean_stack, sel_idx, axis=0)[0]
            std_surr = np.take_along_axis(std_stack, sel_idx, axis=0)[0]

        elif combine == "adaptive_select":
            sel_idx = np.argmin(pval_stack, axis=0, keepdims=True)
            effective_te = np.take_along_axis(eff_stack, sel_idx, axis=0)[0]
            observed_te = np.take_along_axis(obs_stack, sel_idx, axis=0)[0]
            mean_surr = np.take_along_axis(mean_stack, sel_idx, axis=0)[0]
            std_surr = np.take_along_axis(std_stack, sel_idx, axis=0)[0]

        else:  # weighted_sum
            eps = 1e-12
            weights = np.where(
                pval_stack < 0.05, -np.log(pval_stack + eps), 0.0,
            ).astype(np.float32)
            if not np.any(weights > 0):
                raise ValueError(
                    f"weighted_sum: no pair passed p<0.05 at any lag in taus={taus}; "
                    f"the combined result would be identically zero. "
                    f"Loosen n_surrogates, add more lags, or switch to combine='max'."
                )
            # Literal sum — unlike a weighted average, pairs with no
            # significant lag naturally drop to zero and are rejected by
            # the final ``effective_te > 0`` filter.
            effective_te = np.sum(eff_stack * weights, axis=0)
            # Diagnostic fields: take the lag with the largest weight
            # (argmax of -log p → argmin of p).  Tie-broken deterministically
            # by np.argmin.
            sel_idx = np.argmin(pval_stack, axis=0, keepdims=True)
            observed_te = np.take_along_axis(obs_stack, sel_idx, axis=0)[0]
            mean_surr = np.take_along_axis(mean_stack, sel_idx, axis=0)[0]
            std_surr = np.take_along_axis(std_stack, sel_idx, axis=0)[0]

        # Bonferroni across the K lags (conservative but valid under dependence).
        p_values = np.minimum(1.0, pval_stack.min(axis=0) * K).astype(np.float32)
        np.fill_diagonal(p_values, 1.0)

        # BH-FDR on off-diagonal pairs only
        fdr = results[0].fdr
        off_diag = ~np.eye(n_vars, dtype=bool)
        p_off = p_values[off_diag]
        pvals_t = torch.from_numpy(p_off).to(device)
        fdr_adj = bh_fdr_gpu(pvals_t, alpha=fdr).cpu().numpy()
        sig_full = np.zeros((n_vars, n_vars), dtype=bool)
        sig_full[off_diag] = fdr_adj < fdr
        sig_full &= (effective_te > 0)

        pairs = np.argwhere(sig_full).astype(np.int64)
        te_values = effective_te[pairs[:, 0], pairs[:, 1]]
        grn = make_grn(variable_names, pairs, te_values)
        vprint(f"[TENEX] multi_lag ({combine}, taus={taus}): "
              f"{len(grn)} significant edges (FDR<{fdr})")

        first = results[0]
        return SurrogateTestResult(
            observed_te=observed_te.astype(np.float32),
            mean_surrogate_te=mean_surr.astype(np.float32),
            std_surrogate_te=std_surr.astype(np.float32),
            effective_te=effective_te.astype(np.float32),
            p_values=p_values,
            grn=grn,
            n_surrogates=first.n_surrogates,
            shuffle_method=first.shuffle_method,
            block_length=first.block_length,
            p_method=first.p_method,
            fdr=fdr,
        )

    def run(
        self,
        data: np.ndarray | None = None,
        variable_names: np.ndarray | None = None,
        sources: np.ndarray | None = None,
        methods: list[str] | None = None,
        method_kwargs: dict[str, dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """Run multiple inference methods, returning a composite result.

        Parameters
        ----------
        data : (n_vars, T) ndarray, optional
            Multivariate time series.  Rows must be aligned with
            ``variable_names``.  When omitted, reuses the engine passed
            at construction (or the engine built by a previous call).
        variable_names : (n_vars,) str ndarray, optional
            Labels aligned with the rows of ``data``.  Required whenever
            ``data`` is given.
        sources : (k,) str ndarray, optional
            Subset of ``variable_names`` to use as TE sources.  When
            ``None`` all pairs are computed.
        methods : list of method names.
        method_kwargs : optional per-method kwargs (e.g.
            ``{"surrogate_test": {"n_surrogates": 100}}``).
        verbose : print [TENEX] status messages during the run (default False).
        """
        set_verbose(verbose)
        if methods is None:
            raise ValueError("methods is required")
        self._ensure_engine(data, variable_names, sources)

        method_kwargs = method_kwargs or {}
        # Forward the pipeline-level verbosity into the (possibly lazy)
        # engine.compute() so its [TENEX] status messages honor this call
        # instead of being reset by compute()'s own verbose default.
        self._compute_kwargs["verbose"] = verbose
        _ = self.te_result  # ensure TE is computed before the inference loop

        results: dict[str, Any] = {}
        for m in methods:
            results[m] = self.infer(m, **method_kwargs.get(m, {}))

        return PipelineResult(
            te_result=self._te_result,
            inference_results=results,
        )
