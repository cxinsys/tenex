"""
NetWeaver — unified wrapper for link inference from a TE matrix.

Dispatches to inference methods registered in tenex.inference.
Available methods: 'fdr', 'clr', 'nd', 'trace', 'point', 'surrogate_test'.

'trace' is the shipping marginal-TE method (descended from TENET).
'point' is reserved for the Julian Lee 2025 paper-exact procedure and
currently raises NotImplementedError.
'surrogate_test' performs a time-axis shuffle surrogate test yielding
effective-TE values and BH-FDR-thresholded edges.
"""

import numpy as np
import torch

from tenex.inference import GRN, get_method, available_methods
from tenex._log import set_verbose


class NetWeaver:
    """Infer network links from a TE matrix."""

    def __init__(
        self,
        result,
        sources: np.ndarray | list | None = None,
        fdr: float = 0.01,
        links: int = 0,
        is_trimming: bool = True,
        trim_threshold: float = 0.0,
    ):
        """
        Parameters
        ----------
        result        : TransferEntropyResult from engine.compute(), or
                        (matrix, variable_names) tuple.
        sources       : optional list of source variable names.
        fdr           : FDR threshold (default 0.01). Used when links=0.
        links         : number of top links to keep. 0 = use FDR instead.
        is_trimming   : if True, apply DPI trimming (FDR method only).
        trim_threshold: additive threshold for the DPI criterion.
        """
        from tenex.result import TransferEntropyResult

        if isinstance(result, TransferEntropyResult):
            self.result_matrix = result.matrix
            self.variable_names = result.variable_names
            self.bin_arrs = result.bin_arrs
            self.n_per_var = result.n_per_var
            self.b_max = result.b_max
            self.kernel = result.kernel
            self.tau = result.tau
        elif isinstance(result, tuple) and len(result) == 2:
            # Legacy: (matrix, variable_names) tuple
            self.result_matrix = np.asarray(result[0], dtype=np.float32)
            self.variable_names = np.asarray(result[1])
            self.bin_arrs = None
            self.n_per_var = None
            self.b_max = None
            self.kernel = None
            self.tau = 1
        else:
            raise TypeError(
                "result must be a TransferEntropyResult or "
                "(matrix, variable_names) tuple"
            )

        # Inherit the source (TF) filter from the result when not given explicitly,
        # so matrix-based inference restricts itself to the computed pairs.
        self.sources = sources if sources is not None else getattr(result, 'sources', None)
        self.fdr = fdr
        self.links = links
        self.is_trimming = is_trimming
        self.trim_threshold = trim_threshold

    def infer(
        self,
        method: str = 'fdr',
        device: str | torch.device | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> GRN | tuple[GRN, GRN]:
        """
        Run link inference.

        Parameters
        ----------
        method : 'fdr', 'clr', 'nd', 'trace', or 'point'.
        device : torch device (e.g. 'cuda:0'). None = auto.
        verbose : print [TENEX] status messages during the run (default False).
        **kwargs : additional method-specific parameters.
                   For 'fdr': batch_size.
                   For 'trace': n_surrogates, significance.
                     (bin_data and tau are auto-filled from result if available)
                   'point' currently raises NotImplementedError.

        Returns
        -------
        Method-dependent:
          'fdr' with trimming : (grn, trimmed_grn)
          'fdr' without       : grn
          'clr', 'nd'         : grn
          'trace'             : TRACEResult (with .outte, .inte, .grn, etc.)
        """
        set_verbose(verbose)
        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        method_obj = get_method(method)

        # Pass constructor params as kwargs
        params = dict(
            fdr=self.fdr,
            links=self.links,
            sources=self.sources,
            is_trimming=self.is_trimming,
            trim_threshold=self.trim_threshold,
        )

        # Auto-fill TRACE params from TransferEntropyResult.
        # TRACE does not consume fdr/links/sources — drop them from the
        # NetWeaver-provided defaults so the caller doesn't get spurious
        # "unused kwarg" rejections.
        if method == 'trace':
            if 'bin_data' not in kwargs and self.bin_arrs is not None:
                ba = self.bin_arrs
                # TRACE requires K=1; refuse multi-kernel data up-front.
                if ba.ndim == 3 and ba.shape[2] != 1:
                    raise ValueError(
                        f"TRACE requires K=1 bin arrays; the result has shape "
                        f"{ba.shape}. Re-run engine.compute() with a K=1 binning "
                        f"method (e.g. FSBW-L) before passing to TRACE."
                    )
                params['bin_data'] = ba
            if 'tau' not in kwargs:
                params['tau'] = self.tau
            for _drop in ('fdr', 'links', 'sources',
                          'is_trimming', 'trim_threshold'):
                params.pop(_drop, None)

        # Auto-fill surrogate_test params from TransferEntropyResult.
        if method == 'surrogate_test':
            if 'bin_data' not in kwargs and self.bin_arrs is not None:
                params['bin_data'] = self.bin_arrs
            if 'n_per_var' not in kwargs and self.n_per_var is not None:
                params['n_per_var'] = self.n_per_var
            if 'b_max' not in kwargs and self.b_max is not None:
                params['b_max'] = self.b_max
            if 'kernel' not in kwargs and self.kernel:
                params['kernel'] = self.kernel
            if 'tau' not in kwargs:
                params['tau'] = self.tau
            # surrogate_test consumes 'fdr' and 'sources' from NetWeaver defaults;
            # 'sources' restricts the BH correction to the computed source pairs.
            for _drop in ('links', 'is_trimming', 'trim_threshold'):
                params.pop(_drop, None)

        params.update(kwargs)

        return method_obj.infer(
            self.result_matrix, self.variable_names, device, **params,
        )
