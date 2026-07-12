"""
POINT — Pruned Outgoing and INcoming Transfer entropy (Lee 2025, exact).

Reference
---------
Julian Lee, "Identifying key drivers in a stochastic dynamical system
through estimation of transfer entropy between univariate and multivariate
time series," Physical Review E 111, 024308 (2025).

Status
------
Not yet implemented. The paper's procedure — conditional-MI forward
selection + conditional backward elimination + final joint TE significance
test — is a different algorithm from the marginal-filter method currently
shipped as ``TRACE`` (``tenex.inference.trace``). The shipping heuristic
was renamed to TRACE in 2026-04 so this name can be reserved for a
paper-exact implementation.

Usage
-----
Until this module is implemented, ``NetWeaver.infer(method='point')``
raises NotImplementedError. Use ``method='trace'`` for the
currently-available fast marginal inference.
"""

from tenex.inference import InferenceMethod


class POINTMethod(InferenceMethod):
    """Placeholder for the Lee 2025 POINT procedure (exact).

    When implemented, this class will run:

        for each target Y:
            # Forward: conditional-MI greedy addition with surrogate test
            # Backward: conditional-MI elimination of redundant members
            # Final: joint TE significance test of selected set

    and return a POINTResult with paper-exact OutTE / InTE values.
    """

    @property
    def name(self) -> str:
        return 'point'

    def infer(self, te_matrix, variable_names, device, **kwargs):
        raise NotImplementedError(
            "method='point' (Lee 2025 exact procedure) is not yet implemented. "
            "For the currently-available fast marginal inference use "
            "method='trace'. Full POINT is planned as a CUDA-kernel-optimised "
            "reimplementation."
        )
