"""
GRN — Gene Regulatory Network data structure.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GRN:
    """Gene regulatory network edges."""
    source: np.ndarray      # (n_edges,) str — source gene names
    target: np.ndarray      # (n_edges,) str — target gene names
    te: np.ndarray           # (n_edges,) float32 — edge scores
    pairs: np.ndarray        # (n_edges, 2) int32 — (source_idx, target_idx)

    def __len__(self):
        return len(self.te)

    def to_sif(self) -> np.ndarray:
        """Convert to SIF-format (n_edges, 3) string array."""
        return np.stack((self.source, self.te.astype(str), self.target), axis=1)

    def to_edge_list(self) -> list[tuple[str, str, float]]:
        """Convert to list of (source, target, score) tuples."""
        return [(s, t, float(v)) for s, t, v in zip(self.source, self.target, self.te)]
