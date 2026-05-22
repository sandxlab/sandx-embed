"""VectorIndex — approximate nearest-neighbor index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


IndexMethod = Literal["hnsw", "faiss-flat", "faiss-ivf", "exact"]


@dataclass
class SearchResult:
    ids: list[str]
    distances: list[float]


class VectorIndex:
    """ANN index over dense float32 vectors.

    Supports HNSW (default), FAISS, and exact search backends.

    Args:
        method:  Index construction method.
        metric:  Distance metric — "cosine", "l2", or "ip" (inner product).
    """

    def __init__(
        self,
        method: IndexMethod = "hnsw",
        metric: Literal["cosine", "l2", "ip"] = "cosine",
    ) -> None:
        self.method = method
        self.metric = metric
        self._index: object | None = None
        self._ids: list[str] = []

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Build the index from a (N, D) float32 matrix and corresponding IDs."""
        raise NotImplementedError("Phase 2")

    def query(self, vector: np.ndarray, k: int = 10) -> SearchResult:
        """Return the k nearest neighbors of a query vector."""
        raise NotImplementedError("Phase 2")

    def save(self, path: str) -> None:
        """Serialize the index to disk."""
        raise NotImplementedError("Phase 2")

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Deserialize an index from disk."""
        raise NotImplementedError("Phase 2")
