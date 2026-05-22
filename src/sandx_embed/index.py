"""VectorIndex — approximate and exact nearest-neighbor index."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np


IndexMethod = Literal["hnsw", "exact"]
Metric = Literal["cosine", "l2", "ip"]


@dataclass
class SearchResult:
    """Result of a k-nearest-neighbor query."""

    ids: list[str]
    distances: list[float]

    def __len__(self) -> int:
        return len(self.ids)


class VectorIndex:
    """ANN index over dense float32 vectors.

    Supported methods:
        "hnsw"  — Hierarchical Navigable Small World (hnswlib). Fast, production-grade.
                  Recommended for N > 10 000.
        "exact" — Brute-force numpy search. No approximation error.
                  Use for small datasets or as a correctness baseline.

    Args:
        method: Index construction method.
        metric: Distance metric — "cosine", "l2", or "ip" (inner product).

    Usage:
        idx = VectorIndex(method="hnsw", metric="cosine")
        idx.build(vectors, ids=["a", "b", "c"])
        result = idx.query(query_vec, k=5)
        idx.save("/tmp/my_index")
        idx2 = VectorIndex.load("/tmp/my_index")
    """

    def __init__(self, method: IndexMethod = "hnsw", metric: Metric = "cosine") -> None:
        self.method = method
        self.metric = metric
        self._index: object | None = None  # hnswlib.Index or np.ndarray
        self._ids: list[str] = []
        self._dim: int = 0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, ids: list[str]) -> None:
        """Index the given vectors.

        Args:
            vectors: Float32 matrix of shape (N, D). If metric is "cosine",
                     vectors should be L2-normalized beforehand (Encoder does
                     this automatically when normalize=True).
            ids:     String ID for each row in vectors. Must be length N.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        n, d = vectors.shape
        if len(ids) != n:
            raise ValueError(f"len(ids)={len(ids)} must equal vectors.shape[0]={n}")

        self._ids = list(ids)
        self._dim = d

        if self.method == "hnsw":
            self._build_hnsw(vectors)
        elif self.method == "exact":
            self._index = vectors.copy()
        else:
            raise ValueError(f"Unknown index method: {self.method!r}")

    def _build_hnsw(self, vectors: np.ndarray) -> None:
        from usearch.index import Index  # type: ignore[import]

        # usearch metric names differ from our public API names
        _metric_map = {"cosine": "cos", "l2": "l2sq", "ip": "ip"}
        n = vectors.shape[0]
        idx = Index(ndim=self._dim, metric=_metric_map[self.metric], dtype="f32")
        keys = np.arange(n, dtype=np.int64)
        idx.add(keys, vectors)
        self._index = idx

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, vector: np.ndarray, k: int = 10) -> SearchResult:
        """Return the k nearest neighbors of a query vector.

        Args:
            vector: Float32 vector of shape (D,) or (1, D).
            k:      Number of neighbors to return.

        Returns:
            SearchResult with ids and distances lists of length min(k, N).
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call build() first.")

        vec = np.asarray(vector, dtype=np.float32).ravel()
        k = min(k, len(self._ids))

        if self.method == "hnsw":
            return self._query_hnsw(vec, k)
        else:
            return self._query_exact(vec, k)

    def _query_hnsw(self, vec: np.ndarray, k: int) -> SearchResult:
        matches = self._index.search(vec, count=k)  # type: ignore[union-attr]
        ids = [self._ids[int(key)] for key in matches.keys]
        dists = matches.distances.tolist()
        # usearch l2sq returns squared distances; take sqrt to match convention
        if self.metric == "l2":
            import math
            dists = [math.sqrt(max(0.0, d)) for d in dists]
        return SearchResult(ids=ids, distances=dists)

    def _query_exact(self, vec: np.ndarray, k: int) -> SearchResult:
        matrix = np.asarray(self._index)
        if self.metric == "cosine":
            sims = matrix @ vec
            order = np.argsort(-sims)[:k]
            dists = (1.0 - sims[order]).tolist()
        elif self.metric == "l2":
            diffs = matrix - vec
            d = np.linalg.norm(diffs, axis=1)
            order = np.argsort(d)[:k]
            dists = d[order].tolist()
        else:  # ip
            scores = matrix @ vec
            order = np.argsort(-scores)[:k]
            dists = (-scores[order]).tolist()

        ids = [self._ids[int(i)] for i in order]
        return SearchResult(ids=ids, distances=dists)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the index to disk.

        Writes two files: ``{path}.json`` (metadata + IDs) and
        ``{path}.bin`` (HNSW) or ``{path}.npy`` (exact).
        """
        if self._index is None:
            raise RuntimeError("Index is empty. Call build() before save().")

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        meta = {
            "method": self.method,
            "metric": self.metric,
            "dim": self._dim,
            "ids": self._ids,
        }
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        if self.method == "hnsw":
            self._index.save(path + ".usearch")  # type: ignore[union-attr]
        else:
            np.save(path + ".npy", np.asarray(self._index))

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        """Deserialize an index from disk.

        Args:
            path: The same path prefix passed to save().
        """
        with open(path + ".json", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(method=meta["method"], metric=meta["metric"])
        obj._dim = meta["dim"]
        obj._ids = meta["ids"]

        if meta["method"] == "hnsw":
            from usearch.index import Index  # type: ignore[import]

            _metric_map = {"cosine": "cos", "l2": "l2sq", "ip": "ip"}
            idx = Index(ndim=meta["dim"], metric=_metric_map[meta["metric"]], dtype="f32")
            idx.load(path + ".usearch")
            obj._index = idx
        else:
            obj._index = np.load(path + ".npy")

        return obj

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return (
            f"VectorIndex(method={self.method!r}, metric={self.metric!r}, "
            f"n={len(self._ids)}, dim={self._dim})"
        )
