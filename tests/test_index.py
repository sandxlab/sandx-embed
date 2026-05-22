"""Tests for sandx_embed.index."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from sandx_embed.index import SearchResult, VectorIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit_vectors(n: int, d: int, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    ids = [f"id_{i}" for i in range(n)]
    return vecs, ids


# ---------------------------------------------------------------------------
# Exact index
# ---------------------------------------------------------------------------

class TestExactIndex:
    def test_build_and_query_returns_k_results(self):
        vecs, ids = _random_unit_vectors(50, 16)
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(vecs, ids)
        result = idx.query(vecs[0], k=5)
        assert len(result.ids) == 5
        assert len(result.distances) == 5

    def test_query_self_is_nearest(self):
        vecs, ids = _random_unit_vectors(20, 16)
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(vecs, ids)
        result = idx.query(vecs[3], k=1)
        assert result.ids[0] == "id_3"

    def test_distances_non_negative_cosine(self):
        vecs, ids = _random_unit_vectors(30, 8)
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(vecs, ids)
        result = idx.query(vecs[0], k=10)
        assert all(d >= -1e-5 for d in result.distances)

    def test_k_capped_at_n(self):
        vecs, ids = _random_unit_vectors(5, 8)
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(vecs, ids)
        result = idx.query(vecs[0], k=100)
        assert len(result.ids) == 5

    def test_l2_metric(self):
        vecs, ids = _random_unit_vectors(20, 8)
        idx = VectorIndex(method="exact", metric="l2")
        idx.build(vecs, ids)
        result = idx.query(vecs[0], k=1)
        assert result.ids[0] == "id_0"
        assert result.distances[0] < 1e-5

    def test_build_requires_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            idx = VectorIndex(method="exact")
            idx.build(np.zeros(8), ["x"])

    def test_build_id_length_mismatch(self):
        vecs, ids = _random_unit_vectors(5, 8)
        with pytest.raises(ValueError, match="len\\(ids\\)"):
            VectorIndex(method="exact").build(vecs, ids[:3])

    def test_query_before_build_raises(self):
        idx = VectorIndex(method="exact")
        with pytest.raises(RuntimeError, match="empty"):
            idx.query(np.zeros(8))

    def test_save_load_roundtrip(self):
        vecs, ids = _random_unit_vectors(20, 8)
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(vecs, ids)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/test_idx"
            idx.save(path)
            idx2 = VectorIndex.load(path)

        r1 = idx.query(vecs[5], k=3)
        r2 = idx2.query(vecs[5], k=3)
        assert r1.ids == r2.ids

    def test_len(self):
        vecs, ids = _random_unit_vectors(10, 4)
        idx = VectorIndex(method="exact")
        idx.build(vecs, ids)
        assert len(idx) == 10


# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------

class TestHNSWIndex:
    def test_build_and_query(self):
        vecs, ids = _random_unit_vectors(100, 32)
        idx = VectorIndex(method="hnsw", metric="cosine")
        idx.build(vecs, ids)
        result = idx.query(vecs[0], k=5)
        assert len(result.ids) == 5
        # Top-1 should be self (cosine distance ≈ 0)
        assert result.ids[0] == "id_0"

    def test_save_load_roundtrip(self):
        vecs, ids = _random_unit_vectors(50, 16)
        idx = VectorIndex(method="hnsw", metric="cosine")
        idx.build(vecs, ids)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/hnsw_idx"
            idx.save(path)
            idx2 = VectorIndex.load(path)

        r1 = idx.query(vecs[10], k=5)
        r2 = idx2.query(vecs[10], k=5)
        assert r1.ids == r2.ids

    def test_repr(self):
        vecs, ids = _random_unit_vectors(10, 4)
        idx = VectorIndex(method="hnsw", metric="cosine")
        idx.build(vecs, ids)
        r = repr(idx)
        assert "hnsw" in r
        assert "n=10" in r


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

def test_search_result_len():
    result = SearchResult(ids=["a", "b", "c"], distances=[0.1, 0.2, 0.3])
    assert len(result) == 3
