"""Tests for sandx_embed.encoder."""

from __future__ import annotations

import numpy as np
import pytest

from sandx_embed.encoder import BaseEncoder, Encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FixedEncoder(BaseEncoder):
    """Deterministic test encoder — no model download needed."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def encode(self, inputs, *, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        out = []
        for inp in inputs:
            seed = abs(hash(str(inp))) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(v)
                v = v / (norm + 1e-8)
            out.append(v)
        return np.array(out, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------

def test_register_and_use_custom_encoder():
    Encoder.register("test-fixed", lambda: FixedEncoder(dim=8))
    enc = Encoder("test-fixed")
    vecs = enc.encode(["hello", "world"])
    assert vecs.shape == (2, 8)
    assert vecs.dtype == np.float32


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown encoder"):
        Encoder("does-not-exist").encode(["x"])


def test_custom_encoder_dim():
    Encoder.register("test-dim16", lambda: FixedEncoder(dim=16))
    enc = Encoder("test-dim16")
    assert enc.dim == 16


def test_encode_output_shape():
    Encoder.register("test-shape", lambda: FixedEncoder(dim=4))
    enc = Encoder("test-shape")
    vecs = enc.encode(["a", "b", "c"])
    assert vecs.shape == (3, 4)


def test_encode_normalize_true_produces_unit_vectors():
    Encoder.register("test-norm", lambda: FixedEncoder(dim=32))
    enc = Encoder("test-norm")
    vecs = enc.encode(["foo", "bar", "baz"], normalize=True)
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_encode_deterministic():
    Encoder.register("test-det", lambda: FixedEncoder(dim=8))
    enc = Encoder("test-det")
    v1 = enc.encode(["same input"])
    v2 = enc.encode(["same input"])
    np.testing.assert_array_equal(v1, v2)


def test_encode_single_item():
    Encoder.register("test-single", lambda: FixedEncoder(dim=8))
    enc = Encoder("test-single")
    vecs = enc.encode(["only one"])
    assert vecs.shape == (1, 8)


# ---------------------------------------------------------------------------
# SentenceTransformerEncoder — skipped unless network/model available
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sentence_bert_encode():
    enc = Encoder("sentence-bert")
    vecs = enc.encode(["Entity resolution is the task of determining which records refer to the same entity."])
    assert vecs.shape[0] == 1
    assert vecs.shape[1] == 384
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
