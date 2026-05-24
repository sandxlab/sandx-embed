"""Encoder — pluggable model registry for embedding raw inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


class BaseEncoder(ABC):
    """Abstract base for all SandX encoders."""

    @abstractmethod
    def encode(self, inputs: list[Any], *, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Encode inputs into a float32 matrix of shape (N, D)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


class SentenceTransformerEncoder(BaseEncoder):
    """Encoder backed by a sentence-transformers model.

    Args:
        model_name: HuggingFace model name or local path.
            Examples: "all-MiniLM-L6-v2", "intfloat/e5-small-v2", "BAAI/bge-m3"
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self._model = SentenceTransformer(model_name)
        get_dim = getattr(
            self._model, "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim_val: int = get_dim()

    def encode(self, inputs: list[Any], *, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        texts = [str(x) for x in inputs]
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim_val


# Built-in model name → factory function
_BUILTIN_MODELS: dict[str, Callable[[], BaseEncoder]] = {
    "sentence-bert": lambda: SentenceTransformerEncoder("all-MiniLM-L6-v2"),
    "e5-small":      lambda: SentenceTransformerEncoder("intfloat/e5-small-v2"),
    "bge-m3":        lambda: SentenceTransformerEncoder("BAAI/bge-m3"),
}


class Encoder:
    """Model-agnostic encoder with a pluggable backend registry.

    Built-in models (downloaded on first use):
        "sentence-bert"  — all-MiniLM-L6-v2 (384-dim, fast, English)
        "e5-small"       — intfloat/e5-small-v2 (384-dim, higher quality)
        "bge-m3"         — BAAI/bge-m3 (1024-dim, multilingual, large)

    Custom models can be registered:
        Encoder.register("my-model", lambda: MyEncoder())
        enc = Encoder("my-model")

    Usage:
        enc = Encoder("sentence-bert")
        vectors = enc.encode(["text a", "text b"])  # np.ndarray (2, 384)
    """

    _registry: dict[str, Callable[[], BaseEncoder]] = {}

    def __init__(self, model: str = "sentence-bert") -> None:
        self.model = model
        self._backend: BaseEncoder | None = None

    @classmethod
    def register(cls, name: str, factory: Callable[[], BaseEncoder]) -> None:
        """Register a custom encoder factory under the given name."""
        cls._registry[name] = factory

    def _load(self) -> BaseEncoder:
        all_models = {**_BUILTIN_MODELS, **self._registry}
        if self.model not in all_models:
            available = sorted(all_models)
            raise ValueError(
                f"Unknown encoder '{self.model}'. "
                f"Available built-ins: {available}. "
                "Register custom encoders with Encoder.register()."
            )
        return all_models[self.model]()

    def encode(self, inputs: list[Any], *, batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """Encode inputs to a float32 matrix of shape (N, D).

        Args:
            inputs:     List of strings (or any objects; non-strings are str()-cast).
            batch_size: Inference batch size.
            normalize:  L2-normalize output vectors (required for cosine similarity).
        """
        if self._backend is None:
            self._backend = self._load()
        return self._backend.encode(inputs, batch_size=batch_size, normalize=normalize)

    @property
    def dim(self) -> int:
        """Embedding dimensionality (triggers model load on first access)."""
        if self._backend is None:
            self._backend = self._load()
        return self._backend.dim
