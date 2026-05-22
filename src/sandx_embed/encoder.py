"""Encoder — pluggable model registry for embedding raw inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEncoder(ABC):
    """Abstract base for all SandX encoders."""

    @abstractmethod
    def encode(self, inputs: list[Any]) -> np.ndarray:
        """Encode inputs into a float32 matrix of shape (N, D)."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality."""
        ...


class Encoder:
    """Model-agnostic encoder with a pluggable backend registry.

    Usage:
        enc = Encoder(model="sentence-bert")
        vectors = enc.encode(texts)  # np.ndarray shape (N, D)

    Registered models (Phase 2):
        "sentence-bert"  — Sentence-BERT (all-MiniLM-L6-v2)
        "e5-small"       — E5 small multilingual
        "bge-m3"         — BGE-M3 multilingual
    """

    _registry: dict[str, type[BaseEncoder]] = {}

    def __init__(self, model: str = "sentence-bert") -> None:
        self.model = model
        self._backend: BaseEncoder | None = None

    @classmethod
    def register(cls, name: str, encoder_cls: type[BaseEncoder]) -> None:
        cls._registry[name] = encoder_cls

    def _load(self) -> BaseEncoder:
        if self.model not in self._registry:
            raise ValueError(
                f"Unknown encoder '{self.model}'. "
                f"Available: {list(self._registry)}. "
                "Custom encoders can be registered via Encoder.register()."
            )
        return self._registry[self.model]()

    def encode(self, inputs: list[Any]) -> np.ndarray:
        if self._backend is None:
            self._backend = self._load()
        return self._backend.encode(inputs)

    @property
    def dim(self) -> int:
        if self._backend is None:
            self._backend = self._load()
        return self._backend.dim
