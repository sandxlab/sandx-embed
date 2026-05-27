"""sandx-embed — Shared embedding and vector similarity infrastructure."""

from sandx_embed.encoder import BaseEncoder, Encoder, SentenceTransformerEncoder
from sandx_embed.index import SearchResult, VectorIndex

__version__ = "0.1.1"
__all__ = [
    "BaseEncoder",
    "Encoder",
    "SentenceTransformerEncoder",
    "SearchResult",
    "VectorIndex",
]
