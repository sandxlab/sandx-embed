"""sandx-embed — Shared embedding and vector similarity infrastructure."""

from sandx_embed.encoder import Encoder
from sandx_embed.index import VectorIndex

__version__ = "0.1.0.dev0"
__all__ = ["Encoder", "VectorIndex"]
