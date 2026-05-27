# sandx-embed

**Shared embedding and vector similarity infrastructure for the SandX platform.**

[![CI](https://github.com/sandxlab/sandx-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/sandxlab/sandx-embed/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-embed` is the shared latent representation layer used by all SandX engines. It provides:

- **Pluggable encoders** — sentence-transformers models out of the box; register any custom encoder
- **High-performance ANN indexing** — HNSW (production) and exact search (baseline), with save/load
- **Cross-domain similarity** — cosine, L2, inner product; normalized and unnormalized vectors

**Not a standalone product** — consumed by `sandx-er`, `sandx-graph`, and `sandx-compute` as a shared dependency.

## Status

> **v0.1 — Phase 2 active development**

| Component | Status |
|-----------|--------|
| `Encoder` — pluggable model registry | **Working** |
| `SentenceTransformerEncoder` — SBERT, E5, BGE | **Working** |
| `VectorIndex` — HNSW and exact search | **Working** |
| Save / load index | **Working** |
| PyPI package | **Working** |

## Installation

```bash
pip install sandx-embed
```

Or from source:

```bash
git clone https://github.com/sandxlab/sandx-embed
cd sandx-embed
pip install -e ".[dev]"
```

## Quick Start

```python
from sandx_embed import Encoder, VectorIndex

# Encode records into dense vectors
enc = Encoder(model="sentence-bert")   # downloads all-MiniLM-L6-v2 on first use
vectors = enc.encode(["John Smith, Boston", "Jon Smyth, Boston", "Alice Brown, NYC"])
# → np.ndarray shape (3, 384), L2-normalized

# Build an ANN index
idx = VectorIndex(method="hnsw", metric="cosine")
idx.build(vectors, ids=["r0", "r1", "r2"])

# Query nearest neighbors
result = idx.query(vectors[0], k=2)
print(result.ids)        # ["r0", "r1"]
print(result.distances)  # [0.0, 0.12]  (cosine distance)

# Persist and reload
idx.save("/tmp/my_index")
idx2 = VectorIndex.load("/tmp/my_index")
```

## Built-in Models

| Name | HuggingFace model | Dim | Notes |
|------|-------------------|-----|-------|
| `"sentence-bert"` | `all-MiniLM-L6-v2` | 384 | Fast, English, recommended default |
| `"e5-small"` | `intfloat/e5-small-v2` | 384 | Higher quality, English |
| `"bge-m3"` | `BAAI/bge-m3` | 1024 | Multilingual, large |

## Custom Encoders

```python
from sandx_embed.encoder import BaseEncoder, Encoder
import numpy as np

class MyEncoder(BaseEncoder):
    def encode(self, inputs, *, batch_size=64, normalize=True):
        # your model here
        return np.random.rand(len(inputs), 128).astype(np.float32)
    @property
    def dim(self): return 128

Encoder.register("my-model", lambda: MyEncoder())
enc = Encoder("my-model")
```

## Index Methods

| Method | Backend | When to use |
|--------|---------|-------------|
| `"hnsw"` | usearch | N > 10,000; production; fast queries |
| `"exact"` | numpy | Small datasets; correctness baseline |

## Design Principles

- **Pluggable** — any encoder model or index backend can be registered
- **Portable** — indexes serialize to disk and reload without rebuilding
- **Deterministic** — same model version + input → same output
- **No vendor lock-in** — no hard dependency on any hosted vector service

## Related

- [`sandx-er`](https://github.com/sandxlab/sandx-er) — entity resolution engine (uses sandx-embed for blocking + matching)
- [`sandx-graph`](https://github.com/sandxlab/sandx-graph) — graph intelligence over resolved entities
- [sandx.io](https://sandx.io) — project home

## License

Apache 2.0 — see [LICENSE](LICENSE)
