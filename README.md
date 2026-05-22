# sandx-embed

**Shared embedding and vector similarity infrastructure for the SandX platform.**

Part of the [SandX Lab](https://github.com/sandxlab) computational infrastructure ecosystem.

---

## What It Does

`sandx-embed` is the shared latent representation layer used by all other SandX engines. It provides pluggable encoders, high-performance approximate nearest-neighbor (ANN) indexing, and cross-domain similarity computation.

**Not a standalone product** — consumed by `sandx-er`, `sandx-graph`, and `sandx-compute` as a shared dependency.

## Status

> **Phase 1 — Architecture & Foundations**

| Component | Status |
|-----------|--------|
| `sandx_embed.encoder` — pluggable encoder registry | Skeleton |
| `sandx_embed.index` — ANN index (HNSW, FAISS) | Skeleton |
| `sandx_embed.similarity` — cosine, L2, inner product | Skeleton |
| Python SDK on PyPI | Planned (Phase 2) |

## Design Principles

- **Pluggable encoders** — any model can be registered; the interface is model-agnostic
- **Index portability** — indexes are serializable and reloadable without rebuilding
- **Deterministic** — same model version + input → same output
- **No vendor lock-in** — no hard dependency on any single vector database or embedding provider

## Quick Start (planned API)

```python
from sandx_embed import Encoder, VectorIndex

# encode records into dense vectors
enc = Encoder(model="sentence-bert")
vectors = enc.encode(texts)

# build an ANN index
idx = VectorIndex(method="hnsw")
idx.build(vectors, ids=record_ids)

# query nearest neighbors
neighbors = idx.query(query_vector, k=10)
```

## Related

- [`sandx-er`](https://github.com/sandxlab/sandx-er) — consumes sandx-embed for blocking and matching
- [`sandx-graph`](https://github.com/sandxlab/sandx-graph) — consumes sandx-embed for node representations
- [`sandx-compute`](https://github.com/sandxlab/sandx-compute) — distributed compute orchestration

## License

Apache 2.0 — see [LICENSE](LICENSE)
