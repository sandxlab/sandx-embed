# Embedding Systems — Domain Overview

**Domain:** Latent representation learning, vector similarity, approximate nearest neighbor search
**SandX engine:** `sandx-embed`
**Phase 2 priority:** #2 (shared dependency; required before sandx-er v0.2 and sandx-graph)

---

## What Are Embedding Systems?

An embedding is a learned dense vector representation of an object — a record, a word, a graph node, an image — in a continuous latent space where semantic similarity corresponds to geometric proximity. Embedding systems are the foundation of modern AI: they power search, recommendation, deduplication, reasoning, and memory.

`sandx-embed` is not a standalone product but a **shared infrastructure layer** used by every other SandX engine. It provides the encoding and similarity computation primitives that make `sandx-er` embedding-aware and `sandx-graph` representation-aware.

---

## Core Components

### Encoders
Transform raw input into dense vectors.
- **Text encoders:** Pre-trained transformers (BERT, Sentence-BERT, E5) for text fields
- **Tabular encoders:** Field-level embeddings for structured records (used in ER)
- **Graph encoders:** GNN-based node embeddings from sandx-graph
- **Cross-modal alignment:** Shared embedding space across different input types

### Vector Index (ANN)
Approximate Nearest Neighbor search — the data structure that makes embedding-based blocking and similarity search tractable at scale.
- HNSW (Hierarchical Navigable Small World) — high recall, low latency, favored for production
- FAISS (Facebook AI Similarity Search) — GPU-accelerated, high throughput
- Annoy — low memory, suitable for read-heavy workloads
- Exact search — baseline for small datasets

### Similarity Engine
Compute similarity between embedding pairs or between a query embedding and an index.
- Cosine similarity (normalized dot product)
- Euclidean distance
- Inner product (for asymmetric retrieval)
- Batch pairwise similarity for candidate generation

---

## Why This Is Its Own Engine

Embedding infrastructure has three properties that justify treating it as a first-class SandX component:

1. **Shared dependency** — ER blocking, ER matching, graph node similarity, and compute workload routing all use embeddings. A single well-maintained layer avoids three separate implementations.
2. **Performance critical** — ANN search at scale requires careful index tuning. This is non-trivial engineering that should not be duplicated per-engine.
3. **Independent evolution** — encoder models improve rapidly. Isolating the embedding layer allows model upgrades without touching ER or graph logic.

---

## State of the Art

| Component | Key Systems |
|-----------|------------|
| Text embedding models | SBERT, E5, BGE, OpenAI Ada, Cohere Embed |
| Tabular/record embeddings | EmbDI, Sherlock, DODUO |
| ANN indexes | FAISS, HNSW (hnswlib), ScaNN, Annoy, Weaviate |
| Vector databases | Qdrant, Weaviate, Pinecone, Chroma, pgvector |
| Benchmark datasets | BEIR (retrieval), ANN-benchmarks (ANN performance) |

---

## SandX-Embed Design Constraints

1. **Pluggable encoders** — any encoder model can be registered; the interface is model-agnostic.
2. **Index portability** — indexes must be serializable and reloadable without rebuilding.
3. **Deterministic output** — given the same model version and input, output is reproducible.
4. **Lazy loading** — models are not loaded until first use; multiple models can coexist in memory-constrained environments.
5. **No vendor lock-in** — no hard dependency on any single vector database or embedding provider.

---

## Key References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP.*
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-Scale Similarity Search with GPUs (FAISS). *IEEE Transactions on Big Data.*
- Malkov, Y. A., & Yashunin, D. A. (2018). Efficient and Robust Approximate Nearest Neighbor Search Using HNSW. *IEEE TPAMI.*
- Cappuzzo, R., Papotti, P., & Thirumuruganathan, S. (2020). Creating Embeddings of Heterogeneous Relational Datasets (EmbDI). *SIGMOD.*
- Muennighoff, N. et al. (2023). MTEB: Massive Text Embedding Benchmark. *EACL.*
- Wang, L. et al. (2022). Text Embeddings by Weakly-Supervised Contrastive Pre-Training (E5). *arXiv:2212.03533.* — E5 family; strong general-purpose text encoders; leading MTEB performance; candidate default encoder for `sandx-embed`.
- Su, H. et al. (2022). One Embedder, Any Task: Instruction-Finetuned Text Embeddings (INSTRUCTOR). *ACL 2023.* — Instruction-conditioned embeddings; relevant for domain-specific adaptation without fine-tuning.
- Douze, M. et al. (2024). The FAISS Library. *arXiv:2401.08281.* — Comprehensive description of production FAISS; covers GPU indexes, compression, and the IVF+PQ pipeline used in large-scale `sandx-embed` deployments.
