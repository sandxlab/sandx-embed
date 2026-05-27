"""sandx-embed — encode text records and run ANN search.

Demonstrates the full sandx-embed workflow: encode a corpus of company names
into dense vectors, build an ANN index, then run semantic nearest-neighbor
queries to find similar records.

Install:  pip install "sandx-embed[hnsw]"   # for HNSW (recommended)
          pip install sandx-embed             # for exact search only
Run:      python -m examples.encode_and_search
"""

from __future__ import annotations

import numpy as np

from sandx_embed import Encoder, VectorIndex

W = 60
SEP  = "=" * W
RULE = "-" * W

CORPUS = [
    "Acme Corporation",
    "Acme Corp.",
    "Acme Co",
    "GlobalTech Industries",
    "Global Tech Inc.",
    "GlobalTech Inc",
    "Meridian Health Solutions",
    "Meridian Health Soln.",
    "Meridian Health Solution",
    "DataVault Systems",
    "DataVault Sys.",
    "Data Vault Systems Corp.",
    "Vertex Research Labs",
    "Vertex Research Laboratories",
    "Vertex Res. Labs",
]

QUERIES = [
    "Acme Corp",
    "GlobalTech",
    "Meridian Health",
    "Vertex Labs",
]


def run() -> None:
    print()
    print("  " + SEP)
    print("   sandx-embed  --  Encode & ANN Search")
    print("  " + SEP)
    print(f"  {len(CORPUS)} corpus records  |  {len(QUERIES)} queries")
    print()

    # ── 1. Encode ────────────────────────────────────────────────────────
    print("  Encoding corpus (sentence-bert, all-MiniLM-L6-v2) ...")
    enc = Encoder(model="sentence-bert")
    corpus_vecs = enc.encode(CORPUS, normalize=True)
    print(f"  Vectors: shape={corpus_vecs.shape}  dtype={corpus_vecs.dtype}")
    print()

    # ── 2. Build index ───────────────────────────────────────────────────
    ids = [str(i) for i in range(len(CORPUS))]
    try:
        idx = VectorIndex(method="hnsw", metric="cosine")
        idx.build(corpus_vecs, ids=ids)
        method_used = "HNSW"
    except ImportError:
        idx = VectorIndex(method="exact", metric="cosine")
        idx.build(corpus_vecs, ids=ids)
        method_used = "exact (install usearch for HNSW)"

    print(f"  Index: {idx}  [{method_used}]")
    print()

    # ── 3. Query ─────────────────────────────────────────────────────────
    print("  NEAREST NEIGHBORS  (k=3 per query)")
    print("  " + RULE)

    query_vecs = enc.encode(QUERIES, normalize=True)

    for query, qvec in zip(QUERIES, query_vecs):
        result = idx.query(qvec, k=3)
        print(f"  Query: {query!r}")
        for rank, (hit_id, dist) in enumerate(zip(result.ids, result.distances), 1):
            hit_name = CORPUS[int(hit_id)]
            sim = max(0.0, 1.0 - dist)
            bar = "#" * int(sim * 20)
            print(f"    {rank}. {hit_name:<34}  sim={sim:.3f}  {bar}")
        print()

    # ── 4. Pairwise similarity matrix (small demo) ───────────────────────
    sample = CORPUS[:6]
    sample_vecs = corpus_vecs[:6]
    sim_matrix = sample_vecs @ sample_vecs.T

    print("  PAIRWISE SIMILARITY  (first 6 records)")
    print("  " + RULE)
    header = "  " + " " * 26 + "".join(f"{i:>5}" for i in range(6))
    print(header)
    for i, (name, row) in enumerate(zip(sample, sim_matrix)):
        short = name[:24]
        vals = "".join(f"{v:>5.2f}" for v in row)
        print(f"  {short:<26}{vals}")

    print()
    print("  " + SEP)
    print(f"   Encoded {len(CORPUS)} records  ->  dim={enc.dim}  |  index built & queried")
    print("  " + SEP)
    print()


if __name__ == "__main__":
    run()
