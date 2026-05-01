"""08_visualize.py - Build a sigma.js-ready graph artifact.

Reads the existing Chroma collection + BERTopic assignments and produces
data/viz/graph.json:

  - nodes: every dictation as {id, x, y, color, topic_label, ts, ctx, text}
           x/y come from a 2D UMAP so spatial proximity is semantic.
  - edges: each node connected to its KNN_K nearest neighbors by cosine
           similarity in the original 3072-dim embedding space (NOT 2D).
           That is what makes hopping between linked thoughts meaningful;
           the 2D layout is only a hint, the edges are the truth.
  - topics: legend (id, label, color, count).

This file is the single artifact the Next.js + sigma.js explorer in web/
loads. Re-runs are safe and idempotent; the viz directory is gitignored.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from config import CHROMA_DIR, DATA_DIR  # noqa: E402

import chromadb  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from umap import UMAP  # noqa: E402

VIZ_DIR = DATA_DIR / "viz"
COLLECTION_NAME = "voice_twin_v1"
TOPIC_INFO = DATA_DIR / "topics" / "topic_info.json"
DOC_TOPICS = DATA_DIR / "topics" / "doc_topics.jsonl"

KNN_K = int(os.environ.get("VIZ_KNN_K", "3"))
UMAP_NEIGHBORS = int(os.environ.get("VIZ_UMAP_NEIGHBORS", "20"))
UMAP_MIN_DIST = float(os.environ.get("VIZ_UMAP_MIN_DIST", "0.15"))

# 30 visually distinct colors for the top 30 topics by size.
COLORS = [
    "#E74C3C", "#3498DB", "#27AE60", "#F1C40F", "#9B59B6",
    "#E67E22", "#1ABC9C", "#34495E", "#FF6B6B", "#4ECDC4",
    "#FF8C42", "#A569BD", "#2ECC71", "#F39C12", "#5499C7",
    "#48C9B0", "#EC7063", "#5D6D7E", "#AF7AC5", "#52BE80",
    "#F4D03F", "#5DADE2", "#EB984E", "#E59866", "#73C6B6",
    "#85929E", "#F0B27A", "#7FB3D5", "#D7BDE2", "#76D7C4",
]
LONG_TAIL_COLOR = "#3a3a3a"
OUTLIER_COLOR = "#1a1a1a"


def _print(msg: str) -> None:
    print(f"[08_visualize] {msg}", flush=True)


def load_corpus():
    _print(f"Opening Chroma at {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    coll = client.get_collection(COLLECTION_NAME)
    res = coll.get(include=["embeddings", "documents", "metadatas"])
    ids = res["ids"]
    docs = res["documents"]
    metas = res["metadatas"] or [{}] * len(ids)
    embs = np.array(res["embeddings"], dtype=np.float32)
    _print(f"Pulled {len(ids):,} docs, embeddings shape {embs.shape}")
    return ids, docs, metas, embs


def load_topic_assignments(ids):
    by_id = {}
    if not DOC_TOPICS.exists():
        _print(f"WARNING: {DOC_TOPICS} not found, all docs will be outliers")
        return {i: -1 for i in ids}
    with DOC_TOPICS.open() as f:
        for line in f:
            row = json.loads(line)
            by_id[row["id"]] = row["topic_id"]
    return by_id


def load_topic_metadata():
    """Return {topic_id: {label, color, count, rank}} sorted by count desc."""
    if not TOPIC_INFO.exists():
        return {}
    info = json.loads(TOPIC_INFO.read_text())
    real = [t for t in info if t["topic_id"] != -1]
    real.sort(key=lambda t: -t["count"])
    out = {}
    for rank, t in enumerate(real):
        color = COLORS[rank] if rank < len(COLORS) else LONG_TAIL_COLOR
        label = t.get("llm_label") or " ".join(t.get("top_words", [])[:4]) or f"topic {t['topic_id']}"
        out[t["topic_id"]] = {
            "label": label,
            "color": color,
            "count": t["count"],
            "rank": rank,
        }
    return out


def umap_2d(embs):
    _print(f"Running UMAP -> 2D (n_neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST})")
    t0 = time.time()
    model = UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    pos = model.fit_transform(embs)
    _print(f"  UMAP done in {time.time() - t0:.1f}s, range "
           f"x=[{pos[:,0].min():.2f},{pos[:,0].max():.2f}] "
           f"y=[{pos[:,1].min():.2f},{pos[:,1].max():.2f}]")
    return pos


def knn_edges(embs, ids, k):
    _print(f"Computing kNN graph in original embedding space (k={k})")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn.fit(embs)
    dists, idxs = nn.kneighbors(embs)
    _print(f"  kNN search done in {time.time() - t0:.1f}s")

    edges = []
    seen = set()
    for src_idx in range(len(ids)):
        src = ids[src_idx]
        for j in range(1, k + 1):  # j=0 is self
            tgt_idx = int(idxs[src_idx][j])
            tgt = ids[tgt_idx]
            key = (src, tgt) if src < tgt else (tgt, src)
            if key in seen:
                continue
            seen.add(key)
            sim = float(1.0 - dists[src_idx][j])
            edges.append({"source": key[0], "target": key[1], "weight": round(sim, 3)})
    _print(f"  Built {len(edges):,} unique undirected edges")
    return edges


def build_nodes(ids, docs, metas, pos, topic_assignments, topic_meta):
    nodes = []
    for i, did in enumerate(ids):
        topic_id = topic_assignments.get(did, -1)
        info = topic_meta.get(topic_id)
        color = info["color"] if info else OUTLIER_COLOR
        label = info["label"] if info else "outlier"
        meta = metas[i] or {}
        text = docs[i] or ""
        preview = text[:140].replace("\n", " ").strip()
        nodes.append({
            "id": did,
            "x": float(pos[i][0]),
            "y": float(pos[i][1]),
            "label": preview,
            "color": color,
            "topic_id": int(topic_id),
            "topic_label": label,
            "ctx": meta.get("ctx") or "other",
            "ts": meta.get("ts") or "",
            "text": text,
        })
    return nodes


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    ids, docs, metas, embs = load_corpus()
    topic_assignments = load_topic_assignments(ids)
    topic_meta = load_topic_metadata()
    _print(f"Loaded {len(topic_meta)} labeled topics")

    pos = umap_2d(embs)
    edges = knn_edges(embs, ids, KNN_K)
    nodes = build_nodes(ids, docs, metas, pos, topic_assignments, topic_meta)

    topics_legend = sorted(
        [
            {"id": tid, "label": info["label"], "color": info["color"], "count": info["count"]}
            for tid, info in topic_meta.items()
        ],
        key=lambda t: -t["count"],
    )

    output = {
        "nodes": nodes,
        "edges": edges,
        "topics": topics_legend,
        "stats": {
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_topics": len(topics_legend),
        },
    }

    out_path = VIZ_DIR / "graph.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    size_mb = out_path.stat().st_size / 1_048_576
    _print(f"Wrote {out_path}: {len(nodes):,} nodes, "
           f"{len(edges):,} edges, {len(topics_legend)} topics, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
