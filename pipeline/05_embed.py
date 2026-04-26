#!/usr/bin/env python3
"""Embed every dictation and load into a local Chroma collection.

Provider order:
  - Azure OpenAI text-embedding-3-large (default, ~$0.50 to embed all 27k)
  - Local sentence-transformers (free, slower)

The Chroma collection lives at data/chroma and is persisted across runs.
Re-running this script blows the collection away and rebuilds from scratch.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env early so config picks up keys
ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import config  # noqa: E402

COLLECTION_NAME = "voice_twin_v1"
BATCH = 64


def get_embedder():
    """Return (embed_func, embed_dim, label) according to EMBED_PROVIDER."""
    provider = config.EMBED_PROVIDER
    if provider == "auto":
        provider = "azure" if (config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT) else "local"

    if provider == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )
        deployment = config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

        def embed(texts):
            r = client.embeddings.create(input=texts, model=deployment)
            return [d.embedding for d in r.data]

        # text-embedding-3-large is 3072-dim, -small is 1536. Probe one.
        probe = embed(["dim probe"])
        return embed, len(probe[0]), f"Azure OpenAI ({deployment})"

    # Local fallback
    from sentence_transformers import SentenceTransformer
    print(f"Loading local model {config.LOCAL_EMBED_MODEL}...")
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    def embed(texts):
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    return embed, dim, f"local ({config.LOCAL_EMBED_MODEL})"


def main():
    if not config.HISTORY_JSONL.exists():
        print(f"ERROR: {config.HISTORY_JSONL} not found. Run 02_ingest.py first.")
        sys.exit(2)

    # Load records, filter
    records = []
    with config.HISTORY_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if (r["words"] or 0) < config.MIN_WORDS_FOR_INDEX:
                continue
            records.append(r)
    print(f"Indexing {len(records):,} records "
          f"(skipped <{config.MIN_WORDS_FOR_INDEX} word fragments)")

    embed, dim, label = get_embedder()
    print(f"Embedder: {label}, dim={dim}")

    import chromadb
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    t0 = time.time()
    for i in range(0, len(records), BATCH):
        chunk = records[i:i + BATCH]
        texts = [c["text"] for c in chunk]
        embs = embed(texts)
        coll.add(
            ids=[c["id"] for c in chunk],
            embeddings=embs,
            documents=texts,
            metadatas=[
                {
                    "ctx": c["ctx"],
                    "app": c["app"],
                    "ts": c["ts"][:10] if c["ts"] else "",
                    "edited": c["edited"],
                    "n_words": c["words"],
                }
                for c in chunk
            ],
        )
        elapsed = time.time() - t0
        rate = (i + len(chunk)) / max(0.01, elapsed)
        print(f"  {i + len(chunk):>6}/{len(records)}  ({rate:.0f}/s)", end="\r")
    print()
    print(f"Indexed {coll.count():,} vectors in {time.time()-t0:.0f}s")
    print(f"Collection: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
