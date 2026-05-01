#!/usr/bin/env python3
"""07_ingest_delta.py - Apply a Mac-side delta JSONL to this corpus.

Reads new dictation records from data/inbox/wispr-flow-delta.jsonl, embeds
only the ones not already in Chroma, appends them to data/history.jsonl,
and moves the inbox file to data/inbox/processed/<timestamp>.jsonl.

This is the incremental sibling of 05_embed.py. 05 rebuilds from scratch;
07 only ever adds.

Idempotent by id: if you re-run with the same delta file, every row is
detected as already-indexed and the operation is a no-op (apart from the
file move at the end).
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

import config  # noqa: E402

INBOX_FILE = config.DATA_DIR / "inbox" / "wispr-flow-delta.jsonl"
PROCESSED_DIR = config.DATA_DIR / "inbox" / "processed"
COLLECTION_NAME = "voice_twin_v1"
BATCH = 64


def get_embedder():
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

        return embed, f"Azure OpenAI ({deployment})"

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

    def embed(texts):
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    return embed, f"local ({config.LOCAL_EMBED_MODEL})"


def append_to_history(new_records: list[dict]) -> None:
    """Append delta rows to data/history.jsonl, ensuring a newline boundary."""
    config.HISTORY_JSONL.parent.mkdir(parents=True, exist_ok=True)

    if config.HISTORY_JSONL.exists() and config.HISTORY_JSONL.stat().st_size > 0:
        with config.HISTORY_JSONL.open("rb") as f:
            f.seek(-1, os.SEEK_END)
            ends_with_newline = f.read(1) == b"\n"
        if not ends_with_newline:
            with config.HISTORY_JSONL.open("ab") as f:
                f.write(b"\n")

    with config.HISTORY_JSONL.open("a", encoding="utf-8") as f:
        for r in new_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    if not INBOX_FILE.exists():
        print(f"ERROR: {INBOX_FILE} not found. Nothing to do.")
        sys.exit(2)

    # Load delta records
    records: list[dict] = []
    with INBOX_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"Loaded {len(records):,} records from {INBOX_FILE.name}")

    if not records:
        print("Empty delta. Moving file to processed/ anyway.")
        _archive(INBOX_FILE)
        return

    # Open Chroma and get the set of ids already indexed
    import chromadb  # noqa: E402
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    before = coll.count()
    print(f"Chroma {COLLECTION_NAME!r} currently has {before:,} vectors")

    # Pull existing ids in chunks (Chroma .get() with no ids returns everything)
    existing_ids = set(coll.get(include=[])["ids"])
    print(f"  cached {len(existing_ids):,} existing ids for dedupe")

    # Decide which delta rows are new AND eligible for indexing
    eligible: list[dict] = []
    skipped_dupes = 0
    skipped_short = 0
    for r in records:
        if r["id"] in existing_ids:
            skipped_dupes += 1
            continue
        if (r.get("words") or 0) < config.MIN_WORDS_FOR_INDEX:
            skipped_short += 1
            continue
        if not r.get("text"):
            continue
        eligible.append(r)

    print(f"  {len(eligible):,} eligible to embed "
          f"(skipped {skipped_dupes} dupes, {skipped_short} short fragments)")

    # Append the entire delta to history.jsonl regardless of indexing eligibility
    append_to_history(records)
    print(f"Appended {len(records):,} rows to {config.HISTORY_JSONL.name}")

    if not eligible:
        print("No new rows to embed. Done.")
        _archive(INBOX_FILE)
        return

    embed, label = get_embedder()
    print(f"Embedder: {label}")

    t0 = time.time()
    added = 0
    for i in range(0, len(eligible), BATCH):
        chunk = eligible[i:i + BATCH]
        texts = [c["text"] for c in chunk]
        embs = embed(texts)
        coll.add(
            ids=[c["id"] for c in chunk],
            embeddings=embs,
            documents=texts,
            metadatas=[
                {
                    "ctx": c.get("ctx") or "other",
                    "app": c.get("app") or "",
                    "ts": (c.get("ts") or "")[:10],
                    "edited": bool(c.get("edited")),
                    "n_words": c.get("words") or 0,
                }
                for c in chunk
            ],
        )
        added += len(chunk)
        elapsed = time.time() - t0
        rate = added / max(0.01, elapsed)
        print(f"  {added:>5}/{len(eligible)}  ({rate:.0f}/s)", end="\r")
    print()
    after = coll.count()
    print(f"Embedded {added:,} new rows in {time.time() - t0:.1f}s")
    print(f"Chroma now has {after:,} vectors  (was {before:,}, +{after - before})")

    _archive(INBOX_FILE)


def _archive(path: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = PROCESSED_DIR / f"{path.stem}-{stamp}{path.suffix}"
    shutil.move(str(path), str(dest))
    print(f"Moved {path.name} -> {dest.relative_to(config.DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
