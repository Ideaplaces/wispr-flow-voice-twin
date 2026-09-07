#!/usr/bin/env python3
"""02b_enrich.py - backfill the derived fields on every row of history.jsonl.

Deduplicates by id (last copy wins), strips HTML, normalizes proper nouns into
text_norm, flags human-facing rows, and labels each row's kind: rules first,
then the model for the human-facing rows the rules could not settle.

Run once after upgrading, and any time the glossary changes. Safe to re-run:
existing kinds are kept unless --reset-kinds is passed.

    python pipeline/02b_enrich.py            # rules + model
    python pipeline/02b_enrich.py --no-llm   # rules only, no network
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Skip the model pass; unlabeled rows become 'other'")
    ap.add_argument("--reset-kinds", action="store_true", help="Recompute every kind, not only the missing ones")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of rows sent to the model")
    args = ap.parse_args()

    rows = corpus.load_history()
    print(f"{len(rows):,} unique rows (history file had duplicates collapsed)")

    glossary = corpus.load_glossary()
    for r in rows:
        if args.reset_kinds:
            r.pop("kind", None)
        corpus.enrich(r, glossary)

    changed = sum(1 for r in rows if r["text_norm"] != r["text"])
    print(f"{changed:,} rows had a proper noun normalized")

    pending = [r for r in rows if r["human"] and not r.get("kind")]
    print(f"{sum(1 for r in rows if r['human']):,} human-facing rows, {len(pending):,} need the model for a kind")
    if args.limit:
        pending = pending[: args.limit]
    if pending and not args.no_llm:
        labels = corpus.classify_kinds_llm([r["text"] for r in pending])
        for r, label in zip(pending, labels):
            r["kind"] = label
    for r in rows:
        if not r.get("kind"):
            r["kind"] = "other"

    corpus.write_history(rows)
    kinds = Counter(r["kind"] for r in rows if r["human"])
    print("kinds among human-facing rows:", dict(kinds.most_common()))
    print(f"Wrote {len(rows):,} rows")


if __name__ == "__main__":
    main()
