#!/usr/bin/env python3
"""Export the History table to JSONL, one record per dictation.

Captures the fields the voice twin needs at retrieval and generation time:
the three text variants (asrText, formattedText, editedText), context
metadata, and the per-record signals Wispr already computed.

Strips leading BOM characters that macOS-pasted Wispr text sometimes carries.
"""

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

BOM = "﻿"


def clean(s):
    if not s:
        return s
    s = s.replace(BOM, "")
    return s.strip()


def best_text(row):
    """The text we treat as 'what Chip actually shipped'."""
    return clean(row["editedText"]) or clean(row["formattedText"]) or clean(row["asrText"]) or ""


def main():
    src = config.get_snapshot_path()
    if not src.exists():
        print(f"ERROR: {src} not found. Run pipeline/01_snapshot.py first.")
        sys.exit(2)

    conn = sqlite3.connect(f"file:{src}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """SELECT transcriptEntityId, asrText, formattedText, editedText,
                  app, url, timestamp, duration, numWords,
                  numWordsCorrected, numDictionaryReplacements,
                  formattingDivergenceScore, detectedLanguage
           FROM History
           WHERE timestamp IS NOT NULL"""
    )

    config.HISTORY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_short = 0
    skipped_nontext = 0
    with config.HISTORY_JSONL.open("w") as out:
        for r in cur.fetchall():
            text = best_text(r)
            if not text:
                skipped_nontext += 1
                continue
            words = r["numWords"] or len(text.split())
            edited = bool(r["editedText"] and clean(r["editedText"]) != clean(r["formattedText"]))
            ctx = config.context_for(r["app"] or "")
            rec = {
                "id": r["transcriptEntityId"],
                "ctx": ctx,
                "app": r["app"] or "",
                "url": r["url"] or "",
                "ts": r["timestamp"],
                "lang": r["detectedLanguage"] or "unknown",
                "duration_s": r["duration"] or 0,
                "words": words,
                "edited": edited,
                "asr_text": clean(r["asrText"]) or None,
                "formatted_text": clean(r["formattedText"]) or None,
                "edited_text": clean(r["editedText"]) or None,
                "text": text,
                "div_score": r["formattingDivergenceScore"],
                "n_corrected": r["numWordsCorrected"] or 0,
                "n_dict_repl": r["numDictionaryReplacements"] or 0,
            }
            out.write(json.dumps(rec) + "\n")
            written += 1

    print(f"Wrote {written:,} records to {config.HISTORY_JSONL}")
    print(f"  skipped (no text): {skipped_nontext}")


if __name__ == "__main__":
    main()
