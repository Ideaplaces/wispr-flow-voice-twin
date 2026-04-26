#!/usr/bin/env python3
"""Snapshot Wispr Flow's live SQLite to a read-only working copy.

Wispr Flow keeps the file in WAL mode while running. Use SQLite's online
backup API so the copy is consistent regardless of what Flow is doing.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def main():
    src = config.get_flow_sqlite_path()
    dst = config.get_snapshot_path()
    if not src.exists():
        print(f"ERROR: {src} not found. Edit FLOW_SQLITE_PATH in your .env.")
        sys.exit(2)

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    print(f"Source : {src} ({src.stat().st_size / 1e9:.2f} GB)")
    print(f"Target : {dst}")

    src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(str(dst))
    with dst_conn:
        src_conn.backup(
            dst_conn,
            pages=2000,
            progress=lambda r, p, t: print(f"  copied {t - r} / {t} pages", end="\r"),
        )
    print()
    src_conn.close()
    dst_conn.close()
    print(f"Snapshot ready: {dst} ({dst.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
