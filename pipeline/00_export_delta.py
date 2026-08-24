#!/usr/bin/env python3
"""Mac-side incremental export of Wispr Flow dictations.

Reads from the local Wispr Flow SQLite DB, exports new rows since the cursor
in ~/.wispr-delta-state.json (or a fallback if absent), writes them as JSONL
into <repo>/data/inbox/, then rsyncs the file to the Ubuntu cloud dev machine.
State only advances on successful export; rsync failure is non-fatal because
the next run picks the file up and retries.

Every export gets its own file, named for the machine that produced it and the
moment it ran: wispr-flow-delta-<host>-<stamp>.jsonl. Chip dictates from more
than one Mac, and each one keeps its own cursor, so a single fixed filename in
a shared inbox is a data-loss bug in two directions. Two Macs exporting inside
one ingest window would have the second overwrite the first, and the loser has
already advanced its cursor, so those dictations are gone for good. The same
holds for one Mac against itself whenever an ingest is late or fails. Unique
names cost nothing and the box ingests every file it finds.
"""
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = REPO_ROOT / "data" / "inbox"
HOST_OVERRIDE = Path.home() / ".wispr-sync-host"

SOURCE_DB = Path.home() / "Library" / "Application Support" / "Wispr Flow" / "flow.sqlite"
SNAPSHOT_PATH = "/tmp/wispr-flow-snapshot.sqlite"
STATE_PATH = Path.home() / ".wispr-delta-state.json"
DEFAULT_CUTOFF = "2026-04-26 03:02:57.099 +00:00"

# The trailing slash matters: rsync copies into the directory and keeps the
# host-stamped filename. Addressed by the ssh alias, not the VM's name, which
# has already outlived two rebuilds (v2 -> v4).
RSYNC_DEST = "chipdev@chipdev:/home/chipdev/ideaplaces-meta/wispr-flow-voice-twin/data/inbox/"

CONTEXT_GROUPS = {
    "ai_chat": {"com.todesktop.230313mzl4w4u92", "com.microsoft.VSCode", "com.anthropic.claudefordesktop"},
    "team_chat": {"com.tinyspeck.slackmacgap", "com.hnc.Discord", "org.whispersystems.signal-desktop"},
    "personal_chat": {"net.whatsapp.WhatsApp", "com.apple.MobileSMS"},
    "browser": {"company.thebrowser.Browser", "com.google.Chrome", "com.apple.Safari"},
}


def load_cutoff():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
            last_ts = state.get("last_ts")
            if last_ts:
                return last_ts, "state_file"
        except (OSError, ValueError):
            pass
    return DEFAULT_CUTOFF, "hardcoded_fallback"


def save_state(latest_ts):
    tmp_path = str(STATE_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"last_ts": latest_ts}, f)
    os.replace(tmp_path, STATE_PATH)


def clean(s):
    if s is None:
        return ""
    return s.replace("﻿", "").strip()


def context_for(app):
    if not app:
        return "other"
    for group, bundles in CONTEXT_GROUPS.items():
        if app in bundles:
            return group
    return "other"


def host_label():
    """Short, stable, filesystem safe name for this Mac.

    Defaults to the hostname, which drifts on some networks (`foo.lan` ->
    `foo-2.local`). Drift is harmless: it only changes the name of the next
    delta file, and the box ingests every file it finds. ~/.wispr-sync-host
    pins it when a stable label is wanted.
    """
    if HOST_OVERRIDE.exists():
        try:
            raw = HOST_OVERRIDE.read_text().strip()
        except OSError:
            raw = ""
    else:
        raw = ""
    if not raw:
        raw = socket.gethostname().split(".")[0]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()
    return slug or "unknown-host"


def output_path(host):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return INBOX_DIR / f"wispr-flow-delta-{host}-{stamp}.jsonl"


def drop_stale_local(host, keep):
    """Remove this host's earlier local deltas once a newer one is delivered.

    A failed rsync leaves its file behind on purpose, but the cursor did not
    advance, so the next successful run re-exports those rows as a superset.
    The older file is then redundant, and without this it would sit in the
    inbox forever.
    """
    for old in INBOX_DIR.glob(f"wispr-flow-delta-{host}-*.jsonl"):
        if old == keep:
            continue
        try:
            old.unlink()
            print(f"Removed superseded local delta: {old.name}")
        except OSError:
            pass


def make_snapshot(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def rsync_to_remote(local_path, remote_dest):
    rsync = shutil.which("rsync") or "/usr/bin/rsync"
    try:
        result = subprocess.run(
            [rsync, "-avz", str(local_path), remote_dest],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"rsync OK -> {remote_dest}")
            return True
        print(f"rsync failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return False
    except FileNotFoundError as e:
        print(f"rsync binary not found: {e}", file=sys.stderr)
        return False


def main():
    if not SOURCE_DB.exists():
        print(f"Source DB not found: {SOURCE_DB}", file=sys.stderr)
        sys.exit(1)

    host = host_label()
    out_path = output_path(host)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Host label: {host}")

    cutoff, source = load_cutoff()
    if source == "state_file":
        print(f"Loaded cursor from {STATE_PATH}")
    else:
        print("No state file found; using hardcoded fallback cutoff")
    print(f"Applied cutoff: {cutoff}")

    make_snapshot(str(SOURCE_DB), SNAPSHOT_PATH)

    conn = sqlite3.connect(f"file:{SNAPSHOT_PATH}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT transcriptEntityId, asrText, formattedText, editedText,
               app, url, timestamp, duration, numWords,
               numWordsCorrected, numDictionaryReplacements,
               formattingDivergenceScore, detectedLanguage
        FROM History
        WHERE timestamp IS NOT NULL
          AND timestamp > ?
        """,
        (cutoff,),
    )

    count = 0
    earliest = None
    latest = None
    lines = []

    for row in cur:
        asr_clean = clean(row["asrText"])
        fmt_clean = clean(row["formattedText"])
        edt_clean = clean(row["editedText"])

        text = edt_clean or fmt_clean or asr_clean
        if not text:
            continue

        edited = bool(row["editedText"] and edt_clean != fmt_clean)
        app = row["app"] or ""
        url = row["url"] or ""
        lang = row["detectedLanguage"] or "unknown"
        duration_s = row["duration"] or 0
        words = row["numWords"] if row["numWords"] is not None else len(text.split())
        n_corrected = row["numWordsCorrected"] or 0
        n_dict_repl = row["numDictionaryReplacements"] or 0
        ts = row["timestamp"]

        record = {
            "id": row["transcriptEntityId"],
            "ctx": context_for(app),
            "app": app,
            "url": url,
            "ts": ts,
            "lang": lang,
            "duration_s": duration_s,
            "words": words,
            "edited": edited,
            "asr_text": asr_clean if asr_clean else None,
            "formatted_text": fmt_clean if fmt_clean else None,
            "edited_text": edt_clean if edt_clean else None,
            "text": text,
            "div_score": row["formattingDivergenceScore"],
            "n_corrected": n_corrected,
            "n_dict_repl": n_dict_repl,
        }

        lines.append(json.dumps(record, ensure_ascii=False))
        count += 1

        if earliest is None or ts < earliest:
            earliest = ts
        if latest is None or ts > latest:
            latest = ts

    conn.close()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    try:
        os.remove(SNAPSHOT_PATH)
    except OSError:
        pass

    print(f"Records written: {count}")
    print(f"Earliest ts: {earliest}")
    print(f"Latest ts:   {latest}")
    print(f"Output: {out_path}")

    if count == 0 or not latest:
        print("0 new records; skipping rsync and state update")
        try:
            out_path.unlink()
        except OSError:
            pass
        return

    if rsync_to_remote(out_path, RSYNC_DEST):
        save_state(latest)
        print(f"State advanced to {latest}")
        drop_stale_local(host, out_path)
        try:
            out_path.unlink()
        except OSError:
            pass
    else:
        print("rsync failed; state NOT advanced (next run will retry)")


if __name__ == "__main__":
    main()
