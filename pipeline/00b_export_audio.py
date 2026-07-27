#!/usr/bin/env python3
"""Mac-side harvest of Wispr Flow voice recordings.

Wispr Flow keeps raw audio (WAV blobs in the History.audio column) for only
about a week before purging it. This script rescues those recordings before
the purge: it snapshots the local DB, extracts every audio blob newer than
the cursor in ~/.wispr-audio-state.json into a local archive, then rsyncs
the whole archive to the SSH machine. The archive is a keep-forever store,
deliberately separate from the text-delta inbox — nothing on the server
processes it.

Layout: <archive>/<YYYY-MM>/<UTC timestamp>_<id prefix>.wav
Two copies result: the local archive on this Mac and the mirror on the
server's data disk (which has daily Azure snapshots). State only advances
after a successful rsync; because rsync mirrors the whole directory, any
file missed by a failed run is retried on the next one.
"""
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

SOURCE_DB = Path.home() / "Library" / "Application Support" / "Wispr Flow" / "flow.sqlite"
SNAPSHOT_PATH = "/tmp/wispr-flow-audio-snapshot.sqlite"
STATE_PATH = Path.home() / ".wispr-audio-state.json"
ARCHIVE_DIR = Path.home() / "Backups" / "wispr-voice-audio"

RSYNC_DEST = "chipdev@dev-workstation-canada-v2:/home/chipdev/voice-audio-archive/"


def load_cutoff():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                last_ts = json.load(f).get("last_ts")
            if last_ts:
                return last_ts
        except (OSError, ValueError):
            pass
    return "1970-01-01 00:00:00.000 +00:00"


def save_state(latest_ts):
    tmp_path = str(STATE_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"last_ts": latest_ts}, f)
    os.replace(tmp_path, STATE_PATH)


def make_snapshot(src_path, dst_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    dst = sqlite3.connect(dst_path)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def wav_filename(ts, entity_id):
    # "2026-07-27 00:12:02.127 +00:00" -> "2026-07-27T00-12-02"
    stamp = re.sub(r"[ :]", "-", ts[:19]).replace("--", "T", 1)
    return f"{stamp}_{entity_id[:8]}.wav"


def rsync_to_remote():
    rsync = shutil.which("rsync") or "/usr/bin/rsync"
    result = subprocess.run(
        [rsync, "-az", str(ARCHIVE_DIR) + "/", RSYNC_DEST],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"rsync OK -> {RSYNC_DEST}")
        return True
    print(f"rsync failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
    return False


def main():
    if not SOURCE_DB.exists():
        print(f"Source DB not found: {SOURCE_DB}", file=sys.stderr)
        sys.exit(1)

    cutoff = load_cutoff()
    print(f"Applied cutoff: {cutoff}")

    make_snapshot(str(SOURCE_DB), SNAPSHOT_PATH)
    conn = sqlite3.connect(f"file:{SNAPSHOT_PATH}?mode=ro&immutable=1", uri=True)
    cur = conn.execute(
        """
        SELECT transcriptEntityId, timestamp, audio
        FROM History
        WHERE audio IS NOT NULL
          AND timestamp IS NOT NULL
          AND timestamp > ?
        ORDER BY timestamp
        """,
        (cutoff,),
    )

    count = 0
    total_bytes = 0
    latest = None
    for entity_id, ts, audio in cur:
        month_dir = ARCHIVE_DIR / ts[:7]
        month_dir.mkdir(parents=True, exist_ok=True)
        out_path = month_dir / wav_filename(ts, entity_id)
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(audio)
        count += 1
        total_bytes += len(audio)
        if latest is None or ts > latest:
            latest = ts

    conn.close()
    try:
        os.remove(SNAPSHOT_PATH)
    except OSError:
        pass

    print(f"Recordings extracted: {count} ({total_bytes / 1048576:.1f} MB)")
    if count == 0:
        print("0 new recordings; skipping rsync and state update")
        return

    if rsync_to_remote():
        save_state(latest)
        print(f"State advanced to {latest}")
    else:
        print("rsync failed; state NOT advanced (next run will retry)")


if __name__ == "__main__":
    main()
