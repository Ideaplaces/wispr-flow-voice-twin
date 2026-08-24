#!/usr/bin/env bash
# Cron entry: pick up any wispr-flow-delta*.jsonl in the inbox and run the
# incremental ingest. No-op when the inbox is empty, so it's safe to fire
# every 15 minutes regardless of when a Mac last rsynced.
#
# The glob, not one fixed filename: each Mac pushes host-stamped deltas
# (wispr-flow-delta-<host>-<stamp>.jsonl) so two machines cannot overwrite
# each other's export. Matching only the old bare name here would leave every
# new file sitting in the inbox, uningested and silent.
set -euo pipefail
shopt -s nullglob

REPO="/home/chipdev/ideaplaces-meta/wispr-flow-voice-twin"
LOG_DIR="$REPO/data/logs"
LOG_FILE="$LOG_DIR/ingest.log"

mkdir -p "$LOG_DIR"

# Skip silently if there's nothing to ingest. Cron noise reduction.
DELTAS=("$REPO"/data/inbox/wispr-flow-delta*.jsonl)
if [ ${#DELTAS[@]} -eq 0 ]; then
    exit 0
fi

cd "$REPO"
{
    echo
    echo "===== $(date -u +%FT%TZ) ingest ====="
    .venv/bin/python pipeline/07_ingest_delta.py
} >> "$LOG_FILE" 2>&1
