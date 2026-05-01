#!/usr/bin/env bash
# Weekly job: re-fit BERTopic over the full corpus so cluster shape catches
# up with whatever has accumulated. LLM-labels the top 30 topics again.
# Cheap (~60s + a few cents on Azure), safe to skip if it ever fails.
set -euo pipefail

REPO="/home/chipdev/ideaplaces-meta/wispr-flow-voice-twin"
LOG_DIR="$REPO/data/logs"
LOG_FILE="$LOG_DIR/refit.log"

mkdir -p "$LOG_DIR"

cd "$REPO"
{
    echo
    echo "===== $(date -u +%FT%TZ) refit ====="
    TOPIC_LLM_LABELS=1 TOPIC_LLM_MAX=30 .venv/bin/python pipeline/06_topics.py
} >> "$LOG_FILE" 2>&1
