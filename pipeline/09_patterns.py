"""09_patterns.py - find topics that look like things you keep telling
someone to do.

The mirror layer above topics. Topics show what you talk about. This
script asks a different question: which topics look like *recurring
instructions*, not just themes? Where do you keep saying "do this", "fix
that", "deploy this", week after week, month after month?

Those are automation candidates. The script ranks them, dumps evidence,
and writes data/patterns/automation_candidates.json so the CLI can read
the same scores without re-running the analysis.

Heuristics:

  - Imperative density: fraction of dictations in a topic that begin with
    a known imperative verb. High density means the topic is dominated by
    commands, not chitchat.
  - Recurrence: number of distinct calendar days the topic appears on.
    A theme that surfaces on 30 different days is more automation-worthy
    than one that surfaced on 3 days during a single sprint.
  - Volume: total dictation count. Tiny clusters can be ignored.

Score combines all three, gently weighted toward density and recurrence.

The script is read-only. It reads doc_topics.jsonl + topic_info.json +
history.jsonl, computes scores, and writes one JSON artifact. No
embedding calls, no LLM calls, free to re-run.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
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

from config import DATA_DIR  # noqa: E402

TOPIC_INFO = DATA_DIR / "topics" / "topic_info.json"
DOC_TOPICS = DATA_DIR / "topics" / "doc_topics.jsonl"
HISTORY_JSONL = DATA_DIR / "history.jsonl"
PATTERNS_DIR = DATA_DIR / "patterns"

MIN_TOPIC_SIZE = int(os.environ.get("PATTERN_MIN_SIZE", "30"))
MIN_UNIQUE_DAYS = int(os.environ.get("PATTERN_MIN_DAYS", "5"))


# Verbs that strongly signal "do this for me" rather than narration.
IMPERATIVE_VERBS = {
    "add", "amend", "analyze", "apply", "archive", "automate", "back",
    "build", "bump", "check", "clean", "clear", "clone", "commit",
    "configure", "connect", "convert", "copy", "create", "debug",
    "delete", "deploy", "destroy", "detect", "disable", "disconnect",
    "do", "document", "download", "edit", "enable", "ensure", "expand",
    "explain", "export", "extract", "fetch", "find", "finish", "fix",
    "format", "generate", "get", "give", "grab", "handle", "help",
    "implement", "improve", "increase", "indent", "ingest", "init",
    "install", "integrate", "investigate", "kill", "launch", "list",
    "load", "log", "look", "make", "map", "merge", "migrate", "monitor",
    "move", "open", "optimize", "parse", "pause", "ping", "plan",
    "polish", "post", "prepare", "process", "pull", "push", "put",
    "query", "reach", "rebase", "rebuild", "redeploy", "reduce",
    "refactor", "refresh", "register", "release", "remove", "rename",
    "render", "replace", "reply", "report", "request", "research",
    "reset", "resolve", "restart", "restore", "retry", "review",
    "revoke", "rewrite", "roll", "rollback", "run", "save", "scrape",
    "send", "serve", "set", "setup", "ship", "show", "simplify", "skip",
    "solve", "split", "stop", "store", "summarize", "switch", "sync",
    "tag", "take", "teach", "tell", "test", "tighten", "trace",
    "track", "trigger", "tweak", "unblock", "uninstall", "update",
    "upload", "validate", "verify", "version", "view", "wait", "watch",
    "wire", "wrap", "write",
}

# Tokens we strip from the start of a dictation before checking the verb.
# Voice-to-text often inserts these as filler.
LEAD_FILLER = {
    "ok", "okay", "alright", "so", "yeah", "right", "well", "now",
    "also", "and", "but", "yes", "no", "hey", "hi", "please",
    "can", "could", "would", "should", "let", "lets", "i", "im", "ive",
    "you", "your",
}


def first_meaningful_word(text: str) -> str:
    """Return the first probably-content word, lowercased.

    Strips punctuation and skips common filler tokens like "ok", "so",
    "please", "can you", "let me". Helps us evaluate whether a
    dictation is imperative even when it starts with throat-clearing.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^a-zA-Z\s']", " ", text).strip().lower()
    for word in cleaned.split():
        word = word.strip("'")
        if not word or word in LEAD_FILLER:
            continue
        return word
    return ""


def is_imperative(text: str) -> bool:
    return first_meaningful_word(text) in IMPERATIVE_VERBS


def parse_date(ts: str) -> str | None:
    """Return the YYYY-MM-DD prefix of a dictation timestamp, or None."""
    if not ts:
        return None
    return ts[:10] if len(ts) >= 10 else None


def load_topic_info() -> dict[int, dict]:
    if not TOPIC_INFO.exists():
        sys.exit(f"Missing {TOPIC_INFO}. Run pipeline/06_topics.py first.")
    info = json.loads(TOPIC_INFO.read_text())
    return {t["topic_id"]: t for t in info}


def load_doc_topics() -> dict[str, dict]:
    """Map each dictation id to its topic assignment + metadata."""
    if not DOC_TOPICS.exists():
        sys.exit(f"Missing {DOC_TOPICS}. Run pipeline/06_topics.py first.")
    out: dict[str, dict] = {}
    with DOC_TOPICS.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row
    return out


def load_history_text() -> dict[str, str]:
    """Map each dictation id to its full text."""
    if not HISTORY_JSONL.exists():
        sys.exit(f"Missing {HISTORY_JSONL}. Run pipeline/02_ingest.py first.")
    out: dict[str, str] = {}
    with HISTORY_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r.get("text") or r.get("formatted_text") or r.get("asr_text") or ""
    return out


def main() -> None:
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

    topic_info = load_topic_info()
    doc_topics = load_doc_topics()
    history = load_history_text()
    print(f"[09_patterns] {len(doc_topics):,} dictations, {len(topic_info)} topics", flush=True)

    # Per-topic accumulators
    counts: dict[int, int] = defaultdict(int)
    imperative_counts: dict[int, int] = defaultdict(int)
    days: dict[int, set[str]] = defaultdict(set)
    samples: dict[int, list[tuple[str, str, str]]] = defaultdict(list)  # (id, ts, text)

    for did, dt in doc_topics.items():
        tid = dt["topic_id"]
        if tid == -1:
            continue
        text = history.get(did, "")
        ts = dt.get("ts") or ""
        counts[tid] += 1
        if is_imperative(text):
            imperative_counts[tid] += 1
        d = parse_date(ts)
        if d:
            days[tid].add(d)
        if len(samples[tid]) < 8:
            samples[tid].append((did, ts, text))

    # Score and rank
    candidates: list[dict] = []
    for tid, n in counts.items():
        if n < MIN_TOPIC_SIZE:
            continue
        nd = len(days[tid])
        if nd < MIN_UNIQUE_DAYS:
            continue
        imp = imperative_counts[tid]
        density = imp / n
        # Score gently weighted toward density and recurrence so that big
        # but soft topics ("greetings", "small talk") don't dominate.
        score = density * math.log1p(nd) * math.log1p(n)
        info = topic_info.get(tid, {})
        label = info.get("llm_label") or " ".join(info.get("top_words", [])[:4]) or f"topic {tid}"
        sample_dates = sorted(days[tid])
        # Pick representative imperative samples preferentially
        prepared = []
        for did, ts, text in samples[tid]:
            prepared.append({
                "id": did,
                "ts": ts,
                "imperative": is_imperative(text),
                "preview": (text or "").replace("\n", " ")[:240],
            })
        prepared.sort(key=lambda r: (not r["imperative"], r["ts"]))
        candidates.append({
            "topic_id": tid,
            "label": label,
            "count": n,
            "imperative_count": imp,
            "imperative_density": round(density, 4),
            "unique_days": nd,
            "first_seen": sample_dates[0] if sample_dates else None,
            "last_seen": sample_dates[-1] if sample_dates else None,
            "score": round(score, 4),
            "samples": prepared[:5],
        })

    candidates.sort(key=lambda c: -c["score"])
    print(f"[09_patterns] {len(candidates)} topics passed gates "
          f"(min size {MIN_TOPIC_SIZE}, min days {MIN_UNIQUE_DAYS})", flush=True)

    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "thresholds": {
            "min_topic_size": MIN_TOPIC_SIZE,
            "min_unique_days": MIN_UNIQUE_DAYS,
        },
        "candidates": candidates,
    }
    out_path = PATTERNS_DIR / "automation_candidates.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[09_patterns] Wrote {out_path}")

    print()
    print("=" * 80)
    print("TOP 15 AUTOMATION CANDIDATES")
    print("=" * 80)
    for i, c in enumerate(candidates[:15], 1):
        print(f"{i:2d}. score={c['score']:.3f}  {c['count']:>4} dictations  "
              f"{c['imperative_density']*100:>5.1f}% imperative  "
              f"{c['unique_days']:>3} days  -> {c['label']}")


if __name__ == "__main__":
    main()
