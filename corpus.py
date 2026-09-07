"""Shared helpers for the voice corpus: what a row is, how it is cleaned,
which rows are a person talking to people, and how they are labelled.

Every pipeline step and every CLI used to carry its own copy of these rules,
which is how the index ended up holding raw transcriptions ("Cali Travel"),
HTML tags in the Slack closers, and no way to tell a message to a colleague
from an instruction to Cursor. One module, one set of rules.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(env_path: Path | None = None) -> None:
    """Load KEY=value lines from .env into os.environ without overriding."""
    path = env_path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

COLLECTION_NAME = "voice_twin_v1"

# ---------------------------------------------------------------------------
# Who was the row addressed to
# ---------------------------------------------------------------------------

HUMAN_CONTEXTS = {"team_chat", "personal_chat", "email"}
HUMAN_URL_RE = re.compile(
    r"(mail\.google\.com|linkedin\.com|facebook\.com|grip\.events|twitter\.com|x\.com|"
    r"substack\.com|outlook\.(live|office)\.com|discord\.com/channels)",
    re.I,
)
HUMAN_APPS = {"ru.keepcoder.Telegram", "com.monday.desktop", "com.apple.mail"}
HUMAN_SOURCES = {"whatsapp", "gmail", "slack", "discord", "telegram"}


def is_human(rec: dict) -> bool:
    """True when the row is the user writing to a person, not to an AI tool."""
    if rec.get("source", "wispr") in HUMAN_SOURCES:
        return True
    if rec.get("ctx") in HUMAN_CONTEXTS:
        return True
    if rec.get("ctx") == "browser" and HUMAN_URL_RE.search(rec.get("url") or ""):
        return True
    if rec.get("app") in HUMAN_APPS:
        return True
    return False


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>|<![^<>]*>")
HTML_BLOCK_RE = re.compile(r"<(style|script|head)\b[^>]*>.*?</\1\s*>", re.I | re.S)
BOM = "﻿"
WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’\-])*", re.UNICODE)


def strip_html(text: str) -> str:
    """Drop markup that Slack and rich-text editors leak into dictations."""
    if not text or ("<" not in text and "&" not in text):
        return text
    out = HTML_BLOCK_RE.sub(" ", text)
    out = re.sub(r"<br\s*/?>|</p>|</li>|</div>", "\n", out, flags=re.I)
    out = HTML_TAG_RE.sub("", out)
    out = html.unescape(out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_text(text: str) -> str:
    return strip_html((text or "").replace(BOM, "")).strip()


def tokens(text: str) -> list[str]:
    """Unicode-aware word tokens, lowercased. Keeps 'Stéphanie' whole."""
    return [w.lower() for w in WORD_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# Proper-noun normalization
# ---------------------------------------------------------------------------


def load_glossary(min_auto_frequency: int = 3) -> dict[str, str]:
    """Wrong spelling -> canonical spelling.

    The hand-written profile glossary wins. Entries from the auto-generated
    data/glossary.json are added only when they do not collide with it and
    recur enough to be a pattern rather than a one-off edit.
    """
    merged: dict[str, str] = {}
    try:
        from profile import load_profile
        for wrong, correct in (load_profile().glossary or {}).items():
            merged[wrong.strip().lower()] = correct.strip()
    except Exception:
        pass
    if config.GLOSSARY.exists():
        try:
            auto = json.loads(config.GLOSSARY.read_text())
        except json.JSONDecodeError:
            auto = {}
        for wrong, info in auto.items():
            key = wrong.strip().lower()
            if key in merged or not key:
                continue
            if (info.get("frequency") or 0) < min_auto_frequency:
                continue
            merged[key] = info["correct_to"]
    return merged


def _glossary_pattern(glossary: dict[str, str]) -> re.Pattern | None:
    if not glossary:
        return None
    keys = sorted(glossary, key=len, reverse=True)
    alternation = "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in keys)
    return re.compile(rf"(?<![\w'’])(?:{alternation})(?![\w'’])", re.I)


def normalize_text(text: str, glossary: dict[str, str]) -> str:
    """Rewrite misheard names to their canonical spelling, whole words only."""
    pattern = _glossary_pattern(glossary)
    if not text or pattern is None:
        return text or ""

    def repl(m: re.Match) -> str:
        key = re.sub(r"\s+", " ", m.group(0).lower())
        return glossary.get(key, m.group(0))

    return pattern.sub(repl, text)


# ---------------------------------------------------------------------------
# Message kind
# ---------------------------------------------------------------------------

KINDS = ("ack", "thanks", "status", "ask", "scheduling", "opinion", "explanation",
         "instruction", "other")

_ACK_RE = re.compile(
    r"^(yes|no|ok|okay|sure|done|perfect|excellent|great|sounds good|got it|right|yep|nope|"
    r"absolutely|will do|on it|cool|noted|awesome|nice|good)\b", re.I)
_THANKS_RE = re.compile(r"\b(thank you|thanks|thx|merci|mulțumesc|multumesc|mersi)\b", re.I)
_SCHEDULE_RE = re.compile(
    r"\b(tomorrow|next week|this week|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"this afternoon|tonight|this morning|\d{1,2}\s?(am|pm)|\d{1,2}:\d{2}|schedule|meet|call|"
    r"calendar|minutes|reschedule|slot|available)\b", re.I)
_SCHEDULE_INTENT_RE = re.compile(r"\b(let'?s|can we|shall we|want to|do you want|when|book|are you)\b", re.I)
_STATUS_RE = re.compile(
    r"^(i'?m |i am |i'?ve |i have |i just |just |i will |i'?ll |pushed|pushing|deployed|deploying|"
    r"fixed|fixing|merged|merging|working on|done with|finished|it'?s (done|deployed|pushed|live|fixed|working))",
    re.I)


def rule_kind(text: str, human: bool) -> str | None:
    """Cheap first pass. None means the rules are not sure; ask the model."""
    if not human:
        return "instruction"
    t = (text or "").strip()
    if not t:
        return "other"
    low = t.lower()
    n_words = len(low.split())
    if n_words <= 4 and _ACK_RE.match(low):
        return "ack"
    if n_words <= 12 and _THANKS_RE.search(low):
        return "thanks"
    if _SCHEDULE_RE.search(low) and (_SCHEDULE_INTENT_RE.search(low) or low.endswith("?")):
        return "scheduling"
    if low.endswith("?") and n_words <= 30:
        return "ask"
    if _STATUS_RE.match(low) and n_words <= 40:
        return "status"
    return None


_CLASSIFY_PROMPT = """You label short messages a person wrote to other people. For each numbered
message pick exactly one label from this list:

ack: a bare acknowledgement ("done", "perfect", "yes go ahead")
thanks: mainly a thank-you
status: reports what the writer did, is doing, or will do
ask: asks the reader for something or for information
scheduling: about when to meet, call, or do something
opinion: a judgement, a take, a recommendation
explanation: explains how or why something works, teaches, describes
instruction: tells the reader to do something (an order or a request to act)
other: none of the above

Return a JSON array of {n} strings, in order, labels only, nothing else.

Messages:
{items}"""

_REQUEST_PROMPT = """A person is about to write a message and described what it should say. The
description is phrased as an instruction to the writer ("tell X that...", "message Y
about..."); label the message that will be written, not the description.
One label from: ack, thanks, status, ask, scheduling, opinion, explanation, instruction, other.
instruction means the finished message itself tells the reader to do something.
Answer with the label only.

Description: {topic}"""


def _classify_call(prompt: str, max_tokens: int) -> str:
    import llm
    kwargs = {}
    deployment = os.environ.get("CLASSIFY_DEPLOYMENT")
    if deployment:
        kwargs["deployment"] = deployment
        kwargs["model"] = deployment
    out = llm.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                       temperature=0.0, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def classify_kinds_llm(texts: list[str], batch: int = 40) -> list[str]:
    """Label human-facing rows the rules could not settle. Best effort: any
    failure or malformed answer falls back to 'other' for that batch."""
    labels: list[str] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        items = "\n".join(f"{n + 1}. {t.strip()[:400]!r}" for n, t in enumerate(chunk))
        try:
            raw = _classify_call(_CLASSIFY_PROMPT.format(n=len(chunk), items=items), max_tokens=600)
            m = re.search(r"\[.*\]", raw, re.S)
            got = json.loads(m.group(0)) if m else []
            got = [g if g in KINDS else "other" for g in got]
        except Exception as e:  # provider down, bad JSON: keep going
            sys.stderr.write(f"[corpus] kind classification failed for a batch ({e})\n")
            got = []
        got = (got + ["other"] * len(chunk))[: len(chunk)]
        labels.extend(got)
    return labels


def request_kind(topic: str) -> str:
    """Kind of message the user is asking the twin to write."""
    guess = rule_kind(topic, human=True)
    if guess:
        return guess
    try:
        raw = _classify_call(_REQUEST_PROMPT.format(topic=topic.strip()[:600]), max_tokens=20)
        label = raw.strip().strip('"').lower().split()[0].strip(".,") if raw.strip() else "other"
        return label if label in KINDS else "other"
    except Exception:
        return "other"


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def enrich(rec: dict, glossary: dict[str, str]) -> dict:
    """Add the derived fields every consumer relies on. Idempotent."""
    rec.setdefault("source", "wispr")
    rec["text"] = clean_text(rec.get("text") or "")
    rec["human"] = is_human(rec)
    # Only dictation carries mishearings; typed sources (WhatsApp, email)
    # spell names the way the writer typed them.
    if rec["source"] == "wispr":
        rec["text_norm"] = normalize_text(rec["text"], glossary)
    else:
        rec["text_norm"] = rec["text"]
    if not rec.get("kind"):
        rec["kind"] = rule_kind(rec["text"], rec["human"])
    return rec


def index_text(rec: dict) -> str:
    return rec.get("text_norm") or rec.get("text") or ""


def build_metadata(rec: dict) -> dict:
    return {
        "ctx": rec.get("ctx") or "other",
        "app": rec.get("app") or "",
        "ts": (rec.get("ts") or "")[:10],
        "edited": bool(rec.get("edited")),
        "n_words": int(rec.get("words") or 0),
        "lang": rec.get("lang") or "unknown",
        "human": bool(rec.get("human")),
        "kind": rec.get("kind") or "other",
        "source": rec.get("source") or "wispr",
    }


def load_history(path: Path | None = None) -> list[dict]:
    """Every row, one per id. When an id was appended twice the last copy wins."""
    path = path or config.HISTORY_JSONL
    by_id: dict[str, dict] = {}
    if not path.exists():
        return []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_id[r["id"]] = r
    return list(by_id.values())


def write_history(rows: list[dict], path: Path | None = None) -> None:
    path = path or config.HISTORY_JSONL
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def history_ids(path: Path | None = None) -> set[str]:
    path = path or config.HISTORY_JSONL
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------


def iter_collection(coll, include=("documents", "metadatas"), batch: int = 5000):
    """Yield (id, document, metadata, embedding) for every row, in pages.

    A single coll.get() over the whole collection dies past a few tens of
    thousands of rows with "too many SQL variables"; this is what killed the
    weekly topic refit for five weeks.
    """
    include = list(include)
    offset = 0
    while True:
        res = coll.get(include=include, limit=batch, offset=offset)
        ids = res["ids"]
        if not ids:
            break
        docs = res.get("documents") or [None] * len(ids)
        metas = res.get("metadatas") or [{}] * len(ids)
        embs = res.get("embeddings")
        embs = list(embs) if embs is not None else [None] * len(ids)
        for row in zip(ids, docs, metas, embs):
            yield row
        if len(ids) < batch:
            break
        offset += len(ids)


def collection_ids(coll, batch: int = 5000) -> set[str]:
    return {row[0] for row in iter_collection(coll, include=(), batch=batch)}


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def collapse_near_duplicates(rows: list[dict], key: str = "text", threshold: float = 0.92) -> list[dict]:
    """Drop rows whose text is nearly identical to an already-kept row."""
    kept: list[dict] = []
    norms: list[str] = []
    for row in rows:
        text = " ".join((row.get(key) or "").lower().split())
        duplicate = False
        for other in norms:
            if abs(len(text) - len(other)) / max(len(text), len(other), 1) > 0.3:
                continue
            if SequenceMatcher(None, text, other).ratio() >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
            norms.append(text)
    return kept


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion over several ranked id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, did in enumerate(ranking):
            scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
    return scores
