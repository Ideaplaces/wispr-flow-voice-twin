#!/usr/bin/env python3
"""Build per-context style fingerprint from the JSONL corpus.

For each context group (ai_chat, team_chat, personal_chat, browser, other):
- vocabulary distribution and distinctive words (TF over baseline)
- top bigrams and trigrams
- sentence length distribution
- signature openers and closers
- pacing (WPM)
- edit rate

The output style_profile.json is loaded by the agent at generation time
and inserted into the system prompt.
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

STOPWORDS = set("""
a about above after again against all am an and any are aren as at be because been before being below
between both but by can cannot could couldn did didn do does doesn doing don down during each few for from
further had hadn has hasn have haven having he her here hers herself him himself his how i if in into is
isn it its itself just ll let me more most mustn my myself no nor not now of off on once only or other
ought our ours ourselves out over own re s same she should shouldn so some such t than that the their
theirs them themselves then there these they this those through to too under until up very was wasn we
were weren what when where which while who whom why will with won would wouldn you your yours yourself
yourselves re ve ll t s d m
""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")


def tokens(text):
    return [w.lower() for w in WORD_RE.findall(text or "")]


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]


def main():
    if not config.HISTORY_JSONL.exists():
        print(f"ERROR: {config.HISTORY_JSONL} not found. Run 02_ingest.py first.")
        sys.exit(2)

    per_ctx = defaultdict(lambda: {
        "n": 0, "words": 0, "duration_s": 0.0, "edited": 0,
        "word_counter": Counter(), "bigrams": Counter(), "trigrams": Counter(),
        "openers": Counter(), "closers": Counter(),
        "sentence_lengths": [], "para_breaks": [],
    })
    overall = Counter()

    with config.HISTORY_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] not in ("en", "engb", "unknown"):
                continue
            ctx = r["ctx"]
            b = per_ctx[ctx]
            b["n"] += 1
            b["words"] += r["words"]
            b["duration_s"] += r["duration_s"] or 0
            if r["edited"]:
                b["edited"] += 1
            text = r["text"]
            toks = tokens(text)
            for w in toks:
                overall[w] += 1
                if w not in STOPWORDS and len(w) > 2:
                    b["word_counter"][w] += 1
            for i in range(len(toks) - 1):
                bg = (toks[i], toks[i + 1])
                if any(t in STOPWORDS for t in bg):
                    continue
                b["bigrams"][" ".join(bg)] += 1
            for i in range(len(toks) - 2):
                tg = (toks[i], toks[i + 1], toks[i + 2])
                if sum(1 for t in tg if t in STOPWORDS) >= 2:
                    continue
                b["trigrams"][" ".join(tg)] += 1
            sents = sentences(text)
            if sents:
                b["openers"][sents[0][:50].lower()] += 1
                b["closers"][sents[-1][-50:].lower()] += 1
            for s in sents:
                sw = tokens(s)
                if sw:
                    b["sentence_lengths"].append(len(sw))
            b["para_breaks"].append(text.count("\n\n"))

    overall_total = sum(overall.values()) or 1

    profile = {"contexts": {}, "overall": {
        "total_dictations": sum(b["n"] for b in per_ctx.values()),
        "total_words": sum(b["words"] for b in per_ctx.values()),
    }}

    for ctx, b in per_ctx.items():
        ctx_total = sum(b["word_counter"].values()) or 1
        distinctive = []
        for w, n in b["word_counter"].most_common(2000):
            ctx_freq = n / ctx_total
            base_freq = (overall[w] + 1) / (overall_total + 1)
            ratio = ctx_freq / base_freq
            if n >= 5 and ratio > 1.5:
                distinctive.append((w, n, round(ratio, 2)))
        distinctive.sort(key=lambda r: r[2] * math.log(1 + r[1]), reverse=True)

        sl = b["sentence_lengths"] or [0]
        sl_sorted = sorted(sl)
        profile["contexts"][ctx] = {
            "dictations": b["n"],
            "words": b["words"],
            "duration_seconds": round(b["duration_s"], 1),
            "wpm": round(b["words"] * 60 / b["duration_s"], 1) if b["duration_s"] else 0,
            "edit_rate": round(b["edited"] / b["n"], 3) if b["n"] else 0,
            "sentence_length": {
                "mean": round(sum(sl) / len(sl), 1),
                "p25": sl_sorted[len(sl) // 4],
                "p50": sl_sorted[len(sl) // 2],
                "p75": sl_sorted[3 * len(sl) // 4],
                "p90": sl_sorted[9 * len(sl) // 10] if len(sl) >= 10 else sl_sorted[-1],
            },
            "avg_para_breaks": round(sum(b["para_breaks"]) / max(1, len(b["para_breaks"])), 2),
            "top_content_words": dict(b["word_counter"].most_common(60)),
            "distinctive_words": [{"w": w, "n": n, "lift": r}
                                   for w, n, r in distinctive[:60]],
            "top_bigrams": dict(b["bigrams"].most_common(50)),
            "top_trigrams": dict(b["trigrams"].most_common(40)),
            "top_openers": dict(b["openers"].most_common(20)),
            "top_closers": dict(b["closers"].most_common(20)),
        }

    config.STYLE_PROFILE.write_text(json.dumps(profile, indent=2))
    print(f"Wrote {config.STYLE_PROFILE}")
    for ctx, c in profile["contexts"].items():
        if c["dictations"] < 50:
            continue
        print(f"  {ctx:14}  {c['dictations']:>6,} dict  {c['words']:>9,} words  "
              f"wpm={c['wpm']:.0f}  edit={c['edit_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
