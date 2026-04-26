#!/usr/bin/env python3
"""Extract explicit style rules by diffing formatted vs edited text.

The diff between Wispr's formatted output and what Chip actually shipped is
the single cleanest signal of his style preferences. Common patterns become
literal rules baked into the agent's system prompt.

Rules surfaced:
- punctuation substitutions (period -> comma, period -> linebreak, etc.)
- linebreak insertions (Chip prefers vertical breathing room)
- proper-noun fixes (e.g. "mentally" -> "Mentorly")
- short word swaps that recur

Output: data/edit_rules.json plus a glossary of recurring proper-noun corrections.
"""

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

WORD_RE = re.compile(r"[A-Za-z']+|[.,;:!?\-—–\(\)\"\n]")


def toks(text):
    return WORD_RE.findall(text or "")


def main():
    if not config.HISTORY_JSONL.exists():
        print(f"ERROR: {config.HISTORY_JSONL} not found.")
        sys.exit(2)

    punct_subs = Counter()
    linebreaks_added = 0
    word_swaps = Counter()
    word_inserts = Counter()
    word_deletes = Counter()
    rows = 0

    with config.HISTORY_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if not (r.get("formatted_text") and r.get("edited_text")):
                continue
            f_txt, e_txt = r["formatted_text"], r["edited_text"]
            if f_txt.strip() == e_txt.strip():
                continue
            rows += 1

            if "\n" in e_txt and "\n" not in f_txt:
                linebreaks_added += 1

            f_tk = toks(f_txt)
            e_tk = toks(e_txt)
            sm = difflib.SequenceMatcher(a=f_tk, b=e_tk, autojunk=False)
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    continue
                before = f_tk[i1:i2]
                after = e_tk[j1:j2]
                if tag == "replace" and len(before) == 1 and len(after) == 1:
                    if before[0] in ".,;:!?\n" or after[0] in ".,;:!?\n":
                        punct_subs[(before[0], after[0])] += 1
                if tag == "replace" and 1 <= len(before) <= 3 and 1 <= len(after) <= 3:
                    word_swaps[(" ".join(before).lower(),
                                 " ".join(after).lower())] += 1
                if tag == "insert" and 1 <= len(after) <= 4:
                    word_inserts[" ".join(after).lower()] += 1
                if tag == "delete" and 1 <= len(before) <= 4:
                    word_deletes[" ".join(before).lower()] += 1

    # Build glossary: recurring "thing -> Thing" or "wrong -> right" word swaps
    glossary = {}
    for (a, b), n in word_swaps.most_common(200):
        if n < 2:
            continue
        if a == b:
            continue
        if not a or not b:
            continue
        if " " in a or " " in b:
            continue
        if a.lower() == b.lower():
            continue
        glossary[a] = {"correct_to": b, "frequency": n}

    rules = {
        "rows_with_real_edits": rows,
        "punctuation_substitutions": [
            {"from": k[0].replace("\n", "\\n"),
             "to": k[1].replace("\n", "\\n"), "n": n}
            for k, n in punct_subs.most_common(20)
        ],
        "linebreaks_added_count": linebreaks_added,
        "top_word_swaps": [
            {"from": k[0], "to": k[1], "n": n}
            for k, n in word_swaps.most_common(30)
        ],
        "top_word_inserts": dict(word_inserts.most_common(30)),
        "top_word_deletes": dict(word_deletes.most_common(30)),
        "summary_rules_for_prompt": [
            "Prefer commas, colons, or line breaks over hard periods when joining clauses.",
            "Add line breaks to give paragraphs vertical breathing room.",
            "Drop trailing periods from short status messages.",
            "Keep proper nouns intact via the glossary.",
        ],
    }

    config.STYLE_PROFILE.parent.mkdir(parents=True, exist_ok=True)
    config.EDIT_RULES.write_text(json.dumps(rules, indent=2))
    config.GLOSSARY.write_text(json.dumps(glossary, indent=2))
    print(f"Wrote {config.EDIT_RULES} and {config.GLOSSARY}")
    print(f"  Rows analyzed: {rows:,}  Linebreaks added: {linebreaks_added}")
    print(f"  Top punctuation subs:")
    for sub in rules["punctuation_substitutions"][:6]:
        print(f"    {sub['from']!r} -> {sub['to']!r}  ({sub['n']})")
    print(f"  Glossary entries: {len(glossary)}")


if __name__ == "__main__":
    main()
