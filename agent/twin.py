"""The voice twin core.

Given a target context (slack / blog / email / coach) and a topic, this
module:
  1. Embeds the topic with the same embedder used for the corpus
  2. Retrieves the K nearest past dictations, biased toward the same context
  3. Loads the per-context style fingerprint and the edit rules
  4. Builds a system prompt that wires all of the above into the model
  5. Calls the configured LLM via llm.generate (any of azure / openai /
     anthropic / ollama)
  6. Applies post-generation rules (em-dash strip, glossary touch-up)

Identity, taboo phrases, positioning, and prompt section overrides come
from the profile loaded by profile.load_profile() so that the same
codebase produces the right voice for whoever has VOICE_TWIN_PROFILE
pointed at their own profile file.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Bootstrap .env before importing config so AZURE_*, OPENAI_*, etc. are in
# os.environ when config reads them. Without this, importing twin from a
# fresh REPL would see config with all-None credentials.
_ENV = ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import config  # noqa: E402
from llm import generate as llm_generate  # noqa: E402
from profile import load_profile  # noqa: E402

PROMPTS_DIR = ROOT / "agent" / "prompts"


def load_prompt(mode: str) -> str:
    p = PROMPTS_DIR / f"{mode}.md"
    if not p.exists():
        raise FileNotFoundError(f"Prompt for mode '{mode}' not found at {p}")
    return p.read_text()


def load_artifacts():
    sp = json.loads(config.STYLE_PROFILE.read_text()) if config.STYLE_PROFILE.exists() else {}
    er = json.loads(config.EDIT_RULES.read_text()) if config.EDIT_RULES.exists() else {}
    gl = json.loads(config.GLOSSARY.read_text()) if config.GLOSSARY.exists() else {}
    return sp, er, gl


# ---------------------------------------------------------------------------
# Retrieval
#
# Two questions, two lookups. "How does the user write this kind of message
# to a person" is answered from human-facing rows of the same kind and
# language (cadence). "What has the user said about this topic" is answered
# from every context (grounding), and is shown to the model as facts only.
# One topic-similarity lookup used to serve both, and since 83% of the corpus
# is instructions to an AI tool, the cadence reference was mostly Cursor talk.
# ---------------------------------------------------------------------------

import corpus  # noqa: E402
import retrieval  # noqa: E402

LONG_FORM_MODES = {"blog", "rewrite", "linkedin", "twitter"}
CADENCE_RECENCY = float(os.environ.get("CADENCE_RECENCY", "0.5"))
GROUNDING_RECENCY = float(os.environ.get("GROUNDING_RECENCY", "0.2"))


def _langs() -> list[str]:
    lang = os.environ.get("VOICE_LANG", "en").lower()
    return [lang, "engb", "unknown"] if lang == "en" else [lang]


def _as_hit(row: dict, role: str) -> dict:
    return {"id": row["id"], "doc": row["text"], "meta": row["meta"], "sim": row["sim"],
            "sources": row.get("sources", []), "role": role}


def retrieve_for_mode(mode: str, topic: str, k: int = None) -> dict:
    k = k or config.RETRIEVE_K
    langs = _langs()
    kind = None
    if mode in LONG_FORM_MODES:
        cadence = retrieval.search(topic, k=k, human=True, lang=langs, min_words=25,
                                   recency=CADENCE_RECENCY)
    else:
        kind = corpus.request_kind(topic)
        kinds = [kind] if kind not in (None, "other", "instruction") else None
        cadence = retrieval.search(topic, k=k, human=True, lang=langs, kinds=kinds,
                                   recency=CADENCE_RECENCY)
        if kinds and len(cadence) < k:
            seen = {r["id"] for r in cadence}
            widen = retrieval.search(topic, k=k, human=True, lang=langs, recency=CADENCE_RECENCY)
            cadence += [r for r in widen if r["id"] not in seen][: k - len(cadence)]
    seen = {r["id"] for r in cadence}
    grounding = [r for r in retrieval.search(topic, k=max(4, k // 2) + len(seen), lang=langs,
                                             recency=GROUNDING_RECENCY) if r["id"] not in seen]
    grounding = grounding[: max(4, k // 2)]
    return {
        "kind": kind,
        "cadence": [_as_hit(r, "cadence") for r in cadence],
        "grounding": [_as_hit(r, "grounding") for r in grounding],
    }


def retrieve(query: str, target_ctx: str = "team_chat", k: int = None):
    """Compatibility shim: cadence rows for a query, human-facing only."""
    return [_as_hit(r, "cadence") for r in retrieval.search(query, k=k or config.RETRIEVE_K,
                                                              human=True, lang=_langs(),
                                                              recency=CADENCE_RECENCY)]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def style_summary(style_profile, target_ctx):
    if not style_profile.get("contexts"):
        return ""
    blocks = []
    for ctx_name in [target_ctx] + [c for c in style_profile["contexts"] if c != target_ctx]:
        c = style_profile["contexts"].get(ctx_name)
        if not c or c["dictations"] < 50:
            continue
        d = c["distinctive_words"][:20]
        bigrams = list(c["top_bigrams"].keys())[:15]
        openers = list(c["top_openers"].keys())[:8]
        closers = list(c["top_closers"].keys())[:8]
        marker = " (TARGET)" if ctx_name == target_ctx else ""
        blocks.append(
            f"[{ctx_name}{marker}] {c['dictations']:,} dictations, {c['wpm']} WPM, "
            f"sent_p50={c['sentence_length']['p50']}, edit_rate={c['edit_rate']*100:.0f}%\n"
            f"  Distinctive words: {', '.join(x['w'] for x in d)}\n"
            f"  Recurring bigrams: {', '.join(bigrams)}\n"
            f"  Signature openers: {', '.join(openers)}\n"
            f"  Signature closers: {', '.join(closers)}"
        )
        if ctx_name == target_ctx:
            break  # only need target + maybe one comparison
    return "\n\n".join(blocks)


def edit_rules_summary(edit_rules):
    if not edit_rules:
        return ""
    rules = []
    rules.append("Punctuation preferences (Wispr -> Chip's actual edits):")
    for sub in edit_rules.get("punctuation_substitutions", [])[:8]:
        rules.append(f"  '{sub['from']}' tends to become '{sub['to']}' ({sub['n']} times)")
    rules.append("")
    rules.append("Hard rules:")
    for r in edit_rules.get("summary_rules_for_prompt", []):
        rules.append(f"  - {r}")
    return "\n".join(rules)


def glossary_summary(glossary: dict, max_entries=30):
    """glossary is wrong -> correct, the profile entries first (see corpus.load_glossary)."""
    if not glossary:
        return ""
    lines = ["Proper-noun glossary (always use the right-hand spelling):"]
    for wrong, correct in list(glossary.items())[:max_entries]:
        lines.append(f"  '{wrong}' -> '{correct}'")
    return "\n".join(lines)


def build_messages(mode: str, topic: str, retrieved=None, body: str = None):
    """Assemble the messages array for the chat completion call."""
    style_profile, edit_rules, glossary = load_artifacts()
    target_ctx = {
        "slack": "team_chat", "discord": "team_chat",
        "linkedin": "team_chat",   # polished, written-for-humans voice
        "twitter": "team_chat",
        "blog": "ai_chat",         # long-form thinking comes from the AI-chat slice
        "rewrite": "ai_chat",      # blog-shaped rewrite of an existing post
        "email": "team_chat", "coach": "ai_chat",
    }.get(mode, "team_chat")

    template = load_prompt(mode)

    def render_hits(hits):
        items = []
        for i, hit in enumerate(hits, 1):
            meta = hit["meta"]
            label = " ".join(x for x in (meta.get("source", ""), meta.get("ctx", ""), meta.get("kind", ""),
                                         meta.get("ts", "")) if x)
            items.append(f"[{i}] ({label})\n{hit['doc']}\n")
        return "\n".join(items)

    retrieved = retrieved or []
    cadence = [h for h in retrieved if h.get("role", "cadence") == "cadence"]
    grounding = [h for h in retrieved if h.get("role") == "grounding"]
    examples_block = render_hits(cadence)
    grounding_block = render_hits(grounding)

    profile = load_profile()
    system = profile.render(
        template,
        extra={
            "style_summary": style_summary(style_profile, target_ctx) or "(no style profile yet)",
            "edit_rules": edit_rules_summary(edit_rules) or "(no edit rules yet)",
            # The static profile glossary already lives at {{glossary}}, but if
            # the auto-generated glossary.json artifact exists, prefer it: it's
            # frequency-weighted from the user's actual transcription edits.
            "glossary": glossary_summary(corpus.load_glossary()) or _format_profile_glossary(profile),
            "examples": examples_block or "(no past messages retrieved)",
            "grounding": grounding_block or "(nothing retrieved on this topic)",
        },
    )

    user_input = topic
    if body:
        user_input = f"{topic}\n\n---\n\n{body}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


# ---------------------------------------------------------------------------
# LLM dispatch (delegates to llm.py for provider routing)
# ---------------------------------------------------------------------------


def _format_profile_glossary(profile) -> str:
    """Render the static profile glossary if no auto-generated one exists."""
    if not profile.glossary:
        return "(none)"
    lines = ["Always use the right-hand spelling:"]
    for wrong, correct in profile.glossary.items():
        lines.append(f"  '{wrong}' -> '{correct}'")
    return "\n".join(lines)


def generate(messages, deployment=None, max_tokens=1200, temperature=0.7, **kwargs):
    """Provider-agnostic chat completion via llm.py.

    `deployment` is honored when the active provider is Azure OpenAI so
    callers can override the deployment per-call (the topics labeler does
    this). For other providers it is ignored.
    """
    extra: dict = {}
    if deployment:
        extra["deployment"] = deployment
        extra["model"] = deployment
    return llm_generate(messages, max_tokens=max_tokens, temperature=temperature, **extra, **kwargs)


# ---------------------------------------------------------------------------
# Post-generation rules
# ---------------------------------------------------------------------------


DASH_PATTERNS = [
    (re.compile(r" *— *"), ", "),
    (re.compile(r" *– *"), ", "),
]


def post_process(text: str, glossary: dict | None = None) -> str:
    # Strip em / en dashes
    for pat, repl in DASH_PATTERNS:
        text = pat.sub(repl, text)
    # Proper nouns: the merged glossary (profile first), whole words only. The
    # raw auto-glossary used to be applied here and turned "Cloud Run" into
    # "claude Run" on every draft.
    if glossary is None:
        glossary = corpus.load_glossary()
    return corpus.normalize_text(text, glossary).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def speak(mode: str, topic: str, body: str = None, k: int = None,
          deployment: str = None, max_tokens: int = 1500):
    """Generate a Chip-flavored draft for the given mode."""
    retrieved = []
    if config.CHROMA_DIR.exists():
        try:
            bundle = retrieve_for_mode(mode, topic, k)
            retrieved = bundle["cadence"] + bundle["grounding"]
        except Exception as e:
            print(f"(retrieval skipped: {e})", file=sys.stderr)

    messages = build_messages(mode, topic, retrieved=retrieved, body=body)
    out, source = generate(messages, deployment=deployment, max_tokens=max_tokens)
    out = post_process(out)
    return out, source, retrieved
