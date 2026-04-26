"""The voice twin core.

Given a target context (slack / blog / email / coach) and a topic, this
module:
  1. Embeds the topic with the same embedder used for the corpus
  2. Retrieves the K nearest past dictations, biased toward the same context
  3. Loads the per-context style fingerprint and the edit rules
  4. Builds a system prompt that wires all of the above into the model
  5. Calls the configured LLM (Azure OpenAI primary, Anthropic fallback)
  6. Applies post-generation rules (em-dash strip, glossary touch-up)
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

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
# ---------------------------------------------------------------------------


def get_embedder():
    provider = config.EMBED_PROVIDER
    if provider == "auto":
        provider = "azure" if (config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT) else "local"

    if provider == "azure":
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            api_version=config.AZURE_OPENAI_API_VERSION,
        )

        def embed(texts):
            r = client.embeddings.create(
                input=texts, model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
            return [d.embedding for d in r.data]

        return embed

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

    def embed(texts):
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed


def retrieve(query: str, target_ctx: str, k: int = None):
    import chromadb
    k = k or config.RETRIEVE_K
    embed = get_embedder()
    q_emb = embed([query])[0]

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_collection("voice_twin_v1")

    # Pull a generous pool, then re-rank with the same-context bias.
    pool = coll.query(query_embeddings=[q_emb], n_results=max(50, k * 6))
    docs = pool["documents"][0]
    metas = pool["metadatas"][0]
    dists = pool["distances"][0]

    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1 - dist
        ctx_match = meta.get("ctx") == target_ctx
        adjusted = sim + (config.SAME_CONTEXT_BIAS if ctx_match else 0) * 0.05
        scored.append({"doc": doc, "meta": meta, "sim": sim, "adjusted": adjusted})

    scored.sort(key=lambda r: r["adjusted"], reverse=True)
    return scored[:k]


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


def glossary_summary(glossary, max_entries=15):
    if not glossary:
        return ""
    items = sorted(glossary.items(), key=lambda kv: -kv[1]["frequency"])[:max_entries]
    lines = ["Proper-noun glossary (always use the right-hand spelling):"]
    for wrong, info in items:
        lines.append(f"  '{wrong}' -> '{info['correct_to']}'")
    return "\n".join(lines)


def build_messages(mode: str, topic: str, retrieved=None, body: str = None):
    """Assemble the messages array for the chat completion call."""
    style_profile, edit_rules, glossary = load_artifacts()
    target_ctx = {
        "slack": "team_chat", "discord": "team_chat",
        "linkedin": "team_chat",   # polished, written-for-humans voice
        "twitter": "team_chat",
        "blog": "ai_chat",         # long-form thinking comes from the AI-chat slice
        "email": "team_chat", "coach": "ai_chat",
    }.get(mode, "team_chat")

    template = load_prompt(mode)

    examples_block = ""
    if retrieved:
        items = []
        for i, hit in enumerate(retrieved, 1):
            ts = hit["meta"].get("ts", "")
            ctx = hit["meta"].get("ctx", "")
            items.append(f"[{i}] ({ctx} {ts}) sim={hit['sim']:.2f}\n{hit['doc']}\n")
        examples_block = "\n".join(items)

    system = template.format(
        style_summary=style_summary(style_profile, target_ctx),
        edit_rules=edit_rules_summary(edit_rules),
        glossary=glossary_summary(glossary),
        examples=examples_block or "(no past dictations retrieved)",
    )

    user_input = topic
    if body:
        user_input = f"{topic}\n\n---\n\n{body}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]


# ---------------------------------------------------------------------------
# LLM dispatch
# ---------------------------------------------------------------------------


def call_azure(messages, max_tokens=1200, temperature=0.7):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )
    deployment = config.AZURE_OPENAI_CHAT_DEPLOYMENT
    resp = client.chat.completions.create(
        model=deployment, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content.strip(), f"Azure OpenAI ({deployment})"


def call_anthropic(messages, max_tokens=1200, temperature=0.7):
    from anthropic import Anthropic
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL, max_tokens=max_tokens,
        temperature=temperature, system=system, messages=user_msgs,
    )
    return resp.content[0].text.strip(), f"Anthropic ({config.ANTHROPIC_MODEL})"


def generate(messages, **kwargs):
    if config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT:
        try:
            return call_azure(messages, **kwargs)
        except Exception as e:
            print(f"  Azure call failed: {e}", file=sys.stderr)
    if config.ANTHROPIC_API_KEY:
        return call_anthropic(messages, **kwargs)
    raise RuntimeError("No LLM provider configured. Set AZURE_OPENAI_* or ANTHROPIC_API_KEY.")


# ---------------------------------------------------------------------------
# Post-generation rules
# ---------------------------------------------------------------------------


DASH_PATTERNS = [
    (re.compile(r" *— *"), ", "),
    (re.compile(r" *– *"), ", "),
]


def post_process(text: str, glossary: dict) -> str:
    # Strip em / en dashes
    for pat, repl in DASH_PATTERNS:
        text = pat.sub(repl, text)
    # Apply glossary substitutions (case-insensitive whole-word match)
    for wrong, info in glossary.items():
        if not wrong:
            continue
        pattern = re.compile(rf"\b{re.escape(wrong)}\b", flags=re.IGNORECASE)
        text = pattern.sub(info["correct_to"], text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def speak(mode: str, topic: str, body: str = None, k: int = None):
    """Generate a Chip-flavored draft for the given mode."""
    target_ctx = {"slack": "team_chat", "linkedin": "team_chat",
                  "twitter": "team_chat", "blog": "ai_chat",
                  "email": "team_chat", "coach": "ai_chat"}.get(mode, "team_chat")
    if config.CHROMA_DIR.exists():
        try:
            retrieved = retrieve(topic, target_ctx, k)
        except Exception as e:
            print(f"(retrieval skipped: {e})", file=sys.stderr)
            retrieved = []
    else:
        retrieved = []

    messages = build_messages(mode, topic, retrieved=retrieved, body=body)
    out, source = generate(messages)
    _, _, glossary = load_artifacts()
    out = post_process(out, glossary)
    return out, source, retrieved
