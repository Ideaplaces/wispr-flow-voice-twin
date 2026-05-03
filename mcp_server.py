"""mcp_server.py - expose the voice corpus as an MCP server.

Runs in stdio mode so any Claude Desktop / Claude Code / Cursor session
can call into the corpus locally. The whole thing stays on the user's
machine: the MCP server is a child process the client launches, not a
network service.

Tools exposed:
  voice_search           top-K nearest dictations to a free-form query
  voice_topics_list      browse the BERTopic clusters with their LLM labels
  voice_topic_show       full detail for one topic (samples, top words)
  voice_topic_find       best-matching topic for a query + nearby topics
  voice_draft            generate a draft in the user's voice (slack/blog/...)
  voice_coach            critique a draft against the baseline voice
  voice_patterns_list    automation candidates from pipeline/09_patterns.py

Wired up in Claude Desktop with a stanza like:

    {
      "mcpServers": {
        "voice-twin": {
          "command": "/path/to/wispr-flow-voice-twin/.venv/bin/python",
          "args": ["/path/to/wispr-flow-voice-twin/mcp_server.py"]
        }
      }
    }

After that, Claude Desktop sees `voice_search`, `voice_draft`, etc. as
first-class tools and calls them on its own when the conversation
reaches for them. No prompting required.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Bootstrap .env so the underlying agent + provider router get config
_ENV = ROOT / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import Tool, TextContent  # noqa: E402

import config  # noqa: E402


COLLECTION_NAME = "voice_twin_v1"
TOPIC_INFO = config.DATA_DIR / "topics" / "topic_info.json"
DOC_TOPICS = config.DATA_DIR / "topics" / "doc_topics.jsonl"
PATTERNS_FILE = config.DATA_DIR / "patterns" / "automation_candidates.json"


# ---------------------------------------------------------------------------
# Tool implementations: each returns a list[TextContent]
# ---------------------------------------------------------------------------


def _embed_query(query: str):
    """Embed a free-form query with whichever provider is configured."""
    provider = os.environ.get("EMBED_PROVIDER", "auto")
    if provider in ("azure", "auto") and os.environ.get("AZURE_OPENAI_API_KEY"):
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        deployment = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
        return client.embeddings.create(model=deployment, input=[query]).data[0].embedding

    if provider in ("openai",) and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
        return client.embeddings.create(model=model, input=[query]).data[0].embedding

    from sentence_transformers import SentenceTransformer
    model_name = os.environ.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
    st = SentenceTransformer(model_name)
    return st.encode([query], normalize_embeddings=True)[0].tolist()


def tool_search(query: str, k: int = 50, ctx: str | None = None) -> list[TextContent]:
    import chromadb
    qemb = _embed_query(query)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_collection(COLLECTION_NAME)
    where = {"ctx": ctx} if ctx else None
    res = coll.query(
        query_embeddings=[qemb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        **({"where": where} if where else {}),
    )
    rows = []
    for did, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        rows.append({
            "id": did,
            "sim": round(1.0 - float(dist), 4),
            "ts": meta.get("ts", ""),
            "ctx": meta.get("ctx", ""),
            "n_words": meta.get("n_words", 0),
            "text": doc,
        })
    return [TextContent(type="text", text=json.dumps({"query": query, "k": k, "results": rows}, ensure_ascii=False, indent=2))]


def _load_topic_info():
    if not TOPIC_INFO.exists():
        return []
    return json.loads(TOPIC_INFO.read_text())


def tool_topics_list(limit: int = 30) -> list[TextContent]:
    info = [t for t in _load_topic_info() if t["topic_id"] != -1]
    info.sort(key=lambda t: -t["count"])
    info = info[:limit]
    rows = [
        {
            "topic_id": t["topic_id"],
            "label": t.get("llm_label") or " ".join(t.get("top_words", [])[:5]),
            "count": t["count"],
        }
        for t in info
    ]
    return [TextContent(type="text", text=json.dumps(rows, ensure_ascii=False, indent=2))]


def tool_topic_show(topic_id: int, examples: int = 10) -> list[TextContent]:
    info = {t["topic_id"]: t for t in _load_topic_info()}
    t = info.get(topic_id)
    if t is None:
        return [TextContent(type="text", text=json.dumps({"error": f"topic {topic_id} not found"}))]

    out = {
        "topic_id": topic_id,
        "label": t.get("llm_label") or " ".join(t.get("top_words", [])[:5]),
        "count": t["count"],
        "top_words": t.get("top_words", []),
        "sample_dictations": t.get("sample_dictations", []),
    }
    if DOC_TOPICS.exists():
        more = []
        with DOC_TOPICS.open() as f:
            for line in f:
                row = json.loads(line)
                if row["topic_id"] == topic_id:
                    more.append({"ctx": row.get("ctx"), "ts": row.get("ts"), "preview": row.get("preview")})
                    if len(more) >= examples:
                        break
        out["recent_examples"] = more
    return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]


def tool_topic_find(query: str) -> list[TextContent]:
    """Return the best-matching topic plus a few nearby topics."""
    import numpy as np
    from bertopic import BERTopic
    model_path = config.DATA_DIR / "topics" / "model"
    if not model_path.exists():
        return [TextContent(type="text", text=json.dumps({"error": "no topic model on disk; run pipeline/06_topics.py"}))]
    model = BERTopic.load(str(model_path))
    info = {t["topic_id"]: t for t in _load_topic_info()}

    qemb = np.array(_embed_query(query), dtype=np.float32)
    topic_emb = np.array(model.topic_embeddings_, dtype=np.float32)
    valid = sorted(set(info.keys()) - {-1})
    rows_for_id = {tid: idx + 1 for idx, tid in enumerate(sorted(t for t in valid if t != -1))}

    def cos(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    sims = [(tid, cos(qemb, topic_emb[rows_for_id[tid]])) for tid in valid if tid in rows_for_id]
    sims.sort(key=lambda p: -p[1])
    sims = sims[:4]

    best, best_sim = sims[0]
    best_info = info.get(best, {})
    out = {
        "best": {
            "topic_id": best,
            "label": best_info.get("llm_label") or " ".join(best_info.get("top_words", [])[:5]),
            "similarity": round(best_sim, 4),
            "count": best_info.get("count"),
            "sample_dictations": best_info.get("sample_dictations", [])[:3],
        },
        "nearby": [
            {
                "topic_id": tid,
                "label": info.get(tid, {}).get("llm_label") or " ".join(info.get(tid, {}).get("top_words", [])[:5]),
                "similarity": round(sim, 4),
                "count": info.get(tid, {}).get("count"),
            }
            for tid, sim in sims[1:]
        ],
    }
    return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]


def tool_draft(mode: str, topic: str, k: int = 8, body: str | None = None) -> list[TextContent]:
    from agent.twin import speak
    text, source, retrieved = speak(mode=mode, topic=topic, k=k, body=body)
    out = {
        "draft": text,
        "source": source,
        "retrieved": [
            {
                "ctx": r["meta"].get("ctx"),
                "ts": r["meta"].get("ts"),
                "sim": round(r["sim"], 3),
                "doc": r["doc"],
            }
            for r in retrieved
        ],
    }
    return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]


def tool_coach(draft: str, topic: str = "tighten this") -> list[TextContent]:
    from agent.twin import speak
    text, source, retrieved = speak(mode="coach", topic=topic, body=draft)
    out = {"critique": text, "source": source, "retrieved_count": len(retrieved)}
    return [TextContent(type="text", text=json.dumps(out, ensure_ascii=False, indent=2))]


def tool_patterns_list(limit: int = 20, min_density: float | None = None) -> list[TextContent]:
    if not PATTERNS_FILE.exists():
        return [TextContent(type="text", text=json.dumps({"error": "no patterns artifact; run pipeline/09_patterns.py"}))]
    data = json.loads(PATTERNS_FILE.read_text())
    rows = data["candidates"]
    if min_density is not None:
        rows = [r for r in rows if r["imperative_density"] >= min_density]
    rows = rows[:limit]
    return [TextContent(type="text", text=json.dumps({"generated_at": data.get("generated_at"), "candidates": rows}, ensure_ascii=False, indent=2))]


# ---------------------------------------------------------------------------
# MCP server registration
# ---------------------------------------------------------------------------


server = Server("wispr-flow-voice-twin")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="voice_search",
            description=(
                "Semantic search over the user's indexed voice corpus. Returns the top-K nearest "
                "past dictations for a free-form query. Use this when the user asks 'where else "
                "have I talked about X' or wants concrete past phrases on a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query"},
                    "k": {"type": "integer", "default": 20, "description": "How many results to return"},
                    "ctx": {
                        "type": "string",
                        "enum": ["ai_chat", "team_chat", "personal_chat", "browser", "other"],
                        "description": "Optional: restrict to one app context",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="voice_topics_list",
            description=(
                "List the topical clusters discovered in the corpus, ranked by size. Each topic "
                "has an LLM-generated label. Use this to give the user a map of what they think "
                "about most."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 30, "description": "How many topics to return"},
                },
            },
        ),
        Tool(
            name="voice_topic_show",
            description="Show full detail for one topic: top words, sample dictations, recent examples.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_id": {"type": "integer"},
                    "examples": {"type": "integer", "default": 10},
                },
                "required": ["topic_id"],
            },
        ),
        Tool(
            name="voice_topic_find",
            description=(
                "Find the topical cluster that best matches a query, plus three nearby topics for "
                "'hopping' between adjacent themes. Use this when the user wants a topic-level view "
                "rather than individual dictations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="voice_draft",
            description=(
                "Generate a draft in the user's own voice. Pulls retrieval evidence from the "
                "corpus, applies the user's style fingerprint and edit rules, and returns the "
                "draft plus the dictations the model leaned on. Use this whenever drafting "
                "anything that goes out under the user's name."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["slack", "blog", "linkedin", "twitter", "email", "rewrite"],
                    },
                    "topic": {"type": "string"},
                    "k": {"type": "integer", "default": 8},
                    "body": {
                        "type": "string",
                        "description": "Optional body (e.g. for email replies, paste the thread; for rewrite, paste the original post)",
                    },
                },
                "required": ["mode", "topic"],
            },
        ),
        Tool(
            name="voice_coach",
            description=(
                "Critique a draft against the user's baseline voice. Quotes specific drifts "
                "(em-dashes, hedging, corporate filler, taboo phrases) and returns a tightened "
                "version. Use this on any draft before sending."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "draft": {"type": "string", "description": "The draft text to coach"},
                    "topic": {"type": "string", "default": "tighten this", "description": "Optional framing for the coach"},
                },
                "required": ["draft"],
            },
        ),
        Tool(
            name="voice_patterns_list",
            description=(
                "List automation candidates: topics that look like recurring instructions the "
                "user keeps giving across time. Use when the user asks 'what should I automate?' "
                "or wants to find their repeating chores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "min_density": {
                        "type": "number",
                        "description": "Filter to candidates with at least this fraction of imperative dictations (0..1)",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    try:
        if name == "voice_search":
            return tool_search(query=arguments["query"], k=arguments.get("k", 20), ctx=arguments.get("ctx"))
        if name == "voice_topics_list":
            return tool_topics_list(limit=arguments.get("limit", 30))
        if name == "voice_topic_show":
            return tool_topic_show(topic_id=int(arguments["topic_id"]), examples=arguments.get("examples", 10))
        if name == "voice_topic_find":
            return tool_topic_find(query=arguments["query"])
        if name == "voice_draft":
            return tool_draft(
                mode=arguments["mode"],
                topic=arguments["topic"],
                k=arguments.get("k", 8),
                body=arguments.get("body"),
            )
        if name == "voice_coach":
            return tool_coach(draft=arguments["draft"], topic=arguments.get("topic", "tighten this"))
        if name == "voice_patterns_list":
            return tool_patterns_list(
                limit=arguments.get("limit", 20),
                min_density=arguments.get("min_density"),
            )
        return [TextContent(type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({"error": f"{type(e).__name__}: {e}"}))]


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
