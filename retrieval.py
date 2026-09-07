"""Hybrid retrieval over the voice corpus.

Dense (Chroma, cosine over text-embedding vectors) and lexical (BM25 over the
same rows) run side by side and are fused by reciprocal rank. Dense finds the
idea however it was phrased; lexical finds the proper noun the embedding
smeared ("Kalitravel" came back 5 times in 20 on dense alone). Filters apply
to both legs so the fused list is the list the caller asked for.

    from retrieval import search
    rows = search("pricing for the tour operators", k=8, human=True, lang=["en"])

Every row: {id, text, meta, sim, sources: {"dense","lexical"}, score}.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import corpus  # noqa: E402  (loads .env before config reads it)
import config  # noqa: E402

BM25_CACHE = config.DATA_DIR / "bm25.pkl"
RRF_K = 60


# ---------------------------------------------------------------------------
# Embedding the query
# ---------------------------------------------------------------------------

_embedder = None


def get_embedder():
    """Same provider the corpus was embedded with, otherwise distances are noise."""
    global _embedder
    if _embedder is not None:
        return _embedder
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
            r = client.embeddings.create(input=texts, model=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
            return [d.embedding for d in r.data]
    elif provider == "openai" and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-large")

        def embed(texts):
            return [d.embedding for d in client.embeddings.create(model=model, input=texts).data]
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(config.LOCAL_EMBED_MODEL)

        def embed(texts):
            return st.encode(texts, normalize_embeddings=True).tolist()
    _embedder = embed
    return embed


# ---------------------------------------------------------------------------
# Lexical leg
# ---------------------------------------------------------------------------

_lexical = None


class LexicalIndex:
    """BM25 over every indexed row, rebuilt whenever history.jsonl changes."""

    def __init__(self, ids: list[str], texts: list[str], metas: list[dict]):
        from rank_bm25 import BM25Okapi
        self.ids = ids
        self.texts = texts
        self.metas = metas
        self.bm25 = BM25Okapi([corpus.tokens(t) for t in texts])

    @classmethod
    def load(cls) -> "LexicalIndex":
        history = config.HISTORY_JSONL
        if BM25_CACHE.exists() and history.exists() and BM25_CACHE.stat().st_mtime >= history.stat().st_mtime:
            try:
                with BM25_CACHE.open("rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        rows = [r for r in corpus.load_history() if (r.get("words") or 0) >= config.MIN_WORDS_FOR_INDEX]
        idx = cls([r["id"] for r in rows], [corpus.index_text(r) for r in rows],
                  [corpus.build_metadata(r) for r in rows])
        try:
            BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with BM25_CACHE.open("wb") as f:
                pickle.dump(idx, f)
        except Exception as e:
            sys.stderr.write(f"[retrieval] could not cache BM25 index ({e})\n")
        return idx

    def query(self, text: str, n: int, accept) -> list[tuple[str, str, dict, float]]:
        q = corpus.tokens(text)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if not accept(self.metas[i]):
                continue
            out.append((self.ids[i], self.texts[i], self.metas[i], float(scores[i])))
            if len(out) >= n:
                break
        return out


def lexical_index() -> LexicalIndex:
    global _lexical
    if _lexical is None:
        _lexical = LexicalIndex.load()
    return _lexical


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def build_filters(human=None, lang=None, kinds=None, ctx=None, min_words=None,
                  since=None, until=None, sources=None):
    """Return (chroma_where, python_predicate) expressing the same constraints."""
    clauses = []
    if human is not None:
        clauses.append({"human": bool(human)})
    if lang:
        langs = [lang] if isinstance(lang, str) else list(lang)
        clauses.append({"lang": {"$in": langs}})
    if kinds:
        ks = [kinds] if isinstance(kinds, str) else list(kinds)
        clauses.append({"kind": {"$in": ks}})
    if ctx:
        clauses.append({"ctx": ctx})
    if sources:
        srcs = [sources] if isinstance(sources, str) else list(sources)
        clauses.append({"source": {"$in": srcs}})
    if min_words:
        clauses.append({"n_words": {"$gte": int(min_words)}})
    if since:
        clauses.append({"ts": {"$gte": since}})
    if until:
        clauses.append({"ts": {"$lte": until}})

    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}

    def accept(meta: dict) -> bool:
        for clause in clauses:
            (field, cond), = clause.items()
            val = meta.get(field)
            if isinstance(cond, dict):
                for op, ref in cond.items():
                    if op == "$in" and val not in ref:
                        return False
                    if op == "$gte" and (val is None or val < ref):
                        return False
                    if op == "$lte" and (val is None or val > ref):
                        return False
            elif val != cond:
                return False
        return True

    return where, accept


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _age_days(ts: str) -> float:
    try:
        return max(0.0, (date.today() - datetime.strptime(ts[:10], "%Y-%m-%d").date()).days)
    except (ValueError, TypeError):
        return 365.0


def search(query: str, k: int = 8, *, queries: list[str] | None = None, human=None, lang=None,
           kinds=None, ctx=None, min_words=None, since=None, until=None, sources=None,
           lexical: bool = True, dedup: bool = True, recency: float = 0.0,
           fetch_multiplier: int = 4) -> list[dict]:
    """Hybrid search. `queries` adds paraphrases fused with the main query.

    recency: weight of a boost that decays linearly to zero over a year, added
    to the fused score. 0.0 means every year counts the same.
    """
    import chromadb
    where, accept = build_filters(human=human, lang=lang, kinds=kinds, ctx=ctx, min_words=min_words,
                                  since=since, until=until, sources=sources)
    fetch_n = max(k * fetch_multiplier, 20)
    all_queries = [query] + [q for q in (queries or []) if q and q != query]

    coll = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(corpus.COLLECTION_NAME)
    embed = get_embedder()
    rows: dict[str, dict] = {}
    rankings: list[list[str]] = []

    q_embs = embed(all_queries)
    for q_emb in q_embs:
        kwargs = dict(query_embeddings=[q_emb], n_results=fetch_n,
                      include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where
        res = coll.query(**kwargs)
        ranking = []
        for did, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - float(dist)
            row = rows.setdefault(did, {"id": did, "text": doc, "meta": meta, "sim": 0.0, "sources": set()})
            row["sim"] = max(row["sim"], sim)
            row["sources"].add("dense")
            ranking.append(did)
        rankings.append(ranking)

    if lexical:
        try:
            idx = lexical_index()
            for q in all_queries:
                ranking = []
                for did, text, meta, _score in idx.query(q, fetch_n, accept):
                    row = rows.setdefault(did, {"id": did, "text": text, "meta": meta, "sim": 0.0, "sources": set()})
                    row["sources"].add("lexical")
                    ranking.append(did)
                rankings.append(ranking)
        except Exception as e:
            sys.stderr.write(f"[retrieval] lexical leg skipped ({e})\n")

    fused = corpus.rrf_fuse(rankings, k=RRF_K)
    for did, row in rows.items():
        score = fused.get(did, 0.0)
        if recency:
            score += recency * (1.0 / RRF_K) * max(0.0, 1.0 - _age_days(row["meta"].get("ts", "")) / 365.0)
        row["score"] = score
        row["sources"] = sorted(row["sources"])

    ordered = sorted(rows.values(), key=lambda r: r["score"], reverse=True)
    if dedup:
        ordered = corpus.collapse_near_duplicates(ordered, key="text")
    return ordered[:k]
