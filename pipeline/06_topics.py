"""06_topics.py - Discover topical clusters across the embedded corpus.

Pulls every dictation and its precomputed embedding out of Chroma, fits
BERTopic (UMAP + HDBSCAN + c-TF-IDF), and writes:

  data/topics/topic_info.json    one entry per topic (id, count, top words, sample)
  data/topics/doc_topics.jsonl   one row per dictation: {id, topic_id, prob}
  data/topics/model              persisted BERTopic model (reusable)

This sits on top of the existing Chroma collection, so the embedding cost
already paid in pipeline/05_embed.py is reused; nothing is re-embedded.

Tunables (override via env):
  TOPIC_MIN_CLUSTER_SIZE   smallest cluster HDBSCAN will accept (default 40)
  TOPIC_UMAP_NEIGHBORS     UMAP n_neighbors (default 15)
  TOPIC_UMAP_COMPONENTS    UMAP n_components for reduction (default 5)
  TOPIC_LLM_LABELS=1       call Azure OpenAI to name the top topics
  TOPIC_LLM_MAX            how many topics to LLM-name (default 30)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env so Azure keys are available if TOPIC_LLM_LABELS=1
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from config import CHROMA_DIR, DATA_DIR  # noqa: E402

import chromadb  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from hdbscan import HDBSCAN  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402
from umap import UMAP  # noqa: E402

COLLECTION_NAME = "voice_twin_v1"
TOPICS_DIR = DATA_DIR / "topics"
TOPICS_DIR.mkdir(parents=True, exist_ok=True)

MIN_CLUSTER_SIZE = int(os.environ.get("TOPIC_MIN_CLUSTER_SIZE", "40"))
UMAP_NEIGHBORS = int(os.environ.get("TOPIC_UMAP_NEIGHBORS", "15"))
UMAP_COMPONENTS = int(os.environ.get("TOPIC_UMAP_COMPONENTS", "5"))
LLM_LABELS = os.environ.get("TOPIC_LLM_LABELS", "0") == "1"
LLM_MAX = int(os.environ.get("TOPIC_LLM_MAX", "30"))

# Transcription noise that adds nothing to topic semantics
EXTRA_STOPWORDS = {
    "yeah", "okay", "ok", "alright", "um", "uh", "like", "just", "kind",
    "sort", "really", "actually", "basically", "literally", "gonna", "wanna",
    "got", "get", "going", "want", "make", "made", "let", "lets", "thing",
    "things", "stuff", "way", "look", "looks", "see", "saw", "know", "think",
    "thought", "say", "said", "tell", "told", "talk", "talking", "good",
    "right", "lot", "much", "even", "still", "already", "back", "here",
    "there", "yes", "no", "doesn", "don", "didn", "isn", "aren", "wasn",
    "won", "wouldn", "couldn", "shouldn", "ll", "ve", "re", "ve",
}


def _print(msg: str) -> None:
    print(f"[06_topics] {msg}", flush=True)


def load_corpus() -> tuple[list[str], list[str], list[dict], np.ndarray]:
    """Pull every dictation + embedding from the persistent Chroma store."""
    _print(f"Opening Chroma at {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    n = collection.count()
    _print(f"Collection {COLLECTION_NAME!r} has {n} items, fetching")

    res = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = res["ids"]
    docs = res["documents"]
    metas = res["metadatas"] or [{}] * len(ids)
    embeddings = np.array(res["embeddings"], dtype=np.float32)

    _print(f"Pulled {len(ids)} docs, embeddings shape {embeddings.shape}")
    return ids, docs, metas, embeddings


def build_model() -> BERTopic:
    """Configure UMAP + HDBSCAN + vectorizer, return an unfit BERTopic."""
    sklearn_stopwords = list(
        CountVectorizer(stop_words="english").get_stop_words() | EXTRA_STOPWORDS
    )

    umap_model = UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        n_components=UMAP_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        verbose=False,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words=sklearn_stopwords,
        min_df=5,
        ngram_range=(1, 2),
    )

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
        verbose=True,
    )


def fit_topics(model: BERTopic, docs: list[str], embeddings: np.ndarray):
    _print("Fitting BERTopic (UMAP + HDBSCAN + c-TF-IDF)")
    t0 = time.time()
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    _print(f"Fit complete in {time.time() - t0:.1f}s, "
           f"discovered {len(set(topics)) - (1 if -1 in topics else 0)} topics "
           f"({sum(1 for t in topics if t == -1)} outliers)")
    return topics


def maybe_llm_label(model: BERTopic, top_n: int) -> dict[int, str]:
    """Optionally name the top N topics using whichever LLM provider is
    configured. Routes through llm.py so the same code works against
    Azure, OpenAI direct, Anthropic, or local Ollama."""
    if not LLM_LABELS:
        return {}
    try:
        from llm import generate as llm_generate, describe_active_provider
    except Exception as e:
        _print(f"llm.py not importable, skipping LLM labels: {e}")
        return {}

    info = model.get_topic_info().head(top_n + 1)  # +1 because -1 is row 0
    info = info[info.Topic != -1]

    labels: dict[int, str] = {}
    _print(f"LLM-naming top {len(info)} topics via {describe_active_provider()}")
    for _, row in info.iterrows():
        tid = int(row.Topic)
        words = [w for w, _ in model.get_topic(tid)][:10]
        reps = model.get_representative_docs(tid) or []
        sample_block = "\n".join(f"- {d[:240]}" for d in reps[:5])
        prompt = (
            "You are labeling a cluster of voice dictations from one person. "
            "Read the keywords and 5 representative dictations, then output a "
            "short, specific label (3 to 6 words). No quotes, no period.\n\n"
            f"Keywords: {', '.join(words)}\n\n"
            f"Representative dictations:\n{sample_block}\n\n"
            "Label:"
        )
        try:
            text, _src = llm_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2,
            )
            label = (text or "").strip().strip('"').rstrip(".")
            if not label:
                raise RuntimeError("empty label")
            labels[tid] = label
            _print(f"  topic {tid:>3} -> {label}")
        except Exception as e:
            _print(f"  topic {tid:>3} LLM label failed: {e}")
    return labels


def write_artifacts(
    model: BERTopic,
    ids: list[str],
    docs: list[str],
    metas: list[dict],
    topics: list[int],
    llm_labels: dict[int, str],
) -> None:
    info_df = model.get_topic_info()
    topic_info = []
    for _, row in info_df.iterrows():
        tid = int(row.Topic)
        words = [w for w, _ in model.get_topic(tid)] if tid != -1 else []
        reps = model.get_representative_docs(tid) if tid != -1 else []
        topic_info.append({
            "topic_id": tid,
            "count": int(row.Count),
            "name": row.Name,
            "llm_label": llm_labels.get(tid),
            "top_words": words[:15],
            "sample_dictations": [d[:300] for d in (reps or [])[:3]],
        })

    info_path = TOPICS_DIR / "topic_info.json"
    info_path.write_text(json.dumps(topic_info, indent=2, ensure_ascii=False))
    _print(f"Wrote {info_path}")

    doc_topics_path = TOPICS_DIR / "doc_topics.jsonl"
    with doc_topics_path.open("w", encoding="utf-8") as f:
        for did, doc, meta, tid in zip(ids, docs, metas, topics):
            f.write(json.dumps({
                "id": did,
                "topic_id": int(tid),
                "ctx": (meta or {}).get("ctx"),
                "ts": (meta or {}).get("ts"),
                "preview": doc[:160],
            }, ensure_ascii=False) + "\n")
    _print(f"Wrote {doc_topics_path}")

    model_path = TOPICS_DIR / "model"
    model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    _print(f"Saved model to {model_path}")


def print_summary(model: BERTopic, llm_labels: dict[int, str], top_n: int = 30) -> None:
    info = model.get_topic_info().head(top_n + 1)
    info = info[info.Topic != -1]

    print()
    print("=" * 80)
    print(f"TOP {len(info)} TOPICS (by document count)")
    print("=" * 80)
    for _, row in info.iterrows():
        tid = int(row.Topic)
        words = [w for w, _ in model.get_topic(tid)][:8]
        label = llm_labels.get(tid) or " ".join(words[:5])
        print(f"  #{tid:<3} ({row.Count:>4} docs)  {label}")
        print(f"          words: {', '.join(words)}")
    print()


def main() -> None:
    ids, docs, metas, embeddings = load_corpus()
    model = build_model()
    topics = fit_topics(model, docs, embeddings)
    llm_labels = maybe_llm_label(model, top_n=LLM_MAX)
    write_artifacts(model, ids, docs, metas, topics, llm_labels)
    print_summary(model, llm_labels)


if __name__ == "__main__":
    main()
