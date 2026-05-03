# Providers

Two layers of model. Each layer is independent. You can mix any embedding provider with any LLM provider.

| Layer | Local | Azure OpenAI | OpenAI | Anthropic |
|---|---|---|---|---|
| **Embeddings** (per dictation) | sentence-transformers/all-mpnet-base-v2 (768 dim) | text-embedding-3-large (3072 dim) | text-embedding-3-large (3072 dim) | (not used) |
| **LLM** (topic labels, voice generation, coach, suggest) | Ollama (`llama3.1:8b`, `qwen2.5:7b`, etc.) | gpt-4o-mini, gpt-4.1, gpt-5 | gpt-4o-mini, gpt-4o, gpt-4.1 | claude-sonnet-4-6, claude-opus-4-x |

## Selecting providers

```bash
EMBED_PROVIDER=local | azure | openai | auto
LLM_PROVIDER=local | azure | openai | anthropic | ollama | auto
```

`auto` order: azure → openai → anthropic → ollama. The first provider whose credentials (or in Ollama's case, whose local daemon) are reachable is used. If nothing is configured and Ollama is not running, the call raises a clear error telling you what to set.

`local` is an alias for the obvious choice on each layer (sentence-transformers for embeddings, Ollama for LLM).

## Per-provider env vars

### Local (no creds required)

```bash
EMBED_PROVIDER=local
LLM_PROVIDER=ollama

OLLAMA_HOST=http://localhost:11434       # default, override if Ollama is elsewhere
OLLAMA_MODEL=llama3.1:8b                 # default, any pulled model works
LOCAL_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
```

Install Ollama: <https://ollama.com>. Then `ollama pull llama3.1:8b`. The first model fetch is around 4.7 GB. After that, all generation runs on your machine.

### Azure OpenAI

```bash
EMBED_PROVIDER=azure
LLM_PROVIDER=azure

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
```

Reasoning-style deployments (`gpt-5`, `o1`, `o3`) automatically switch to `max_completion_tokens` with a 4x budget; temperature is omitted for those models since the API rejects it.

### OpenAI direct

```bash
EMBED_PROVIDER=openai
LLM_PROVIDER=openai

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini                 # default if unset
OPENAI_EMBED_MODEL=text-embedding-3-large
```

### Anthropic (LLM only)

```bash
LLM_PROVIDER=anthropic

ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-sonnet-4-6        # default if unset
```

Anthropic is LLM-only here. There is no Anthropic embedding provider, so pick one of the other layers for embeddings.

## What each provider costs

Approximate, for a corpus of 25,000 dictations averaging 30 words each.

| Provider | Embeddings (initial 25k) | LLM (per draft) | Topic labels (90 clusters) |
|---|---|---|---|
| Local (sentence-transformers + Ollama 8B) | 0 USD, ~10 min CPU time | 0 USD, ~30 sec on Apple Silicon | 0 USD, ~10 min |
| Azure OpenAI | ~0.50 USD | ~0.01 USD | ~0.30 USD |
| OpenAI direct | ~0.50 USD | ~0.01 USD | ~0.30 USD |
| Anthropic | n/a (LLM only) | ~0.05 USD | ~1.50 USD |

Cloud providers are noticeably faster (one to two orders of magnitude on the LLM layer) and produce somewhat better topic labels. Local is good enough for daily use and free.

## Switching mid-corpus

If you change `EMBED_PROVIDER`, the dimensionality of the new vectors will not match the existing Chroma collection (768 vs 3072). Two options:

1. Rebuild from scratch: `rm -rf data/chroma && python pipeline/05_embed.py`. Cheap when the source data is local.
2. Run two collections side by side. Set `CHROMA_COLLECTION=voice_twin_local_v1` for the local one, leave the Azure one as `voice_twin_v1`. The CLI tools accept the env var.

`LLM_PROVIDER` can be switched freely at any time. It only affects the next call.

## Picking a local LLM

Tested on Apple Silicon and Linux:

- `llama3.1:8b` (4.7 GB) is the default. Decent topic labels, decent voice generation, fast on M-series.
- `qwen2.5:7b` (4.4 GB) is a strong alternative. Slightly better at following structured-output rules.
- `mistral-nemo:12b` (7 GB) is heavier but produces more polished blog drafts.
- Anything smaller than 7B tends to drift on the multi-paragraph voice-generation prompts. Use them for topic labels only.

For coach mode, larger is better. The coach prompt asks for structured markdown analysis; 7B+ models handle it cleanly, smaller models often skip the section structure.
