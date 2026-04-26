# Setup

## 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
chmod +x cli/voice
```

## 2. Configure

```bash
cp .env.example .env
# Edit .env. At minimum, set:
#   AZURE_OPENAI_ENDPOINT
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_CHAT_DEPLOYMENT      (gpt-4o-mini works, gpt-4o better)
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT (text-embedding-3-large recommended)
```

If you have Claude access instead, set `ANTHROPIC_API_KEY` and `ANTHROPIC_MODEL` and the agent will use that. Embeddings fall back to local `sentence-transformers` if Azure is not configured.

## 3. Build the corpus

Five steps, run from the repo root. Each writes to `data/` and is idempotent.

```bash
python pipeline/01_snapshot.py        # safe copy of flow.sqlite
python pipeline/02_ingest.py          # History -> data/history.jsonl
python pipeline/03_style_profile.py   # data/style_profile.json
python pipeline/04_edit_rules.py      # data/edit_rules.json + data/glossary.json
python pipeline/05_embed.py           # data/chroma/ (vector store)
```

The first run takes a few minutes. Subsequent runs only need `01` through `05` again when you want to ingest new dictations.

## 4. Speak

```bash
./cli/voice slack "let the team know I'm taking the morning to work on c3 mobile"
./cli/voice blog "the two voices: how I talk to AI vs how I talk to humans"
./cli/voice email --body "$(pbpaste)" "draft a polite no for now"
./cli/voice coach --body "$(pbpaste)" "tighten this up"
```

Add `--show-retrieved` on any call to see which past dictations the agent pulled in as voice reference. Add `-k 12` to retrieve more examples (default 8).

## 5. Re-running as the corpus grows

The Wispr Flow corpus grows daily. To pull in new dictations:

```bash
python pipeline/01_snapshot.py
python pipeline/02_ingest.py
python pipeline/05_embed.py     # re-embeds the full set; ~$0.50 via Azure
```

You can skip 03 and 04 unless your style is meaningfully shifting (rare on a weekly cadence). Re-run them quarterly.

## 6. Privacy

Everything runs locally. The only network calls are:
- Azure OpenAI for embeddings during step 05 (each dictation goes out as text, gets back a vector). If you do not want that, set `EMBED_PROVIDER=local`.
- Azure OpenAI or Anthropic for generation during `cli/voice` (the system prompt and the K retrieved examples are sent; raw corpus is never bulk-uploaded).

The source `flow.sqlite`, the JSONL, the style profile, the edit rules, and the Chroma collection all stay on disk under `data/` and are gitignored.
