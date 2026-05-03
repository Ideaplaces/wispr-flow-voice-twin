# Privacy

This project is local-first by default. A fresh install does not need a single API key, and a fresh install does not make any network calls outside your own machine.

## What stays on your machine, always

- Your Wispr Flow `flow.sqlite` source file. Read via SQLite's online backup API into `data/snapshot.sqlite`. Never uploaded anywhere.
- The derived `data/history.jsonl`, `data/style_profile.json`, `data/edit_rules.json`, `data/glossary.json`. All gitignored, all local.
- The Chroma vector store at `data/chroma/`. Local on disk.
- The BERTopic model at `data/topics/model/`. Local on disk.
- The graph artifact at `data/viz/graph.json` that the explorer reads. Local on disk.
- The Mac → Ubuntu sync goes over your own SSH; no third party touches the bytes in transit.

## What can leave your machine, only if you opt in

- **Embeddings**, when `EMBED_PROVIDER=azure` or `openai`. Each new dictation's text is sent once to the provider's embedding endpoint to produce a vector. The text and the vector are received back; nothing else. With `EMBED_PROVIDER=local` (default), embeddings are produced on-CPU by sentence-transformers and never leave the machine.
- **LLM completions**, when `LLM_PROVIDER=azure | openai | anthropic`. The system prompt (your style fingerprint, edit rules, glossary, retrieved past dictations, and the user's topic) is sent to the provider's chat endpoint to produce a draft. With `LLM_PROVIDER=ollama` (local), all of this happens on your machine via the Ollama daemon.

You can run the entire system, including topic labels and voice generation, without ever sending anything off your machine. We do that ourselves on contributor laptops.

## Where the personal profile lives

`VOICE_TWIN_PROFILE` points at any path or URL. The repo loads it once and renders prompts against its values. The repo never writes a copy back. Common choices:

```bash
VOICE_TWIN_PROFILE=$HOME/.voice-twin/profile.md         # local file
VOICE_TWIN_PROFILE=https://your-azure-blob/profile.md   # private blob
VOICE_TWIN_PROFILE=$HOME/Library/Mobile\ Documents/.../profile.md   # iCloud
```

If you commit your profile inside the repo by mistake, run `git rm --cached profile.md` and add it to `.gitignore`. The bundled `profile.example.md` is the safe public template.

## How to verify

If you want to confirm with your own eyes that the local mode is sealed:

```bash
# On macOS
sudo opensnoop -n python3        # while running pipeline/05_embed.py
# or
sudo lsof -i -P -n | grep python3

# On Linux
sudo strace -f -e trace=network -p $(pgrep -f voice) 2>&1 | grep -v ENOENT
```

In `EMBED_PROVIDER=local LLM_PROVIDER=ollama` mode you should see only loopback traffic to `127.0.0.1:11434` (Ollama) and disk reads. No outbound packets.

## What we never do

- We never ship telemetry. The codebase has no analytics SDK, no error reporter, no "phone home" beacon.
- We never read your Wispr Flow account. We only read your local `flow.sqlite` file. We do not talk to wisprflow.ai's servers.
- We never persist your profile inside the repo.

## Data retention

You own everything. Delete `data/` and the corpus is gone. Delete `~/.voice-twin/profile.md` and your profile is gone. There is no third-party copy.
