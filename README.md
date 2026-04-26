# wispr-flow-voice-twin

Personal voice twin built on top of my Wispr Flow corpus. Given a topic, retrieves the K nearest past dictations from my own history, loads my computed style fingerprint and edit rules, and generates a draft that reads like I wrote it. Slack messages, blog posts, email replies, and a coach mode that critiques drafts against my baseline voice.

Private. Personal. Chip-only.

## Why

The public sibling repo, [Ideaplaces/wispr-flow-analysis](https://github.com/Ideaplaces/wispr-flow-analysis), surfaces the *aggregate* patterns in the corpus: total volume, two-voices split, time saved, achievements. That repo is for visitors to see what Wispr Flow analytics can look like.

This repo is for actually *using* the corpus. 27,000 dictations and a million words of my own writing turn into a working agent that can produce new text in the same voice.

## How it works

```
Wispr Flow flow.sqlite
        |
        +--> 01_snapshot   safe read-only working copy
        +--> 02_ingest     History table -> JSONL (one record per dictation)
        +--> 03_style      per-context fingerprint (vocab, n-grams, rhythm, openers, closers)
        +--> 04_edit_rules diff formatted vs edited -> punctuation rules + proper-noun glossary
        +--> 05_embed      every dictation -> Chroma vector store (Azure or local embeddings)

Voice CLI
        |
        +--> retrieve    K nearest past dictations, biased to the target context
        +--> load        style profile + edit rules + glossary
        +--> generate    Azure OpenAI gpt-4o-mini / gpt-4o / Claude Sonnet 4.6
        +--> post        strip em dashes, apply glossary, return
```

The model never sees my full corpus. It sees:

1. The system prompt for the chosen mode (slack / blog / email / coach), with my style fingerprint baked in.
2. The K (default 8) retrieved past dictations as voice reference.
3. The user's topic.

That is the whole context. Everything else stays local.

## Modes

| Mode | What it does |
|---|---|
| `slack` | One Slack-shaped message in my voice. Brief, warm, status-update CTO tone. |
| `blog` | A 350-600 word personal blog post for ciprianrarau.com. |
| `email` | An email reply, optionally taking the thread as `--body`. |
| `coach` | Reads a draft, flags every drift from my baseline voice, returns a tightened version. |

Each mode has its own prompt template under `agent/prompts/` that I can edit freely.

## Quick start

```bash
git clone git@github.com:Ideaplaces/wispr-flow-voice-twin.git
cd wispr-flow-voice-twin
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add Azure OpenAI keys

# Build the corpus once
python pipeline/01_snapshot.py
python pipeline/02_ingest.py
python pipeline/03_style_profile.py
python pipeline/04_edit_rules.py
python pipeline/05_embed.py

# Use it
./cli/voice slack "tell the team I'm pushing the C3 mobile fix today"
./cli/voice blog "the two voices: AI vs humans"
./cli/voice coach --body "$(pbpaste)" "tighten this"
```

Full setup details in [SETUP.md](SETUP.md).

## What goes into the system prompt at generation time

A real example, abridged, of what the agent sends to the chat model when I run `voice slack "topic..."`:

```
You are Chip writing his own Slack message. Not Claude. Not an AI. Chip.

ABSOLUTE RULES
- No em dashes or en dashes. Use periods, commas, parentheses.
- No AI attribution.
- No corporate filler ("leveraging", "synergy", ...).
- Never call Chip a "fractional CTO". Founder, serial entrepreneur.

STYLE FINGERPRINT
[team_chat (TARGET)] 3,714 dictations, 104 WPM, sent_p50=7, edit_rate=91%
  Distinctive words: phanie, excellent, woof, thanks, worries, mouhamed, stas, ...
  Recurring bigrams: i'm checking, make sure, mobile app, we're going, ...
  Signature openers: excellent., done!, perfect., i'm checking this, ...
  Signature closers: thank you!, what do you think?, let me know.

EDIT RULES (literal patterns)
  '.' tends to become ',' (55 times)
  '.' tends to become '\n' (27 times)
  '.' tends to become ':' (11 times)
  - Prefer commas, colons, or line breaks over hard periods
  - Add line breaks for vertical breathing room
  - Drop trailing periods from short status messages

GLOSSARY
  'mentally' -> 'Mentorly'
  ...

PAST DICTATIONS BY CHIP, RETRIEVED FOR THIS TOPIC
[1] (team_chat 2026-03-12) sim=0.71
Quick one. I'm checking this now and will get back to you shortly. ...
[2] (team_chat 2026-02-14) sim=0.66
...
```

That structure is what makes the output sound like me instead of like a generic LinkedIn assistant.

## Files

```
wispr-flow-voice-twin/
  config.py
  requirements.txt
  .env.example
  README.md / SETUP.md
  pipeline/
    01_snapshot.py     safe SQLite copy
    02_ingest.py       History -> JSONL
    03_style_profile.py
    04_edit_rules.py
    05_embed.py        Azure or local embeddings into Chroma
  agent/
    twin.py            retrieval + prompt assembly + generation + post-processing
    prompts/
      slack.md
      blog.md
      email.md
      coach.md
  cli/
    voice              entry point: ./cli/voice slack|blog|email|coach "topic"
  data/                generated artifacts, all gitignored
```

## Privacy

Local-first by default. The source `flow.sqlite` and every derived file (JSONL, style profile, edit rules, vector store) live in `data/` and are gitignored. The only network calls are:

- Embedding step (`05_embed.py`): one outbound call per batch of 64 dictations. Set `EMBED_PROVIDER=local` to skip this entirely and use sentence-transformers locally.
- Generation (`cli/voice`): one outbound call per draft, sending only the system prompt and K retrieved examples. Never the full corpus.

If I want zero outbound calls, I can set `EMBED_PROVIDER=local` and use Claude or gpt-4o-mini via a fully isolated proxy.

## Roadmap

- [ ] Phase 5: extract the 16 hours of audio blobs and fine-tune a voice clone (XTTS or ElevenLabs PVC). Then the twin can also *speak* in my voice.
- [ ] Optional FastAPI server on a 5000-9999 port so a Raycast extension or menubar app can call the same backend.
- [ ] Re-rank retrieval with cross-encoder for higher quality.
- [ ] Auto-pick mode (slack vs blog vs email) based on topic length and signal words.
- [ ] Streaming output so long blog drafts feel responsive.
