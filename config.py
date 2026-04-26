"""Configuration for wispr-flow-voice-twin.

All paths and provider settings live here. Override anything via
environment variables or a .env file.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Source data
# ---------------------------------------------------------------------------

FLOW_SQLITE_PATH = os.environ.get(
    "FLOW_SQLITE_PATH",
    str(Path.home() / "Library" / "Application Support" / "Wispr Flow" / "flow.sqlite"),
)
SNAPSHOT_PATH = os.environ.get("SNAPSHOT_PATH", str(ROOT / "data" / "snapshot.sqlite"))

# ---------------------------------------------------------------------------
# Outputs (all gitignored, all stay local)
# ---------------------------------------------------------------------------

DATA_DIR = ROOT / "data"
HISTORY_JSONL = DATA_DIR / "history.jsonl"
STYLE_PROFILE = DATA_DIR / "style_profile.json"
EDIT_RULES = DATA_DIR / "edit_rules.json"
GLOSSARY = DATA_DIR / "glossary.json"
CHROMA_DIR = DATA_DIR / "chroma"

# ---------------------------------------------------------------------------
# App context groups (controls how voice is split during retrieval/style)
# ---------------------------------------------------------------------------

CONTEXT_GROUPS = {
    "ai_chat": {
        "com.todesktop.230313mzl4w4u92",      # Cursor
        "com.microsoft.VSCode",
        "com.anthropic.claudefordesktop",
    },
    "team_chat": {
        "com.tinyspeck.slackmacgap",
        "com.hnc.Discord",
        "org.whispersystems.signal-desktop",
    },
    "personal_chat": {
        "net.whatsapp.WhatsApp",
        "com.apple.MobileSMS",
    },
    "browser": {
        "company.thebrowser.Browser",
        "com.google.Chrome",
        "com.apple.Safari",
    },
}


def context_for(app_id: str) -> str:
    for group, apps in CONTEXT_GROUPS.items():
        if app_id in apps:
            return group
    return "other"


# ---------------------------------------------------------------------------
# Embedding provider
#
#   Default: Azure OpenAI text-embedding-3-large (high quality, cheap, ~$0.50
#   to embed all 27k dictations).
#   Fallback: sentence-transformers all-mpnet-base-v2 (fully local, free).
# ---------------------------------------------------------------------------

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "auto")  # "azure" | "local" | "auto"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ.get(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)
LOCAL_EMBED_MODEL = os.environ.get(
    "LOCAL_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
)

# ---------------------------------------------------------------------------
# Generation provider
# ---------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# How many past dictations to pull as in-context examples for any generation.
RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "8"))

# Minimum word count to bother indexing (filters out yes/no/ack-style fragments).
MIN_WORDS_FOR_INDEX = int(os.environ.get("MIN_WORDS_FOR_INDEX", "5"))

# When generating for a target context, prefer retrieved examples from the
# same context. Set to 1.0 to require, 0.0 to ignore.
SAME_CONTEXT_BIAS = float(os.environ.get("SAME_CONTEXT_BIAS", "0.85"))


def get_flow_sqlite_path() -> Path:
    return Path(FLOW_SQLITE_PATH).expanduser()


def get_snapshot_path() -> Path:
    return Path(SNAPSHOT_PATH).expanduser()
