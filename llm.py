"""llm.py - provider-agnostic chat completion.

Four backends, picked at runtime by the LLM_PROVIDER env var:

    LLM_PROVIDER=auto       # default: pick the first one with credentials
    LLM_PROVIDER=azure      # Azure OpenAI deployment
    LLM_PROVIDER=openai     # OpenAI direct API
    LLM_PROVIDER=anthropic  # Anthropic direct API
    LLM_PROVIDER=ollama     # local Ollama server (no creds required)

In auto mode the order is azure -> openai -> anthropic -> ollama. The
first provider whose credentials (or in Ollama's case, whose local
daemon) are reachable is used.

Public API is one function plus the config:

    text, source = generate(messages, max_tokens=1200, temperature=0.7)

`messages` is the OpenAI-style list of {"role": ..., "content": ...} dicts.
`source` is a human-readable label like "Azure OpenAI (gpt-5)" so callers
can log which provider answered.

Reasoning-style models (gpt-5, o-series) auto-switch to
`max_completion_tokens` and skip temperature, since those parameters
are not accepted there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


# Auto-load .env so this module is usable both from entry-point scripts
# (which would normally load it themselves) and from a fresh REPL.
def _bootstrap_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_bootstrap_env()

import config  # noqa: E402  (imported after env bootstrap)


REASONING_MODEL_PREFIXES = ("gpt-5", "gpt-5.1", "o1", "o1-mini", "o3", "o3-mini")


def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(n.startswith(p) for p in REASONING_MODEL_PREFIXES)


# ---------------------------------------------------------------------------
# Azure OpenAI
# ---------------------------------------------------------------------------


def _azure_credentials_present() -> bool:
    return bool(config.AZURE_OPENAI_API_KEY and config.AZURE_OPENAI_ENDPOINT)


def call_azure(messages, max_tokens=1200, temperature=0.7, deployment=None):
    from openai import AzureOpenAI
    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )
    deployment = deployment or config.AZURE_OPENAI_CHAT_DEPLOYMENT
    kwargs: dict[str, Any] = {"model": deployment, "messages": messages}
    if is_reasoning_model(deployment):
        kwargs["max_completion_tokens"] = max(max_tokens * 4, 4000)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    return text, f"Azure OpenAI ({deployment})"


# ---------------------------------------------------------------------------
# OpenAI direct
# ---------------------------------------------------------------------------


def _openai_credentials_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def call_openai(messages, max_tokens=1200, temperature=0.7, model=None):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if is_reasoning_model(model):
        kwargs["max_completion_tokens"] = max(max_tokens * 4, 4000)
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    return text, f"OpenAI ({model})"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _anthropic_credentials_present() -> bool:
    return bool(config.ANTHROPIC_API_KEY)


def call_anthropic(messages, max_tokens=1200, temperature=0.7, model=None):
    from anthropic import Anthropic
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = model or config.ANTHROPIC_MODEL
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=user_msgs,
    )
    text = resp.content[0].text.strip() if resp.content else ""
    return text, f"Anthropic ({model})"


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------


def _ollama_reachable() -> bool:
    """Is a local Ollama server responding?"""
    import requests
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


def call_ollama(messages, max_tokens=1200, temperature=0.7, model=None):
    import requests
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }
    r = requests.post(f"{base}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    text = (data.get("message", {}).get("content") or "").strip()
    return text, f"Ollama ({model})"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


_AUTO_ORDER = [
    ("azure", _azure_credentials_present, call_azure),
    ("openai", _openai_credentials_present, call_openai),
    ("anthropic", _anthropic_credentials_present, call_anthropic),
    ("ollama", _ollama_reachable, call_ollama),
]


def _resolve_provider() -> str:
    """Return the provider name to use for this call."""
    selected = os.environ.get("LLM_PROVIDER", "auto").lower()
    if selected != "auto":
        return selected
    for name, check, _call in _AUTO_ORDER:
        try:
            if check():
                return name
        except Exception:
            continue
    raise RuntimeError(
        "No LLM provider available. Set one of LLM_PROVIDER=azure / openai / "
        "anthropic / ollama, or run `ollama serve` for the local fallback."
    )


def generate(messages, max_tokens=1200, temperature=0.7, **kwargs):
    """Run a chat completion through whichever provider is configured."""
    provider = _resolve_provider()
    fn_map = {
        "azure": call_azure,
        "openai": call_openai,
        "anthropic": call_anthropic,
        "ollama": call_ollama,
    }
    fn = fn_map.get(provider)
    if fn is None:
        raise RuntimeError(f"Unknown LLM_PROVIDER {provider!r}")
    try:
        return fn(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
    except Exception as e:
        # If we were in auto mode, try the next provider; otherwise re-raise.
        if os.environ.get("LLM_PROVIDER", "auto").lower() != "auto":
            raise
        sys.stderr.write(f"[llm] {provider} failed ({e}), trying next provider\n")
        for name, check, alt_fn in _AUTO_ORDER:
            if name == provider:
                continue
            try:
                if check():
                    return alt_fn(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
            except Exception:
                continue
        raise


def describe_active_provider() -> str:
    """Best-effort label for the provider that would be used right now.

    Used by CLI tools so you can verify which backend a session is on
    without having to make a real call.
    """
    try:
        provider = _resolve_provider()
    except RuntimeError as e:
        return f"none ({e})"

    if provider == "azure":
        return f"Azure OpenAI ({config.AZURE_OPENAI_CHAT_DEPLOYMENT})"
    if provider == "openai":
        return f"OpenAI ({os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')})"
    if provider == "anthropic":
        return f"Anthropic ({config.ANTHROPIC_MODEL})"
    if provider == "ollama":
        return f"Ollama ({os.environ.get('OLLAMA_MODEL', 'llama3.1:8b')})"
    return provider


if __name__ == "__main__":
    print("active provider:", describe_active_provider())
