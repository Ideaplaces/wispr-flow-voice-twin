"""profile.py - load a personal profile from outside the repository.

The voice twin needs a small bag of personal facts (the user's name, taboo
phrases they don't want to be called, glossary fixes for transcription
errors, custom prompt overrides). None of that should live in the repo so
that the same code can run for anyone.

The profile lives at whatever path/URL the user points VOICE_TWIN_PROFILE
at. It can be a local file, an https URL, or any uri requests can reach.
The repo never persists a copy. If VOICE_TWIN_PROFILE is unset, the
loader falls back to ./profile.md if present, otherwise to the bundled
profile.example.md so the system still runs in anonymous mode.

File format: YAML frontmatter for structured fields, plus optional
markdown sections that override per-mode prompt fragments. Example:

    ---
    name: Chip Rarau
    nickname: Chip
    positioning: founder and serial entrepreneur
    taboo_phrases:
      - fractional CTO
      - synergy
    glossary:
      mentally: Mentorly
      claude run: Cloud Run
    ---

    # Custom prompt overrides

    ## blog
    Voice should read like coffee with a senior technical leader...

The frontmatter fields are accessible as attributes on the returned
profile object (profile.name, profile.taboo_phrases, profile.glossary).
The markdown section bodies are accessible via profile.section('blog').
The render() helper performs {{name}} / {{taboo_phrases}} / {{glossary}}
substitution on any prompt template.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROFILE_ENV_VAR = "VOICE_TWIN_PROFILE"


@dataclass
class Profile:
    """Resolved personal profile, loaded once at startup."""

    source: str = "default"
    name: str = "the user"
    nickname: str = ""
    positioning: str = ""
    companies: list[str] = field(default_factory=list)
    taboo_phrases: list[str] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)
    preferred_tone: str = ""
    preferred_punctuation: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def section(self, name: str) -> str:
        """Return the markdown body of `## name` if present, else empty string."""
        return self.sections.get(name.lower(), "")

    def render(self, template: str) -> str:
        """Substitute {{...}} placeholders in a prompt template.

        Supported placeholders:
          {{name}}, {{nickname}}, {{positioning}}, {{preferred_tone}},
          {{preferred_punctuation}}, {{companies}}, {{taboo_phrases}},
          {{glossary}}, {{section.blog}}, {{section.slack}}, etc.

        Lists render as bullet lines; dicts render as 'key -> value' lines.
        Unknown placeholders pass through unchanged so templates can be
        partial-rendered safely.
        """
        ctx = {
            "name": self.name,
            "nickname": self.nickname or self.name,
            "positioning": self.positioning,
            "preferred_tone": self.preferred_tone,
            "preferred_punctuation": self.preferred_punctuation,
            "companies": _render_list(self.companies),
            "taboo_phrases": _render_list(self.taboo_phrases),
            "glossary": _render_glossary(self.glossary),
        }

        out = template
        for key, val in ctx.items():
            out = out.replace(f"{{{{{key}}}}}", str(val) if val is not None else "")

        # Section overrides: {{section.blog}}, {{section.slack}}, ...
        def section_sub(match: re.Match[str]) -> str:
            return self.sections.get(match.group(1).lower(), "")

        out = re.sub(r"\{\{section\.([a-zA-Z0-9_-]+)\}\}", section_sub, out)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "nickname": self.nickname,
            "positioning": self.positioning,
            "companies": self.companies,
            "taboo_phrases": self.taboo_phrases,
            "glossary": self.glossary,
            "preferred_tone": self.preferred_tone,
            "preferred_punctuation": self.preferred_punctuation,
            "sections": list(self.sections.keys()),
            "extra": self.extra,
        }


def _render_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {x}" for x in items)


def _render_glossary(g: dict[str, str]) -> str:
    if not g:
        return "(none)"
    return "\n".join(f"- {k!r} -> {v!r}" for k, v in g.items())


def _read_uri(uri: str) -> str:
    """Read a profile from a local path or any http(s) URL."""
    if uri.startswith(("http://", "https://")):
        import requests
        resp = requests.get(uri, timeout=15)
        resp.raise_for_status()
        return resp.text

    p = Path(os.path.expanduser(uri)).expanduser()
    return p.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Pull YAML frontmatter from the head of a markdown document."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        return {}, text
    yaml_block = parts[0].lstrip("-").lstrip("\n")
    body = parts[1].lstrip("\n") if len(parts) >= 2 else ""
    try:
        front = yaml.safe_load(yaml_block) or {}
    except yaml.YAMLError as e:
        sys.stderr.write(f"[profile] YAML parse error: {e}\n")
        front = {}
    if not isinstance(front, dict):
        front = {}
    return front, body


def _parse_sections(body: str) -> dict[str, str]:
    """Extract `## heading` markdown sections into a flat dict.

    Section names are lowercased. The top-level `# Custom prompt overrides`
    heading and any other H1 are ignored. Anything before the first H2 is
    discarded. This keeps the file readable as a document while letting
    the loader pick out per-mode overrides cleanly.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []

    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1).strip().lower()
            buf = []
            continue
        if current is None:
            continue
        buf.append(line)

    if current is not None:
        sections[current] = "\n".join(buf).strip()

    return sections


def _coerce_glossary(raw: Any) -> dict[str, str]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: dict[str, str] = {}
        for item in raw:
            if isinstance(item, dict) and len(item) == 1:
                k, v = next(iter(item.items()))
                out[str(k)] = str(v)
        return out
    return {}


def _coerce_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    return []


def _build(front: dict, sections: dict[str, str], source: str) -> Profile:
    used = {
        "name", "nickname", "positioning", "companies", "taboo_phrases",
        "glossary", "preferred_tone", "preferred_punctuation",
    }
    extra = {k: v for k, v in front.items() if k not in used}
    return Profile(
        source=source,
        name=str(front.get("name") or "the user"),
        nickname=str(front.get("nickname") or ""),
        positioning=str(front.get("positioning") or ""),
        companies=_coerce_list(front.get("companies")),
        taboo_phrases=_coerce_list(front.get("taboo_phrases")),
        glossary=_coerce_glossary(front.get("glossary")),
        preferred_tone=str(front.get("preferred_tone") or ""),
        preferred_punctuation=str(front.get("preferred_punctuation") or ""),
        sections=sections,
        extra=extra,
    )


_cached_profile: Profile | None = None


def load_profile(force: bool = False) -> Profile:
    """Resolve and cache the user's profile.

    Resolution order:
      1. VOICE_TWIN_PROFILE env var (path or URL)
      2. ./profile.md in the repo root
      3. ./profile.example.md (bundled template, anonymous defaults)
    """
    global _cached_profile
    if _cached_profile is not None and not force:
        return _cached_profile

    repo_root = Path(__file__).resolve().parent

    candidates: list[tuple[str, str]] = []
    env_uri = os.environ.get(PROFILE_ENV_VAR)
    if env_uri:
        candidates.append(("env", env_uri))
    candidates.append(("local", str(repo_root / "profile.md")))
    candidates.append(("example", str(repo_root / "profile.example.md")))

    for label, uri in candidates:
        try:
            text = _read_uri(uri)
        except FileNotFoundError:
            continue
        except Exception as e:
            sys.stderr.write(f"[profile] could not read {label} {uri}: {e}\n")
            continue
        front, body = _split_frontmatter(text)
        sections = _parse_sections(body)
        prof = _build(front, sections, source=f"{label}:{uri}")
        _cached_profile = prof
        return prof

    # Total fallback: empty anonymous profile.
    _cached_profile = Profile(source="empty")
    return _cached_profile


def main() -> None:
    """`python profile.py` prints the resolved profile for debugging."""
    p = load_profile()
    print(json.dumps(p.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
