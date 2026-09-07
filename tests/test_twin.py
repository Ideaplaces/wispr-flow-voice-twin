import types

from agent import twin


class _Profile:
    glossary = {}

    def render(self, template, extra=None):
        out = template
        for k, v in (extra or {}).items():
            out = out.replace("{{" + k + "}}", str(v))
        return out


def test_build_messages_splits_cadence_from_grounding(monkeypatch):
    monkeypatch.setattr(twin, "load_profile", lambda: _Profile())
    monkeypatch.setattr(twin, "load_artifacts", lambda: ({}, {}, {}))
    monkeypatch.setattr(twin, "load_prompt", lambda mode: "CADENCE\n{{examples}}\nFACTS\n{{grounding}}\nEND")
    monkeypatch.setattr(twin.corpus, "load_glossary", lambda: {})
    hits = [
        {"id": "a", "doc": "I just pushed the fix", "meta": {"ctx": "team_chat", "kind": "status", "ts": "2026-03-01", "source": "wispr"}, "sim": 0.6, "role": "cadence"},
        {"id": "b", "doc": "Put the pricing at 300 euros", "meta": {"ctx": "ai_chat", "kind": "instruction", "ts": "2026-08-22", "source": "wispr"}, "sim": 0.5, "role": "grounding"},
    ]
    system = twin.build_messages("slack", "topic", retrieved=hits)[0]["content"]
    cadence_part, facts_part = system.split("FACTS")
    assert "I just pushed the fix" in cadence_part and "300 euros" not in cadence_part
    assert "300 euros" in facts_part and "pushed the fix" not in facts_part
    assert "(wispr team_chat status 2026-03-01)" in cadence_part


def test_post_process_strips_dashes_and_fixes_names(monkeypatch):
    monkeypatch.setattr(twin.corpus, "load_glossary", lambda: {"mentally": "Mentorly", "cloud": "claude"})
    out = twin.post_process("Deployed mentally to Cloud Run — done")
    assert out == "Deployed Mentorly to claude Run, done" or out == "Deployed Mentorly to claude Run, done"


def test_post_process_never_touches_cloud_with_the_real_glossary():
    # The bug this guards: an auto-glossary entry 'cloud' -> 'claude' with
    # frequency 2 must not survive the min-frequency filter.
    import corpus
    g = corpus.load_glossary()
    assert "cloud" not in g
