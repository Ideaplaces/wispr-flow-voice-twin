import corpus


def test_strip_html_removes_tags_and_keeps_text():
    assert corpus.strip_html("<ul><li>one</li><li>two</li></ul>") == "one\ntwo"
    assert corpus.strip_html("plain text") == "plain text"
    assert corpus.strip_html("a &amp; b") == "a & b"
    full = "<!DOCTYPE html><html><head><style>body { color: #333; }</style></head><body><p>Hey Ashley!</p></body></html>"
    assert corpus.strip_html(full) == "Hey Ashley!"


def test_tokens_keep_accented_names_whole():
    assert corpus.tokens("Merci Stéphanie, c'est bon") == ["merci", "stéphanie", "c'est", "bon"]


def test_normalize_text_rewrites_whole_words_only():
    g = {"mentally": "Mentorly", "cali travel": "Kalitravel", "claude run": "Cloud Run"}
    assert corpus.normalize_text("I fixed mentally today", g) == "I fixed Mentorly today"
    assert corpus.normalize_text("Send it to Cali  Travel", g) == "Send it to Kalitravel"
    assert corpus.normalize_text("deploy on claude run", g) == "deploy on Cloud Run"
    assert corpus.normalize_text("he acted mentallyish", g) == "he acted mentallyish"
    assert corpus.normalize_text("Cloud Run is fine", g) == "Cloud Run is fine"


def test_is_human_uses_context_url_app_and_source():
    assert corpus.is_human({"ctx": "team_chat"})
    assert corpus.is_human({"ctx": "browser", "url": "https://mail.google.com/mail/u/0/#inbox"})
    assert corpus.is_human({"ctx": "other", "app": "ru.keepcoder.Telegram"})
    assert corpus.is_human({"ctx": "ai_chat", "source": "whatsapp"})
    assert not corpus.is_human({"ctx": "ai_chat", "app": "com.microsoft.VSCode"})
    assert not corpus.is_human({"ctx": "browser", "url": "https://github.com/x"})


def test_rule_kind():
    assert corpus.rule_kind("anything", human=False) == "instruction"
    assert corpus.rule_kind("Perfect!", human=True) == "ack"
    assert corpus.rule_kind("Thanks a lot for the review", human=True) == "thanks"
    assert corpus.rule_kind("Let's meet tomorrow at 3pm", human=True) == "scheduling"
    assert corpus.rule_kind("Did you merge the PR?", human=True) == "ask"
    assert corpus.rule_kind("I'm pushing this fix to production", human=True) == "status"
    assert corpus.rule_kind("The pricing model has three layers and the second one matters most", human=True) is None


def test_enrich_and_metadata():
    rec = {"id": "1", "ctx": "team_chat", "app": "slack", "url": "", "ts": "2026-03-01 10:00:00",
           "lang": "en", "words": 5, "edited": False, "text": "<b>Done</b> with mentally"}
    corpus.enrich(rec, {"mentally": "Mentorly"})
    assert rec["text"] == "Done with mentally"
    assert rec["text_norm"] == "Done with Mentorly"
    assert rec["human"] is True
    assert rec["kind"] == "ack"
    meta = corpus.build_metadata(rec)
    assert meta == {"ctx": "team_chat", "app": "slack", "ts": "2026-03-01", "edited": False,
                    "n_words": 5, "lang": "en", "human": True, "kind": "ack", "source": "wispr"}


def test_load_history_last_copy_wins(tmp_path):
    p = tmp_path / "h.jsonl"
    p.write_text('{"id":"a","text":"old"}\n{"id":"b","text":"b"}\n{"id":"a","text":"new"}\n')
    rows = corpus.load_history(p)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["text"] == "new"


def test_collapse_near_duplicates_and_rrf():
    rows = [{"text": "I just pushed the fix"}, {"text": "I just pushed the fix."}, {"text": "Totally different"}]
    assert len(corpus.collapse_near_duplicates(rows)) == 2
    scores = corpus.rrf_fuse([["a", "b"], ["b", "c"]], k=1)
    assert scores["b"] > scores["a"] > scores["c"]
