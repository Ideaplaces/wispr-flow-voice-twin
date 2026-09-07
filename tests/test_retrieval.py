import retrieval


def test_build_filters_where_and_predicate_agree():
    where, accept = retrieval.build_filters(human=True, lang=["en", "engb"], kinds="status", min_words=10)
    assert where == {"$and": [{"human": True}, {"lang": {"$in": ["en", "engb"]}},
                              {"kind": {"$in": ["status"]}}, {"n_words": {"$gte": 10}}]}
    assert accept({"human": True, "lang": "en", "kind": "status", "n_words": 12})
    assert not accept({"human": False, "lang": "en", "kind": "status", "n_words": 12})
    assert not accept({"human": True, "lang": "fr", "kind": "status", "n_words": 12})
    assert not accept({"human": True, "lang": "en", "kind": "ask", "n_words": 12})
    assert not accept({"human": True, "lang": "en", "kind": "status", "n_words": 3})


def test_build_filters_single_clause_and_empty():
    where, accept = retrieval.build_filters(ctx="team_chat")
    assert where == {"ctx": "team_chat"}
    assert accept({"ctx": "team_chat"}) and not accept({"ctx": "ai_chat"})
    where, accept = retrieval.build_filters()
    assert where is None and accept({})


def test_lexical_index_finds_proper_nouns_and_respects_filters():
    idx = retrieval.LexicalIndex(
        ["a", "b", "c"],
        ["Kalitravel wants the pricing", "the pricing is fine", "Kalitravel again"],
        [{"human": True}, {"human": True}, {"human": False}],
    )
    hits = idx.query("Kalitravel", 5, lambda m: m["human"])
    assert [h[0] for h in hits] == ["a"]
    hits = idx.query("pricing", 5, lambda m: True)
    assert {h[0] for h in hits} == {"a", "b"}
