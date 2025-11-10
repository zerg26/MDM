def test_registry_boost_effect():
    from src.mdm.verifier import verify_candidates

    # candidate without registry boost
    candidates = [
        {"field": "company", "value": "Acme Furnture", "source": "serpapi", "confidence": 0.6},
        {"field": "company", "value": "Acme Furnture", "source": "openai", "confidence": 0.3},
        {"field": "company", "value": "Other Co", "source": "serpapi", "confidence": 0.5},
    ]

    # Even with a small typo, registry should fuzzy-match 'Acme Furniture' strongly
    best = verify_candidates(candidates)
    # Expect the fuzzy-matched value to win
    assert "Acme" in best.get("company", "")


def test_registry_no_match():
    from src.mdm.verifier import verify_candidates

    candidates = [
        {"field": "company", "value": "Unknown Co", "source": "openai", "confidence": 0.4},
        {"field": "company", "value": "Another Inc", "source": "serpapi", "confidence": 0.5},
    ]
    best = verify_candidates(candidates)
    # some best guess should still be returned
    assert best.get("company") in {"Unknown Co", "Another Inc"}
