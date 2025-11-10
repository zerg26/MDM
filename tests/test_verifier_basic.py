from src.mdm.verifier import verify_candidates


def test_verify_basic_votes():
    candidates = [
        {"field": "company", "value": "Acme Corp", "source": "openai", "confidence": 0.4},
        {"field": "company", "value": "Acme Corp", "source": "serpapi", "confidence": 0.6},
        {"field": "website", "value": "https://acme.com", "source": "serpapi", "confidence": 0.9},
    ]
    best = verify_candidates(candidates)
    assert best["company"] == "Acme Corp"
    assert best["website"] == "https://acme.com"
