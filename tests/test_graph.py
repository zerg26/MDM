"""Tests for the LangGraph self-healing orchestration (src/mdm/graph.py).

The search layer is mocked so the graph's control flow (confirm vs. heal-and-retry)
is exercised deterministically without any network access.
"""
import src.mdm.graph as graph


def _mock_search(monkeypatch, side_effect):
    """Patch the graph's search call. ``side_effect`` is a callable(row, agents,
    use_multi_query) or a list of return values consumed one per invocation."""
    calls = {"n": 0}

    async def fake(row, agents=None, use_multi_query=True):
        i = calls["n"]
        calls["n"] += 1
        if callable(side_effect):
            return side_effect(i, row, agents, use_multi_query)
        # list of per-call return values; last value repeats
        return side_effect[min(i, len(side_effect) - 1)]

    monkeypatch.setattr(graph, "run_search_agents", fake)
    return calls


def test_confirmed_first_pass_no_heal(monkeypatch):
    monkeypatch.setenv("MDM_MAX_HEAL_ATTEMPTS", "2")
    calls = _mock_search(monkeypatch, [[
        {"field": "company", "value": "Acme Corporation", "source": "serpapi",
         "agent": "serpapi", "confidence": 0.9},
    ]])

    out = graph.run_row_sync({"name": "Acme Corp", "company": "", "website": ""})

    assert out["confirmed"] is True
    assert out["attempts"] == 0
    assert calls["n"] == 1  # searched exactly once
    nodes = [t["node"] for t in out["trace"]]
    assert nodes == ["router_agent", "search_agent", "verifier_agent"]
    assert "healer_agent" not in nodes
    assert out["verified"]["company"] == "Acme Corporation"


def test_unconfirmed_heals_then_stops(monkeypatch):
    monkeypatch.setenv("MDM_MAX_HEAL_ATTEMPTS", "2")
    # Always return nothing -> never confirmed -> heal until attempts exhausted.
    calls = _mock_search(monkeypatch, [[]])

    out = graph.run_row_sync({"name": "Ghost Co", "company": "", "website": ""})

    assert out["confirmed"] is False
    assert out["attempts"] == 2                      # bounded by max heal attempts
    assert calls["n"] == 3                           # initial + 2 retries
    nodes = [t["node"] for t in out["trace"]]
    assert nodes.count("healer_agent") == 2
    assert nodes.count("search_agent") == 3
    assert out["category"]                           # a non-match category is set


def test_heals_then_confirms(monkeypatch):
    monkeypatch.setenv("MDM_MAX_HEAL_ATTEMPTS", "3")
    # First pass empty, second pass finds the company.
    _mock_search(monkeypatch, [
        [],
        [{"field": "company", "value": "Globex", "source": "google_kg",
          "agent": "google", "confidence": 0.85}],
    ])

    out = graph.run_row_sync({"name": "Globex", "company": "", "website": ""})

    assert out["confirmed"] is True
    assert out["attempts"] == 1                       # healed exactly once
    nodes = [t["node"] for t in out["trace"]]
    assert nodes.count("healer_agent") == 1
    assert out["verified"]["company"] == "Globex"


def test_router_widens_agents_on_retry(monkeypatch):
    monkeypatch.setenv("MDM_MAX_HEAL_ATTEMPTS", "1")
    seen_agents = []

    async def fake(row, agents=None, use_multi_query=True):
        seen_agents.append(list(agents or []))
        return []

    monkeypatch.setattr(graph, "run_search_agents", fake)
    graph.run_row_sync({"name": "X", "company": "", "website": ""})

    # Second (heal) pass must include the full widened agent set.
    assert len(seen_agents) == 2
    assert set(graph.ALL_AGENTS).issubset(set(seen_agents[1]))
