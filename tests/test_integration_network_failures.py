import asyncio


def test_run_search_agents_with_partial_network_failure(monkeypatch):
    """Simulate SerpAPI failing and OpenAI succeeding; pipeline should still return OpenAI results."""
    from src.mdm.search_agents import run_search_agents

    async def fake_serpapi_connect_error(query, field):
        # Simulate network error
        import httpx

        raise httpx.ConnectError("getaddrinfo failed")

    async def fake_tavily_ok(query, field):
        return [{"field": field, "value": "https://example.tavily", "source": "tavily", "confidence": 0.6}]

    async def fake_openai_ok(query, field):
        return [{"field": field, "value": "OpenAI Co", "source": "openai", "confidence": 0.4}]

    monkeypatch.setattr("src.mdm.search_agents.search_serpapi", fake_serpapi_connect_error)
    monkeypatch.setattr("src.mdm.search_agents.search_tavily", fake_tavily_ok)
    monkeypatch.setattr("src.mdm.search_agents.search_openai", fake_openai_ok)

    row = {"id": 5, "name": "TestCo", "company": "", "website": ""}
    res = asyncio.run(run_search_agents(row, agents=["serpapi", "tavily", "openai"]))
    # ensure tavily and openai candidates present, serpapi absent
    sources = {r["source"] for r in res}
    assert "tavily" in sources or any(s.startswith("tavily") for s in sources)
    assert "openai" in sources
    assert not any(s.startswith("serpapi") for s in sources)
