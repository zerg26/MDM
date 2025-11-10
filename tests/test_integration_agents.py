import asyncio

from src.mdm.search_agents import run_search_agents


def test_run_search_agents_handles_agent_failure(monkeypatch):
    # Prepare a row needing company and website
    row = {"id": 1, "name": "Acme Corp", "company": "", "website": ""}

    async def fake_serpapi(q, field):
        raise RuntimeError("serpapi down")

    async def fake_tavily(q, field):
        return [{"field": field, "value": f"fake-{field}", "source": "tavily", "confidence": 0.7}]

    monkeypatch.setattr("src.mdm.search_agents.search_serpapi", fake_serpapi)
    monkeypatch.setattr("src.mdm.search_agents.search_tavily", fake_tavily)

    # run the coroutine using asyncio.run
    candidates = asyncio.run(run_search_agents(row, agents=["serpapi", "tavily"]))
    # Even though serpapi failed, tavily results should still be returned
    values = [c["value"] for c in candidates]
    assert any(v.startswith("fake-") for v in values)
