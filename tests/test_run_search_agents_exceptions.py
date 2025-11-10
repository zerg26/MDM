import asyncio
import pytest
from unittest.mock import AsyncMock


def test_run_search_agents_handles_agent_exceptions(monkeypatch):
    """If one agent raises, run_search_agents should still return other agents' candidates."""
    from src.mdm.search_agents import run_search_agents

    async def fake_serpapi_raise(query, field):
        raise RuntimeError("serpapi down")

    async def fake_openai_ok(query, field):
        return [{"field": field, "value": "OpenAI Result", "source": "openai", "confidence": 0.4}]

    monkeypatch.setattr("src.mdm.search_agents.search_serpapi", fake_serpapi_raise)
    monkeypatch.setattr("src.mdm.search_agents.search_openai", fake_openai_ok)

    row = {"id": 1, "name": "Acme Corp", "company": "", "website": ""}
    res = asyncio.run(run_search_agents(row, agents=["serpapi", "openai"]))
    # expect that openai candidate is present and no exception was propagated
    assert any(r["source"] == "openai" for r in res)
    assert all(r["source"] != "serpapi" for r in res)
