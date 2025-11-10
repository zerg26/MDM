import asyncio
import json
import types

import httpx

from src.mdm.search_agents import search_serpapi


def test_search_serpapi_retries(monkeypatch):
    """Simulate a transient failure on the first httpx.get call, then success."""
    # Ensure API key is set so the function doesn't early-return
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    # Also update the module-level constant used by the function (import-time)
    import src.mdm.search_agents as sa
    sa.SERPAPI_API_KEY = "test-key"
    calls = {"count": 0}

    async def fake_get(self, url, params=None):
        # first call raises a network-like error, second returns a mock response
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.HTTPError("simulated transient error")

        class DummyResponse:
            def raise_for_status(self):
                return None

            def json(self):
                # return a structure similar to SerpAPI knowledge_graph
                return {"knowledge_graph": {"title": "Acme Corporation", "website": "https://acme.example"}, "organic_results": []}

        return DummyResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    # call the decorated function; because it has tenacity retry, it should succeed
    loop = asyncio.new_event_loop()
    try:
        res = loop.run_until_complete(search_serpapi("Acme Corp", "company"))
    finally:
        loop.close()

    # should have at least one candidate
    assert isinstance(res, list)
    assert any(c.get("value") for c in res)
