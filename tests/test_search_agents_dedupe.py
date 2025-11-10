import asyncio
from src.mdm.search_agents import search_serpapi, search_openai


def test_serpapi_dedupe(monkeypatch):
    # monkeypatch httpx AsyncClient.get to return a fake response
    class FakeResp:
        def __init__(self, data):
            self._data = data
        def raise_for_status(self):
            return None
        def json(self):
            return self._data

    data = {
        "knowledge_graph": {"title": "Acme Furniture", "website": "https://www.acmecorp.com/"},
        "organic_results": [
            {"title": "Acme Furniture", "link": "https://www.acmecorp.com/"},
            {"title": "Acme Furniture - Store", "link": "https://www.acmecorp.com/store"},
        ],
    }

    async def fake_get(*args, **kwargs):
        return FakeResp(data)

    monkeypatch.setattr('httpx.AsyncClient.get', fake_get)
    res = asyncio.run(search_serpapi('Acme Corp', 'company'))
    # deduped: expect first company candidate and organic ones (unique)
    assert any(r['value'].lower().startswith('acme furniture') for r in res)


def test_openai_parsing(monkeypatch):
    class FakeResp:
        def __init__(self, data):
            self._data = data
        def raise_for_status(self):
            return None
        def json(self):
            return self._data

    data = {"choices": [{"message": {"content": "Acme Corp"}}]}

    async def fake_post(*args, **kwargs):
        return FakeResp(data)

    monkeypatch.setattr('httpx.AsyncClient.post', fake_post)
    res = asyncio.run(search_openai('Acme Corp', 'company'))
    assert any(r['value'] == 'Acme Corp' for r in res)
