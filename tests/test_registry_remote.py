import json
from unittest.mock import patch


def test_registry_remote_parsing(monkeypatch):
    """Mock a remote registry response and ensure check_registry parses it."""
    from src.mdm.registries import check_registry

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            return None
        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        # emulate nested {"data": {...}} response
        return FakeResp({"data": {"match": True, "confidence": 0.9}})

    monkeypatch.setattr('httpx.post', fake_post)

    # set env var to trigger remote call
    import os
    os.environ['REGISTRY_URL'] = 'https://registry.example/lookup'

    r = check_registry('company', 'Acme Corp')
    assert r['match'] is True
    assert r['confidence'] >= 0.8

    # cleanup
    del os.environ['REGISTRY_URL']