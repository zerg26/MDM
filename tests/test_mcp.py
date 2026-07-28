"""Tests for the MCP layer (in-process path).

The networked streamable-http path is exercised manually; here we verify the
in-process client dispatches to the same wrapped functions and that the graph
delegates its search step through the client when MDM_USE_MCP=1.
"""
import asyncio

import pytest

import src.mdm.mcp_client as mcp_client
import src.mdm.mcp_server as mcp_server
import src.mdm.graph as graph


def test_inprocess_search_record_matches_direct(monkeypatch):
    monkeypatch.delenv("MDM_MCP_ENDPOINT", raising=False)

    async def fake_run_search_agents(row, agents=None, use_multi_query=True):
        return [{"field": "company", "value": "Acme Corporation", "source": "serpapi",
                 "agent": "serpapi", "confidence": 0.9}]

    # Patch the underlying function used by the server impl.
    monkeypatch.setattr(mcp_server, "run_search_agents", fake_run_search_agents)

    out = asyncio.run(mcp_client.search_record({"name": "Acme"}, agents=["serpapi"]))
    assert out == [{"field": "company", "value": "Acme Corporation", "source": "serpapi",
                    "agent": "serpapi", "confidence": 0.9}]


def test_inprocess_route_and_verify(monkeypatch):
    monkeypatch.delenv("MDM_MCP_ENDPOINT", raising=False)

    agents = asyncio.run(mcp_client.route_record({"name": "Globex", "company": "", "website": ""}))
    assert isinstance(agents, list) and agents  # some agents chosen

    verified = asyncio.run(mcp_client.verify_record(
        [{"field": "company", "value": "Globex", "source": "google_kg", "agent": "google", "confidence": 0.85}],
        row={"name": "Globex"},
    ))
    assert verified["company"] == "Globex"
    assert verified["presence_confirmed"] is True


def test_graph_routes_search_through_mcp(monkeypatch):
    """With MDM_USE_MCP=1, the graph's search step must go via mcp_client."""
    monkeypatch.setenv("MDM_USE_MCP", "1")
    monkeypatch.delenv("MDM_MCP_ENDPOINT", raising=False)
    monkeypatch.setenv("MDM_MAX_HEAL_ATTEMPTS", "0")

    used = {"mcp": False}

    async def fake_search_record(row, agents=None, use_multi_query=True):
        used["mcp"] = True
        return [{"field": "company", "value": "Acme Corporation", "source": "serpapi",
                 "agent": "serpapi", "confidence": 0.9}]

    monkeypatch.setattr(mcp_client, "search_record", fake_search_record)

    out = graph.run_row_sync({"name": "Acme Corp", "company": "", "website": ""})
    assert used["mcp"] is True
    assert out["confirmed"] is True
    assert out["verified"]["company"] == "Acme Corporation"
