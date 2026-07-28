import os
import tempfile
import pandas as pd
import asyncio

from src.mdm.utils import read_csv


def test_pipeline_monkeypatched(monkeypatch, tmp_path):
    # prepare input CSV
    inp = tmp_path / "input.csv"
    df = pd.DataFrame([{"id": 1, "name": "Acme Corp", "company": "", "website": ""}])
    df.to_csv(inp, index=False)

    out = tmp_path / "output.csv"

    # patch run_search_agents (as bound in src.cli) to return deterministic candidates
    async def fake_run_search_agents(row, agents=None, use_multi_query=True):
        return [
            {"field": "company", "value": "Acme Corporation", "source": "serpapi", "agent": "serpapi", "confidence": 0.9},
            {"field": "website", "value": "https://acme.example", "source": "tavily", "agent": "tavily", "confidence": 0.8},
        ]

    monkeypatch.setattr("src.cli.run_search_agents", fake_run_search_agents)

    # run CLI pipeline
    from src.cli import run_pipeline

    run_pipeline(str(inp), str(out), chunk_size=1)

    assert out.exists()
    odf = pd.read_csv(out)
    assert odf.loc[0, "company"] == "Acme Corporation"
    assert odf.loc[0, "website"] == "https://acme.example"


def test_pipeline_graph_engine(monkeypatch, tmp_path):
    inp = tmp_path / "input.csv"
    pd.DataFrame([{"id": 1, "name": "Acme Corp", "company": "", "website": ""}]).to_csv(inp, index=False)
    out = tmp_path / "output.csv"

    # Patch the search layer the graph uses so the run is deterministic/offline.
    import src.mdm.graph as graph

    async def fake(row, agents=None, use_multi_query=True):
        return [{"field": "company", "value": "Acme Corporation", "source": "serpapi",
                 "agent": "serpapi", "confidence": 0.9}]

    monkeypatch.setattr(graph, "run_search_agents", fake)

    from src.cli import run_pipeline
    run_pipeline(str(inp), str(out), chunk_size=1, engine="graph")

    odf = pd.read_csv(out)
    assert odf.loc[0, "company"] == "Acme Corporation"
    # graph engine emits explainability columns
    assert "mdm_decision_trace" in odf.columns
    assert "mdm_failure_category" in odf.columns
    assert "router_agent" in odf.loc[0, "mdm_decision_trace"]
