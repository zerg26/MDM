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

    # patch run_search_agents to return deterministic candidates
    async def fake_run_search_agents(row, agents=None):
        return [
            {"field": "company", "value": "Acme Corporation", "source": "serpapi", "confidence": 0.9},
            {"field": "website", "value": "https://acme.example", "source": "tavily", "confidence": 0.8},
        ]

    monkeypatch.setattr("src.mdm.search_agents.run_search_agents", fake_run_search_agents)

    # run CLI pipeline
    from src.cli import run_pipeline

    run_pipeline(str(inp), str(out), chunk_size=1)

    assert out.exists()
    odf = pd.read_csv(out)
    assert odf.loc[0, "company"] == "Acme Corporation"
    assert odf.loc[0, "website"] == "https://acme.example"
