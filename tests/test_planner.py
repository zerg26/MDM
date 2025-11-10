import math
from src.mdm.planner import decide_agents_for_row, plan_tasks


def test_decide_agents_missing_website():
    row = {"id": 1, "name": "Acme Corp", "company": "Acme Corporation", "website": ""}
    agents = decide_agents_for_row(row)
    assert "serpapi" in agents
    assert "registry" in agents


def test_decide_agents_missing_company():
    row = {"id": 2, "name": "Globex", "company": "", "website": "https://globex.example"}
    agents = decide_agents_for_row(row)
    assert "serpapi" in agents
    assert "tavily" in agents
    assert "openai" in agents


def test_decide_agents_ambiguous_name():
    row = {"id": 3, "name": "X", "company": "", "website": ""}
    agents = decide_agents_for_row(row)
    assert "openai" in agents


def test_plan_tasks_with_config_override():
    rows = [
        {"id": 1, "name": "Acme", "company": "", "website": ""},
        {"id": 2, "name": "Beta", "company": "", "website": ""},
    ]
    config = {"company": ["tavily"], "website": ["serpapi", "registry"], "default": ["tavily"]}
    tasks = plan_tasks(rows, chunk_size=1, config=config)
    # each task should include tavily due to default and company override
    for t in tasks:
        assert "tavily" in t["agents"]
        assert "serpapi" in t["agents"] or "registry" in t["agents"]


def test_handle_pandas_nan_values():
    # NaN values (represented as float('nan')) should be treated as missing
    row = {"id": 4, "name": "Acme", "company": float("nan"), "website": float("nan")}
    agents = decide_agents_for_row(row)
    assert "serpapi" in agents
    assert "registry" in agents
