from src.mdm.planner import decide_agents_for_row, plan_tasks


def test_override_by_id_forces():
    row = {"id": 42, "name": "Acme Corp", "company": "", "website": ""}
    cfg = {"overrides": [{"match": {"id": 42}, "agents": ["openai", "serpapi"], "force": True}]}
    agents = decide_agents_for_row(row, config=cfg)
    assert set(agents) == {"openai", "serpapi"}


def test_override_by_name_merges():
    row = {"id": 2, "name": "Globex", "company": "", "website": ""}
    cfg = {"overrides": [{"match": {"name_contains": "glob"}, "agents": ["registry"], "force": False}]}
    agents = decide_agents_for_row(row, config=cfg)
    assert "registry" in agents
    # still contains default serpapi for website/company
    assert "serpapi" in agents


def test_plan_tasks_chunking_and_union():
    rows = [
        {"id": 1, "name": "A", "company": "", "website": ""},
        {"id": 2, "name": "B Ltd", "company": "", "website": ""},
        {"id": 3, "name": "C", "company": "X Corp", "website": ""},
    ]
    tasks = plan_tasks(rows, chunk_size=2)
    assert len(tasks) == 2
    # first task should include agents union from row 1 and 2
    assert isinstance(tasks[0]["agents"], list)
