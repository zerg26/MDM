from typing import List, Dict, Any
import math
import difflib


def decide_agents_for_row(row: Dict[str, Any], config: Dict[str, Any] | None = None) -> List[str]:
    """Decide which agents to use for a single row using simple heuristics.

    - If `website` is missing: prefer SerpAPI + registry
    - If `company` is missing: query SerpAPI, Tavily and optionally OpenAI
    - If `name` looks ambiguous (short or contains common tokens), include OpenAI
    """
    agents = set()
    def is_missing(v: Any) -> bool:
        if v is None:
            return True
        # pandas may give NaN as float
        if isinstance(v, float) and math.isnan(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    name = str(row.get("name") or "").strip()
    company = None if is_missing(row.get("company")) else str(row.get("company")).strip()
    website = None if is_missing(row.get("website")) else str(row.get("website")).strip()

    # If a config mapping exists (field -> agents), use it to force agents for missing fields
    cfg = config or {}

    # Apply per-row overrides if present. Overrides is expected as a list of dicts:
    # {"match": {"id": <id>} or {"name_contains": "Acme"}, "agents": [...], "force": bool}
    overrides = cfg.get("overrides") or []
    for ov in overrides:
        match = ov.get("match", {})
        try:
            if "id" in match and str(match.get("id")) == str(row.get("id")):
                if ov.get("force"):
                    return sorted(set(ov.get("agents", [])))
                else:
                    agents.update(ov.get("agents", []))
            if "name_contains" in match and match.get("name_contains", "").lower() in name.lower():
                if ov.get("force"):
                    return sorted(set(ov.get("agents", [])))
                else:
                    agents.update(ov.get("agents", []))
        except Exception:
            # ignore malformed override entries
            pass

    # website missing -> use serpapi + registry
    if not website:
        # allow config override
        if cfg.get("website"):
            agents.update(cfg.get("website", []))
        else:
            agents.add("serpapi")
            agents.add("registry")

    # company missing -> try multiple sources
    if not company:
        if cfg.get("company"):
            agents.update(cfg.get("company", []))
        else:
            agents.add("serpapi")
            agents.add("tavily")
            agents.add("openai")

    # name ambiguous heuristic: short or generic suffix
    lower = name.lower()
    if len(name) <= 4 or any(tok in lower for tok in ["inc", "corp", "llc", "ltd"]):
        agents.add("openai")

    # fuzzy match tokens: if name contains a token close to business suffixes, mark ambiguous
    tokens = [t for t in lower.replace(".", " ").split() if t]
    suffixes = ["inc", "corp", "llc", "ltd", "co", "company", "corporation"]
    for tok in tokens:
        close = difflib.get_close_matches(tok, suffixes, n=1, cutoff=0.8)
        if close:
            agents.add("openai")
            break

    # If config supplies a default set for 'default', include them
    if cfg.get("default"):
        agents.update(cfg.get("default", []))

    # default fallback to serpapi if nothing selected
    if not agents:
        agents.add("serpapi")

    return sorted(agents)


def plan_tasks(rows: List[Dict[str, Any]], chunk_size: int = 1, config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """Create tasks: each task is a dict with 'rows' and 'agents' keys.

    We compute agents per-row then group rows into chunks of `chunk_size` and
    create a task where agents is the union of agents required by rows in the chunk.
    """
    if chunk_size <= 0:
        chunk_size = 1

    tasks: List[Dict[str, Any]] = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        # union agents
        agents_set = set()
        for r in chunk:
            agents_set.update(decide_agents_for_row(r, config=config))
        tasks.append({"rows": chunk, "agents": sorted(agents_set)})

    return tasks
