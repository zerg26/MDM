from typing import List, Dict, Any
import math
import difflib

# --- ARCHITECTURAL CONSTANTS ---
AMBIGUOUS_NAME_MAX_LENGTH = 4
FUZZY_MATCH_CONFIDENCE_CUTOFF = 0.8
SHORT_BUSINESS_SUFFIXES = ["inc", "corp", "llc", "ltd"]
EXTENDED_BUSINESS_SUFFIXES = ["inc", "corp", "llc", "ltd", "co", "company", "corporation"]

def decide_agents_for_row(customer_record: Dict[str, Any], routing_config: Dict[str, Any] | None = None) -> List[str]:
    """
    AI Agent Router:
    Evaluates incomplete customer records to dynamically route them to the correct enrichment agents.
    Why dynamic routing: Static routing for every record is too expensive and slow. By inspecting missing fields 
    and name ambiguity, we invoke heavy agents (like OpenAI) only when simple search 
    heuristics (Google/Registry) are likely to fail; reducing API costs and processing time.
    """
    required_agents = set()
    def is_missing(v: Any) -> bool:
        if v is None:
            return True
        # pandas may give NaN as float
        if isinstance(v, float) and math.isnan(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    name = str(customer_record.get("name") or "").strip()
    company = None if is_missing(customer_record.get("company")) else str(customer_record.get("company")).strip()
    website = None if is_missing(customer_record.get("website")) else str(customer_record.get("website")).strip()

    required_agents.add("google")

    # If a config mapping exists (field -> agents), use it to force agents for missing fields
    cfg = routing_config or {}

    # Apply per-row overrides if present. Overrides is expected as a list of dicts:
    # {"match": {"id": <id>} or {"name_contains": "Acme"}, "agents": [...], "force": bool}
    overrides = cfg.get("overrides") or []
    for ov in overrides:
        match = ov.get("match", {})
        try:
            if "id" in match and str(match.get("id")) == str(customer_record.get("id")):
                if ov.get("force"):
                    return sorted(set(ov.get("agents", [])))
                else:
                    required_agents.update(ov.get("agents", []))
            if "name_contains" in match and match.get("name_contains", "").lower() in name.lower():
                if ov.get("force"):
                    return sorted(set(ov.get("agents", [])))
                else:
                    required_agents.update(ov.get("agents", []))
        except Exception:
            # ignore malformed override entries
            pass

    # website missing -> use serpapi + registry
    if not website:
        # allow config override
        if cfg.get("website"):
            required_agents.update(cfg.get("website", []))
        else:
            required_agents.add("serpapi")
            required_agents.add("registry")

    # company missing -> try multiple sources
    if not company:
        if cfg.get("company"):
            required_agents.update(cfg.get("company", []))
        else:
            required_agents.add("serpapi")
            required_agents.add("tavily")
            required_agents.add("openai")

    # name ambiguous heuristic: short or generic suffix
    lower_name = name.lower()
    if len(name) <= AMBIGUOUS_NAME_MAX_LENGTH or any(tok in lower_name for tok in SHORT_BUSINESS_SUFFIXES):
        required_agents.add("openai")

    # fuzzy match tokens: if name contains a token close to business suffixes, mark ambiguous
    tokens = [t for t in lower_name.replace(".", " ").split() if t]
    for tok in tokens:
        close = difflib.get_close_matches(tok, EXTENDED_BUSINESS_SUFFIXES, n=1, cutoff=FUZZY_MATCH_CONFIDENCE_CUTOFF)
        if close:
            required_agents.add("openai")
            break

    # If config supplies a default set for 'default', include them
    if cfg.get("default"):
        required_agents.update(cfg.get("default", []))

    # default fallback to serpapi if nothing selected
    if not required_agents:
        required_agents.add("serpapi")

    return sorted(required_agents)


def plan_tasks(records: List[Dict[str, Any]], chunk_size: int = 1, routing_config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """
    Batches records into tasks. 
    Why: Processing one record at a time via external APIs creates high latency. Grouping them 
    into batches (chunks) allows us to execute agents in parallel across multiple records.
    """
    if chunk_size <= 0:
        chunk_size = 1

    tasks: List[Dict[str, Any]] = []
    for i in range(0, len(records), chunk_size):
        record_batch = records[i : i + chunk_size]
        # union required_agents
        agents_set = set()
        for rec in record_batch:
            agents_set.update(decide_agents_for_row(rec, routing_config))
        tasks.append({"records": record_batch, "agents": sorted(agents_set)})

    return tasks
