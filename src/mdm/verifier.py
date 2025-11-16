from typing import List, Dict, Any, Optional
from collections import defaultdict
from .registries import check_registry
from .utils import normalize_text, normalize_website
from .utils import normalize_company


# Non-match classification taxonomy
NON_MATCH_CLASSIFICATIONS = {
    "WRONG_STREET": "Correct company name but wrong street address",
    "WRONG_STATE": "Correct company name but wrong state/jurisdiction",
    "WRONG_POSTAL": "Correct company name but wrong postal code",
    "OFFICE_VACATED": "Address no longer in use by company",
    "SPELLING_ERROR": "Input contains spelling or number mistake",
    "BRANCH_MISMATCH": "Company exists but not at specified branch location",
    "PO_BOX": "Input is a PO Box, company has no physical office there",
    "MERGED_ACQUIRED": "Company merged/acquired; address ownership changed",
    "SUBSIDIARY": "Address belongs to subsidiary, not parent company",
    "NOT_FOUND": "No evidence company exists at address",
    "AMBIGUOUS": "Multiple companies match; cannot confirm specific one",
}


def verify_candidates(
    candidates: List[Dict[str, Any]], 
    row: Dict[str, Any] | None = None,
    threshold: float = 0.6
) -> Dict[str, Any]:
    """Given a list of candidate dicts, return a mapping with presence confirmation and classification.

    Strategy:
    - Group by field, then by value, sum confidences across queries/agents.
    - If an external registry confirms a value, boost its score.
    - Determine presence_confirmed based on whether candidates were found.
    - Classify non-matches based on available data.
    - Extract parent company info if available.
    - If top value for a field has summed weight >= threshold, accept it.
    - Otherwise return the top-scoring value as best guess or None.
    """
    # We'll group by canonical value (for voting) but remember original display values
    by_field = defaultdict(lambda: defaultdict(float))
    display_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    source_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    agent_map: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))

    # Normalize candidate values before aggregation to avoid duplicates, but keep originals
    for c in candidates:
        field = c.get("field")
        raw_value = c.get("value")
        conf = float(c.get("confidence", 0.5) or 0.5)
        source = c.get("source", "unknown")
        agent = c.get("agent", source)
        
        if not field or raw_value is None:
            continue
        
        if field == "website":
            canon = normalize_website(raw_value)
            display = normalize_website(raw_value)
        elif field in ("company", "name"):
            canon = normalize_company(raw_value)
            display = str(raw_value).strip()
        else:
            canon = normalize_text(raw_value)
            display = str(raw_value).strip()
        
        if not canon:
            continue
        
        by_field[field][canon] += conf
        display_map[field][canon][display] += conf
        if source not in source_map[field][canon]:
            source_map[field][canon].append(source)
        agent_map[field][canon].add(agent)

    result: Dict[str, Any] = {}
    best_sources = {}
    
    for field, val_map in by_field.items():
        if not val_map:
            continue
        
        # For each candidate canonical value, check registry to possibly boost confidence
        boosted_map = dict(val_map)
        for canon in list(val_map.keys()):
            try:
                reg = check_registry(field, canon)
                if reg.get("match"):
                    import os
                    boost_factor = float(os.getenv("REGISTRY_BOOST", "0.5"))
                    boosted_map[canon] = boosted_map.get(canon, 0.0) + float(reg.get("confidence", 0.0)) * boost_factor
            except Exception:
                pass

        # find top value
        top_canon, top_score = max(boosted_map.items(), key=lambda kv: kv[1])
        total = sum(boosted_map.values())
        weight = top_score / total if total > 0 else 0
        
        # Choose the best display value for the winning canonical form
        display_candidates = display_map.get(field, {}).get(top_canon, {})
        if display_candidates:
            best_display = max(display_candidates.items(), key=lambda kv: kv[1])[0]
        else:
            best_display = top_canon
        
        result[field] = best_display
        best_sources[field] = source_map.get(field, {}).get(top_canon, [])
    
    # Determine presence confirmation
    presence_confirmed = len(result) > 0 and result.get("company") is not None
    presence_source = "; ".join(best_sources.get("company", []))
    
    # Classify non-match if no company found
    non_match_reason = None
    if not presence_confirmed:
        non_match_reason = classify_non_match(row or {}, candidates)
    
    result["presence_confirmed"] = presence_confirmed
    result["presence_source"] = presence_source
    result["non_match_reason"] = non_match_reason
    result["parent_company"] = None  # TODO: extract from candidates if available
    result["parent_address"] = None  # TODO: extract from candidates if available
    
    return result


def classify_non_match(row: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    """Classify why a company was not confirmed at the given address.
    
    Uses heuristics based on what was found and input patterns.
    """
    if not row:
        return "NOT_FOUND"
    
    company = row.get("company_name", "").lower().strip()
    address = row.get("address", "").lower().strip()
    state = row.get("state", "").lower().strip()
    postal = row.get("postal_code", "").lower().strip()
    
    # If no candidates at all
    if not candidates:
        return "NOT_FOUND"
    
    # Check for spelling indicators in address
    if not address or len(address) < 5:
        return "SPELLING_ERROR"
    
    if "p.o. box" in address or "po box" in address or "pob" in address:
        return "PO_BOX"
    
    # Check if we found the company but no address match
    company_found = any(
        company in str(c.get("value", "")).lower() 
        for c in candidates 
        if c.get("field") == "company"
    )
    
    if company_found:
        # We found the company, but address didn't match
        if state:
            return "WRONG_STATE"
        elif postal:
            return "WRONG_POSTAL"
        else:
            return "WRONG_STREET"
    
    # Check for acquisition/merger hints
    acquired_keywords = ["acquired", "merged", "subsidiary", "parent"]
    candidate_text = " ".join(str(c.get("value", "")) for c in candidates).lower()
    if any(kw in candidate_text for kw in acquired_keywords):
        return "MERGED_ACQUIRED"
    
    return "AMBIGUOUS"
