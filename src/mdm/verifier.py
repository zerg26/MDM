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
    "WRONG_CITY": "Correct company name but wrong city",
    "OFFICE_VACATED": "Address no longer in use by company",
    "SPELLING_ERROR": "Input contains spelling or number mistake",
    "BRANCH_MISMATCH": "Company exists but not at specified branch location",
    "PO_BOX": "Input is a PO Box, company has no physical office there",
    "MERGED_ACQUIRED": "Company merged/acquired; address ownership changed",
    "SUBSIDIARY": "Address belongs to subsidiary, not parent company",
    "NOT_FOUND": "No evidence company exists at address",
    "AMBIGUOUS": "Multiple companies match; cannot confirm specific one",
    "MISSING_ADDRESS": "Critical address information missing from input",
    "INCOMPLETE_DATA": "Insufficient input data for verification (missing company or address)",
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
    - Determine presence_confirmed based on whether candidates were found for company/name fields.
    - Classify non-matches based on available data and input patterns.
    - Extract company status info if available.
    - If top value for a field has summed weight >= threshold, accept it.
    - Otherwise return the top-scoring value as best guess or None.
    """
    # We'll group by canonical value (for voting) but remember original display values
    by_field = defaultdict(lambda: defaultdict(float))
    display_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    source_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    agent_map: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    status_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    parent_company_map: Dict[str, float] = {}
    parent_address_map: Dict[str, float] = {}

    # Normalize candidate values before aggregation to avoid duplicates, but keep originals
    for c in candidates:
        field = c.get("field")
        raw_value = c.get("value")
        conf = float(c.get("confidence", 0.5) or 0.5)
        source = c.get("source", "unknown")
        agent = c.get("agent", source)
        company_status = c.get("company_status")
        parent_company = c.get("parent_company")
        parent_address = c.get("parent_address")

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
        if company_status and company_status not in status_map[field][canon]:
            status_map[field][canon].append(company_status)

        # Track parent info by confidence (take highest)
        if parent_company:
            if parent_company not in parent_company_map or parent_company_map[parent_company] < conf:
                parent_company_map[parent_company] = conf
        if parent_address:
            if parent_address not in parent_address_map or parent_address_map[parent_address] < conf:
                parent_address_map[parent_address] = conf

    result: Dict[str, Any] = {}
    best_sources = {}
    company_statuses = []

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
        
        # Collect company statuses if available
        statuses = status_map.get(field, {}).get(top_canon, [])
        if statuses:
            company_statuses.extend(statuses)

    # Determine presence confirmation - look for company or name field with value
    presence_confirmed = False
    presence_source = ""
    
    if result.get("company"):
        presence_confirmed = True
        presence_source = "; ".join(best_sources.get("company", []))
    elif result.get("name"):
        presence_confirmed = True
        presence_source = "; ".join(best_sources.get("name", []))

    # Classify non-match if no company found
    non_match_reason = None
    if not presence_confirmed:
        non_match_reason = classify_non_match(row or {}, candidates)

    result["presence_confirmed"] = presence_confirmed
    result["presence_source"] = presence_source
    result["non_match_reason"] = non_match_reason

    # Choose best parent info by highest confidence
    if parent_company_map:
        result["parent_company"] = max(parent_company_map.items(), key=lambda kv: kv[1])[0]
    else:
        result["parent_company"] = None

    if parent_address_map:
        result["parent_address"] = max(parent_address_map.items(), key=lambda kv: kv[1])[0]
    else:
        result["parent_address"] = None

    result["company_status"] = company_statuses[0] if company_statuses else None

    return result
def classify_non_match(row: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    """Classify why a company was not confirmed at the given address.

    Uses heuristics based on what was found and input patterns.
    Works with query CSV schema (SOURCE_NAME, SOURCE_ADDRESS, etc).
    """
    if not row:
        return "NOT_FOUND"

    # Extract fields from query CSV schema (with fallback for backward compatibility)
    company = row.get("SOURCE_NAME") or row.get("SRC_CLEANSED_SOURCE_NAME") or row.get("company_name", "")
    address = row.get("SOURCE_ADDRESS") or row.get("SRC_CLEANSED_STREET_ADDRESS") or row.get("address", "")
    city = row.get("SOURCE_CITY") or row.get("SRC_CLEANSED_CITY") or row.get("city", "")
    state = row.get("SOURCE_STATE") or row.get("SRC_CLEANSED_STATE") or row.get("state", "")
    postal = row.get("SOURCE_POSTAL_CODE") or row.get("postal_code", "")
    country = row.get("SOURCE_COUNTRY") or row.get("country", "")

    company = str(company).strip().lower() if company else ""
    address = str(address).strip().lower() if address else ""
    city = str(city).strip().lower() if city else ""
    state = str(state).strip().lower() if state else ""
    postal = str(postal).strip().lower() if postal else ""
    country = str(country).strip().lower() if country else ""

    # Check for missing critical input fields
    missing_fields = []
    if not company:
        missing_fields.append("company_name")
    if not address:
        missing_fields.append("address")
    
    if missing_fields:
        if len(missing_fields) == 2:
            return "INCOMPLETE_DATA"
        # If only address is missing, it's a data quality issue
        if "address" in missing_fields:
            return "MISSING_ADDRESS"

    # If no candidates at all
    if not candidates:
        return "NOT_FOUND"

    # Check for spelling indicators in address - very short or suspicious patterns
    if address and len(address) < 3:
        return "SPELLING_ERROR"

    if "p.o. box" in address or "po box" in address or "pob" in address or "post office" in address:
        return "PO_BOX"

    # Check if we found the company but no address match
    company_found = any(
        (company and company in str(c.get("value", "")).lower())
        or (company and company.split()[0] in str(c.get("value", "")).lower())  # Check first word
        for c in candidates
        if c.get("field") in ("company", "name")
    )

    if company_found:
        # We found the company, but address didn't match - narrow down reason
        if state and state not in [str(c.get("value", "")).lower() for c in candidates if c.get("field") == "state"]:
            return "WRONG_STATE"
        elif postal and postal not in [str(c.get("value", "")).lower() for c in candidates if c.get("field") == "postal_code"]:
            return "WRONG_POSTAL"
        elif city and city not in [str(c.get("value", "")).lower() for c in candidates if c.get("field") == "city"]:
            return "WRONG_CITY"
        else:
            return "WRONG_STREET"

    # Check for acquisition/merger hints
    acquired_keywords = ["acquired", "merged", "subsidiary", "parent company", "owned by", "acquired by"]
    candidate_text = " ".join(str(c.get("value", "")) for c in candidates).lower()
    if any(kw in candidate_text for kw in acquired_keywords):
        return "MERGED_ACQUIRED"

    # Check for office vacated hints
    vacated_keywords = ["defunct", "closed", "ceased operations", "no longer in operation", "out of business", "bankruptcy"]
    if any(kw in candidate_text for kw in vacated_keywords):
        return "OFFICE_VACATED"

    # Check for spelling mistake indicators (unusual character patterns, very short fields)
    if company and len(company.split()) == 1 and len(company) > 20:
        # Long single word company name - might be malformed
        return "SPELLING_ERROR"
    
    # If we have some candidates but no clear company match
    if candidates:
        return "AMBIGUOUS"

    return "NOT_FOUND"