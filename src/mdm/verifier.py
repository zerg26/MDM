from typing import List, Dict, Any, Optional
from collections import defaultdict
from .registries import check_registry
from .utils import normalize_text, normalize_website
from .utils import normalize_company


def verify_candidates(candidates: List[Dict[str, Any]], threshold: float = 0.6) -> Dict[str, Any]:
    """Given a list of candidate dicts, return a mapping field->decided_value.

    Strategy:
    - Group by field, then by value, sum confidences.
    - If an external registry confirms a value, boost its score.
    - If top value for a field has summed weight >= threshold, accept it.
    - Otherwise return the top-scoring value as best guess.
    """
    # We'll group by canonical value (for voting) but remember original display values
    by_field = defaultdict(lambda: defaultdict(float))
    display_map: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    # Normalize candidate values before aggregation to avoid duplicates, but keep originals
    for c in candidates:
        field = c.get("field")
        raw_value = c.get("value")
        conf = float(c.get("confidence", 0.5) or 0.5)
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

    result: Dict[str, Any] = {}
    for field, val_map in by_field.items():
        if not val_map:
            continue
        # For each candidate canonical value, check registry to possibly boost confidence
        boosted_map = dict(val_map)
        for canon in list(val_map.keys()):
            try:
                # check using canonical value (or display?) — registry expects readable company/website; use canon
                reg = check_registry(field, canon)
                if reg.get("match"):
                    # boost by registry confidence, weighted by REGISTRY_BOOST env or default 0.5
                    try:
                        import os

                        boost_factor = float(os.getenv("REGISTRY_BOOST", "0.5"))
                    except Exception:
                        boost_factor = 0.5
                    boosted_map[canon] = boosted_map.get(canon, 0.0) + float(reg.get("confidence", 0.0)) * boost_factor
            except Exception:
                # On any registry error, skip boosting
                pass

        # find top value
        top_canon, top_score = max(boosted_map.items(), key=lambda kv: kv[1])
        total = sum(boosted_map.values())
        # normalized weight (not used for now, but left for future)
        weight = top_score / total if total > 0 else 0
        # Choose the best display value for the winning canonical form (highest summed confidence)
        display_candidates = display_map.get(field, {}).get(top_canon, {})
        if display_candidates:
            best_display = max(display_candidates.items(), key=lambda kv: kv[1])[0]
        else:
            best_display = top_canon
        result[field] = best_display
    return result
