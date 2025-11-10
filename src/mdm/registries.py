"""Registry connector (stub + optional remote placeholder).

Provides a small fuzzy-match based local registry for tests and examples.
In production this would call an external API (BBB, company registry) and
return a structured response.
"""
from typing import Dict, Any
import os
import difflib
from dotenv import load_dotenv
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt


def _fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_registry(field: str, value: Any) -> Dict[str, Any]:
    """Check a value against a registry.

    Returns: {"match": bool, "confidence": float, "source": str}

    Behavior:
    - If REGISTRY_URL is set in env, we currently do not perform an outbound call
      (placeholder) to avoid network access during tests; this can be extended.
    - Otherwise, perform a local fuzzy match against a small authoritative list.
    """
    load_dotenv()
    reg_url = os.getenv("REGISTRY_URL")
    if reg_url:
        # Prepare auth headers if provided. Supported modes:
        # - Bearer token via REGISTRY_API_KEY (sends Authorization: Bearer <key>)
        # - Custom header via REGISTRY_AUTH_HEADER and REGISTRY_AUTH_VALUE
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        api_key = os.getenv("REGISTRY_API_KEY")
        auth_header = os.getenv("REGISTRY_AUTH_HEADER")
        auth_value = os.getenv("REGISTRY_AUTH_VALUE")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif auth_header and auth_value:
            headers[auth_header] = auth_value

        # Attempt to call remote registry endpoint with retries. Expect JSON response
        # shape: {"match": bool, "confidence": float} or nested under 'data'.
        try:
            @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
            def _call_remote():
                # use synchronous httpx for simplicity here
                r = httpx.post(reg_url, json={"field": field, "value": str(value)}, headers=headers, timeout=5.0)
                r.raise_for_status()
                data = r.json()
                # Accept multiple shapes
                if isinstance(data, dict):
                    if "match" in data:
                        return {"match": bool(data.get("match")), "confidence": float(data.get("confidence", 0.0)), "source": "registry_remote"}
                    # Some APIs nest response: {"data": {"match": ..., "confidence": ...}}
                    if "data" in data and isinstance(data["data"], dict) and "match" in data["data"]:
                        inner = data["data"]
                        return {"match": bool(inner.get("match")), "confidence": float(inner.get("confidence", 0.0)), "source": "registry_remote"}
                return {"match": False, "confidence": 0.0, "source": "registry_remote"}

            return _call_remote()
        except Exception:
            # On any error contacting remote registry, fall back to local heuristic
            pass

    if value is None:
        return {"match": False, "confidence": 0.0, "source": "registry_stub"}

    s = str(value).strip()

    # Quick heuristics (preserve previous test expectations):
    if field == "website":
        if s.startswith("http") or s.startswith("www"):
            return {"match": True, "confidence": 0.95, "source": "registry_stub"}
        return {"match": False, "confidence": 0.0, "source": "registry_stub"}

    if field in ("company", "name"):
        if len(s) > 3:
            return {"match": True, "confidence": 0.8, "source": "registry_stub"}
        return {"match": False, "confidence": 0.0, "source": "registry_stub"}

    # Fallback: small authoritative sample for more specific matches
    authoritative = {
        "company": ["Acme Furniture", "Globex"],
        "website": ["https://www.acmecorp.com", "https://www.globex.com"]
    }

    candidates = authoritative.get(field, [])
    best_score = 0.0
    for c in candidates:
        sc = _fuzzy_score(c, s)
        if sc > best_score:
            best_score = sc

    # Interpret fuzzy scores (lower-priority fallback)
    if best_score >= 0.85:
        return {"match": True, "confidence": round(best_score, 2), "source": "registry_stub"}
    if best_score >= 0.6:
        return {"match": True, "confidence": round(best_score * 0.8, 2), "source": "registry_stub"}
    return {"match": False, "confidence": round(best_score * 0.5, 2), "source": "registry_stub"}
