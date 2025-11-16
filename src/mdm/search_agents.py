"""Search agents: SerpAPI, Tavily (generic), and OpenAI optional hooks.

These functions are async and return candidate dicts in the format:
    {"field": <field_name>, "value": <value>, "source": <source>, "confidence": float}
"""
from typing import List, Dict, Any
import os
import asyncio
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv
from .utils import normalize_text, normalize_website


# NOTE: read env inside functions so newly-created/updated `.env` files or
# environment variables are picked up at runtime (avoids stale module-level values).


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
async def search_serpapi(query: str, field: str) -> List[Dict[str, Any]]:
    """Call SerpAPI (Google/Knowledge-graph) to fetch entity data.

    Returns a small list of candidate dicts.
    """
    # load env values at call time
    load_dotenv()
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    SERPAPI_URL = os.getenv("SERPAPI_URL", "https://serpapi.com/search.json")

    if not SERPAPI_API_KEY:
        # No key: return empty to let tests stub this out
        return []

    params = {"q": query, "api_key": SERPAPI_API_KEY}

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(SERPAPI_URL, params=params)
        r.raise_for_status()
        data = r.json()
        candidates = []
        seen = set()
        # Try knowledge_graph first (common for entities)
        kg = data.get("knowledge_graph") or {}
        # Map common fields
        if field == "website":
            val = kg.get("website") or kg.get("url") or kg.get("site")
            if not val:
                # fallback to organic_results
                for o in data.get("organic_results", []):
                    link = o.get("link") or o.get("displayed_link")
                    if link:
                        val = link
                        break
        else:
            val = kg.get(field) or kg.get("title") or kg.get("description")
        if val:
                # normalize and dedupe
                if field == "website":
                    # normalize to homepage/root domain if possible
                    from urllib.parse import urlparse, urlunparse

                    parsed = urlparse(val)
                    netloc = parsed.netloc or parsed.path
                    homepage = urlunparse((parsed.scheme or "http", netloc.lower(), "", "", "", ""))
                    canon = normalize_website(homepage)
                    display = canon
                else:
                    canon = normalize_text(val).lower()
                    display = normalize_text(val)
                if (field, canon) not in seen:
                    seen.add((field, canon))
                    candidates.append({"field": field, "value": display, "source": "serpapi.kg", "confidence": 0.9})

        # Prefer the top organic result only (less noisy than scanning every organic/local result)
        top = None
        organic = data.get("organic_results") or []
        if organic and isinstance(organic, list):
            for o in organic:
                if isinstance(o, dict):
                    top = o
                    break

        if top:
            if field == "website":
                link = top.get("link") or top.get("displayed_link")
                if link:
                    # normalize to homepage/root domain
                    from urllib.parse import urlparse, urlunparse

                    p = urlparse(link)
                    netloc = p.netloc or p.path
                    homepage = urlunparse((p.scheme or "http", netloc.lower(), "", "", "", ""))
                    canon = normalize_website(homepage)
                    if (field, canon) not in seen:
                        seen.add((field, canon))
                        candidates.append({"field": field, "value": canon, "source": "serpapi.organic", "confidence": 0.6})
            else:
                title = top.get("title") or top.get("snippet")
                if title:
                    canon = normalize_text(title).lower()
                    if (field, canon) not in seen:
                        seen.add((field, canon))
                        candidates.append({"field": field, "value": normalize_text(title), "source": "serpapi.organic", "confidence": 0.5})

        return candidates


@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
async def search_tavily(query: str, field: str) -> List[Dict[str, Any]]:
    """Call a generic Tavily search endpoint (assumed JSON response).

    Replace the URL below with your Tavily testing/search endpoint.
    """
    # load env at call time
    load_dotenv()
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    TAVILY_URL = os.getenv("TAVILY_URL")

    if not TAVILY_API_KEY or not TAVILY_URL:
        # Tavily not configured -> skip
        return []
    url = TAVILY_URL
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {"q": query, "field": field}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        # Expect data["candidates"] format, be defensive
        candidates = []
        seen = set()
        for c in data.get("candidates", []):
            raw = c.get("value")
            if raw is None:
                continue
            val = normalize_website(raw) if field == "website" else normalize_text(raw)
            canon = val if field == "website" else val.lower()
            if (field, canon) in seen:
                continue
            seen.add((field, canon))
            candidates.append({"field": field, "value": val, "source": "tavily", "confidence": float(c.get("score", 0.6) or 0.6)})
        return candidates


async def generate_multiple_queries(row: Dict[str, Any], num_queries: int = 6) -> List[str]:
    """Use OpenAI to generate 5-7 diverse queries for a single record.
    
    Given company_name, address, state, country, postal_code fields,
    generate varied search queries to maximize recall across search agents.
    """
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        # Fallback: return simple queries
        company = row.get("company_name", "")
        address = row.get("address", "")
        state = row.get("state", "")
        return [f"{company} {state}", company, f"{company} {address}"]
    
    company = row.get("company_name", "").strip()
    address = row.get("address", "").strip()
    state = row.get("state", "").strip()
    country = row.get("country", "").strip()
    postal = row.get("postal_code", "").strip()
    
    # Build context string
    context_parts = [p for p in [company, address, state, postal, country] if p]
    context = " | ".join(context_parts)
    
    prompt_text = f"""Generate exactly {num_queries} diverse search queries to verify if the company '{company}' is located at the given address.
    
Context: {context}

Requirements:
1. Queries should vary in specificity (some specific with address, some just company name)
2. Include queries for company + full address
3. Include queries for company + state/postal code
4. Include queries for just the company name
5. Include queries with possible variations (e.g., without Inc, with parent company search)
6. Generate queries that would work well for web search, knowledge graph lookup, and address verification

Return ONLY the queries, one per line, no numbering or additional text."""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 256,
        "temperature": 0.5,
    }
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                queries = [q.strip() for q in text.split("\n") if q.strip()]
                return queries[:num_queries]
    except Exception as e:
        # Fallback on any error
        pass
    
    # Fallback queries if OpenAI fails
    return [
        f"{company} {address} {state}",
        f"{company} {address}",
        f"{company} {postal_code}",
        f"{company} {state}",
        company,
        f"{company} headquarters",
    ]


async def search_openai(prompt: str, field: str) -> List[Dict[str, Any]]:
    """Lightweight wrapper that would call OpenAI to suggest values.

    This is intentionally minimal: if no key, returns []. You can extend this to use
    the official openai library for completions or newer APIs.
    """
    # load env at call time
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return []

    # Use the Chat Completions API (simple wrapper). This is optional and minimal.
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    # Create a short prompt asking for the field value
    prompt_text = f"Given the entity named '{prompt}', what is the best {field}? Return just the value."
    body = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        candidates = []
        seen = set()
        # Parse response
        choices = data.get("choices") or []
        for ch in choices:
            text = ch.get("message", {}).get("content") if isinstance(ch, dict) else None
            if text:
                val = text.strip().splitlines()[0]
                val_norm = normalize_website(val) if field == "website" else normalize_text(val)
                canon = val_norm if field == "website" else val_norm.lower()
                if (field, canon) in seen:
                    continue
                seen.add((field, canon))
                candidates.append({"field": field, "value": val_norm, "source": "openai", "confidence": 0.4})
        return candidates
    
@retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=4), stop=stop_after_attempt(3))
async def google_entity_lookup(query: str, field: str) -> List[Dict[str, Any]]:
    """Use Google Knowledge Graph Search API to lookup entities.
    
    Extended to extract parent company, address info, and presence metadata.
    """
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_KG_URL = "https://kgsearch.googleapis.com/v1/entities:search"

    if not GOOGLE_API_KEY:
        return []
    
    params = {"query": query, "key": GOOGLE_API_KEY, "limit": 5, "indent": True}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(GOOGLE_KG_URL, params=params)
            r.raise_for_status()
            response = r.json()
    except Exception as e:
        return []

    candidates = []
    seen = set()
    
    for element in response.get("itemListElement", []):
        result = element.get("result", {})

        if field == "website":
            val = result.get("url") or result.get("detailedDescription", {}).get("url")
            if not val:
                continue
            canon = normalize_website(val)

        else:
            val = result.get(field) or result.get("name") or result.get("description")
            if not val:
                continue

            canon = normalize_text(val).lower()

        if (field, canon) not in seen:
            seen.add((field, canon))
            
            # Extract parent company if available
            parent_company = None
            parent_address = None
            description = result.get("detailedDescription", {})
            if description:
                desc_text = description.get("articleBody", "")
                # Simple heuristic: look for parent company mentions
                if "parent" in desc_text.lower():
                    # In production, use NLP to extract parent company
                    pass
            
            candidate = {
                "field": field,
                "value": normalize_website(val) if field == "website" else normalize_text(val),
                "source": "google",
                "confidence": 0.85 if field == "website" else 0.75,
                "parent_company": parent_company,
                "parent_address": parent_address,
            }
            candidates.append(candidate)
    
    return candidates



async def run_search_agents(row: Dict[str, Any], agents: list | None = None, use_multi_query: bool = True) -> List[Dict[str, Any]]:
    """Run selected search agents (or all agents if agents is None) for missing fields.

    agents: list of strings like ['serpapi','tavily','openai','google']
    use_multi_query: if True, generate multiple queries per row for higher recall
    Returns flattened list of candidate dicts, each with extended metadata for tracking.
    """
    # determine which agents to run
    selected = set(a.lower() for a in (agents or [])) if agents is not None else None

    candidates = []

    # helper to decide whether to call an agent
    def want(agent_name: str) -> bool:
        if selected is None:
            return True
        return agent_name.lower() in selected

    # Build queries: either multi-query via LLM or fallback to simple query
    queries = []
    if use_multi_query:
        # Check if row has address fields (new schema)
        if any(row.get(f) for f in ["company_name", "address", "state", "postal_code"]):
            try:
                queries = await generate_multiple_queries(row, num_queries=6)
            except Exception as e:
                # Fallback to simple query on LLM error
                pass
    
    # If multi-query didn't produce results, fall back to simple query
    if not queries:
        query_keys = [
            "company_name",
            "name",
            "company",
            "SRC_CLEANSED_SOURCE_NAME",
            "SOURCE_NAME",
            "ORGANIZATIONPRIMARYNAME",
        ]
        query = ""
        for k in query_keys:
            v = row.get(k)
            if isinstance(v, str) and v.strip():
                query = v.strip()
                break
            if v is not None and not isinstance(v, str):
                query = str(v)
                break
        if query:
            queries = [query]

    if not queries:
        # nothing to query for this row
        return []

    # Run all queries across all target agents
    target_fields = ["company", "name", "website"]
    tasks = []
    query_metadata = []  # Track which query/agent produced each result
    
    for query in queries:
        for tfield in target_fields:
            # Skip field if already present in input
            existing = row.get(tfield)
            if existing is not None and (not isinstance(existing, str) or existing.strip() != ""):
                continue
            
            if want("serpapi"):
                tasks.append((search_serpapi(query, tfield), query, tfield, "serpapi"))
            if want("tavily"):
                tasks.append((search_tavily(query, tfield), query, tfield, "tavily"))
            if want("openai"):
                tasks.append((search_openai(query, tfield), query, tfield, "openai"))
            if want("google"):
                tasks.append((google_entity_lookup(query, tfield), query, tfield, "google"))

    if tasks:
        # Extract coroutines and metadata
        coros = [t[0] for t in tasks]
        metadata = [(t[1], t[2], t[3]) for t in tasks]
        
        results = await asyncio.gather(*coros, return_exceptions=True)
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                continue
            if isinstance(res, list):
                query_used, field, agent = metadata[idx]
                for candidate in res:
                    # Extend candidate with tracking metadata
                    candidate["query_used"] = query_used
                    candidate["agent"] = agent
                    candidates.append(candidate)
    
    return candidates
