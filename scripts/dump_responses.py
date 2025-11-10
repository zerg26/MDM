import os
import json
from dotenv import load_dotenv
import httpx


def trunc(s, n=4000):
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s)
        except Exception:
            s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def main():
    load_dotenv()
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    SERPAPI_URL = os.getenv("SERPAPI_URL", "https://serpapi.com/search.json")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    TAVILY_URL = os.getenv("TAVILY_URL", "https://api.tavily.example/search")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    q = "Acme Corp"

    print("=== SERPAPI ===", SERPAPI_URL)
    try:
        params = {"q": q, "api_key": SERPAPI_API_KEY}
        with httpx.Client(timeout=20.0) as client:
            r = client.get(SERPAPI_URL, params=params)
        print("STATUS:", r.status_code)
        try:
            j = r.json()
            print("JSON keys:", list(j.keys()))
            print("RAW:", trunc(json.dumps(j, indent=2)))
        except Exception:
            print("TEXT:", trunc(r.text))
    except Exception as e:
        print("ERROR:", repr(e))

    print("\n=== TAVILY ===", TAVILY_URL)
    try:
        headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"} if TAVILY_API_KEY else {}
        payload = {"q": q, "field": "company"}
        with httpx.Client(timeout=20.0) as client:
            r = client.post(TAVILY_URL, json=payload, headers=headers)
        print("STATUS:", r.status_code)
        try:
            j = r.json()
            print("JSON keys:", list(j.keys()))
            print("RAW:", trunc(json.dumps(j, indent=2)))
        except Exception:
            print("TEXT:", trunc(r.text))
    except Exception as e:
        print("ERROR:", repr(e))

    print("\n=== OPENAI === (chat/completions)")
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": f"Given the entity named '{q}', what is the company name? Return just the value."}],
            "max_tokens": 60,
            "temperature": 0,
        }
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=body, headers=headers)
        print("STATUS:", r.status_code)
        try:
            j = r.json()
            print("JSON keys:", list(j.keys()))
            print("RAW:", trunc(json.dumps(j, indent=2)))
        except Exception:
            print("TEXT:", trunc(r.text))
    except Exception as e:
        print("ERROR:", repr(e))


if __name__ == "__main__":
    main()
