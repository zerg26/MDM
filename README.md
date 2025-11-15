# MDM — Multi-Agent Data Matching Pipeline

Note: recent alpha changes are summarized in `ALPHA_NOTES.md`.

This repository provides a small multi-agent pipeline to fill missing CSV fields
by querying multiple search agents (SerpAPI, OpenAI, Tavily), reconciling
their answers via a verifier with optional registry boosting, and producing
an output CSV.

Quick features
- Planner that routes rows to selected agents (per-row overrides supported).
- Search agents: SerpAPI, OpenAI, Tavily (Tavily optional).
- Normalization, deduplication, and a verifier that uses a registry boost.
- Tests and debug scripts included.

Setup
1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with keys (example in `.env.example`):

- OPENAI_API_KEY
- SERPAPI_API_KEY
- GOOGLE_API_KEY
- (optional) TAVILY_API_KEY and TAVILY_URL
- (optional) REGISTRY_URL (if you have an external registry)
- (optional) REGISTRY_BOOST (float, default 0.5)

Quick runs
- Run the CLI on the sample data:

```powershell
python -m src.cli --input sample_data/input.csv --output sample_data/output.csv --chunk-size 1
```

- Provide a planner config JSON to override agent selection per-row:

```json
{
  "overrides": [
    {"match": {"id": 42}, "agents": ["serpapi","openai"], "force": true}
  ]
}
```

Troubleshooting
- for GOOGLE_API_KEY look at https://support.google.com/googleapi/answer/6158841?hl=en and enable Knowledge Graph Search API
- Tavily DNS / connection errors: if you see `getaddrinfo failed` when running
  Tavily probes, ensure `TAVILY_URL` is set to a resolvable endpoint and the
  machine has outbound network access. Tavily is optional; if unset, the
  pipeline will skip it.
- Registry integration: set `REGISTRY_URL` to enable remote registry lookups.
  The code can call a remote registry at `REGISTRY_URL`. Expected request:

  POST {REGISTRY_URL}
  Content-Type: application/json
  Body: {"field": "company", "value": "Acme Corp"}

  Expected response (either):
  - {"match": true, "confidence": 0.87}
  - {"data": {"match": true, "confidence": 0.87}}

  Authentication options (via `.env`):
  - `REGISTRY_API_KEY`: if set, will send Authorization: Bearer <key>
  - `REGISTRY_AUTH_HEADER` and `REGISTRY_AUTH_VALUE`: to send a custom header

  On network or parsing errors the code falls back to a local fuzzy-stub
  heuristic to avoid breaking the pipeline. The verifier supports `REGISTRY_BOOST`.

Tests
- Run the test suite with:

```powershell
pytest -q
```

If you add or change async tests, `pytest-asyncio` is already listed in
`requirements.txt` and `pytest.ini` config is present.

Notes
- The repository uses defensive parsing and normalization to avoid noisy
  duplicates. For production use you may want to wire a real registry API and
  implement rate-limiting/backoff tuned to your vendor quotas.

License: MIT
MDM — Multi-Agent Data Matching Pipeline

This repository contains a scaffold for a multi-agent pipeline that reads a CSV with missing data, uses a Planner agent to split tasks, multiple search agents (OpenAI, Tavily, SerpAPI) to fetch candidate values, and a Verifier agent to reconcile results and produce a completed CSV.

Quick start
1. Copy `.env.example` to `.env` and add your API keys.
2. Create a Python environment and install dependencies:

```powershell
# from repo root
python -m pip install -r requirements.txt
```

3. Run the CLI (example):

```powershell
python -m src.cli --input sample_data/input.csv --output sample_data/output.csv
```

4. Run tests:

```powershell
python -m pytest -q
```

Notes
- The search agent implementations include HTTP hooks to SerpAPI and a generic Tavily endpoint; OpenAI calls are optional. In tests, these are mocked.
- This is a scaffolded example to demonstrate the architecture; please fill in API endpoint details and keys before doing large-scale runs.
