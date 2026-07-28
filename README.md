# Multi-Agent Data Matching (MDM) Pipeline

## Business Context
This pipeline was developed to tackle data entropy in enterprise systems,
specifically targeting the resolution of 90K unmatched company records from a
5.5M customer Master Data Management (MDM) system.

## Overview
Traditional data enrichment tools often suffer from single-source bias, where a
single API provider might miss a niche business or return outdated physical
addresses. This project implements a consensus-based, **agentic** architecture to
resolve ambiguous or incomplete corporate records. It orchestrates concurrent
queries across multiple search providers, normalizes the heterogeneous outputs,
and synthesizes a final, high-confidence enriched record — with an explainable,
self-healing control loop.

## Tech Stack & Search Providers
* **Language:** Python (async execution for unblocking batch operations)
* **Orchestration:** LangGraph state machine with a self-healing control loop
* **Agent-to-agent:** Model Context Protocol (MCP) server + client
* **Interfaces:** CLI for batch processing; Gradio for interactive debugging and
  human-in-the-loop validation
* **Search APIs:** SerpAPI, Google Knowledge Graph, Tavily, OpenAI (dynamic query
  generation)

## Core System Components
* **Multi-Agent Orchestration:** Routes missing records to the search APIs
  concurrently. If a company name is ambiguous, an LLM agent generates diverse
  query variations to improve recall.
* **Registry Boost:** A verification mechanism that anchors search results to
  known-good data registries, reducing false positives in website / headquarters
  fields.
* **Granular Non-Match Classification:** Instead of a binary "match/no-match", the
  verifier classifies resolution failures into a granular taxonomy
  (`NON_MATCH_CLASSIFICATIONS` in `src/mdm/verifier.py`) — e.g. `WRONG_STATE`,
  `OFFICE_VACATED`, `PO_BOX`, `MERGED_ACQUIRED`, `AMBIGUOUS`, `INCOMPLETE_DATA` —
  for explainable reasons a record could not be confirmed.
* **Self-Healing Graph:** A LangGraph engine that escalates routing and retries on
  unconfirmed records, emitting a per-record decision trace.

## Agent Orchestration Architecture

The pipeline is orchestrated as an explicit **LangGraph** state machine
(`src/mdm/graph.py`) coordinating specialized agents end-to-end:

1. **router_agent** — dynamic per-record agent selection (cheap heuristics first;
   widens to the full agent set on a heal retry).
2. **search_agent** — concurrent fan-out across SerpAPI, Google Knowledge Graph,
   Tavily, and OpenAI (with LLM query expansion for recall).
3. **verifier_agent** — cross-agent consensus voting with registry boost, plus
   presence confirmation and non-match classification.
4. **healer_agent** — *self-healing* control loop: on an unconfirmed record it
   escalates routing (widens the agent set, forces multi-query) and retries up to
   `MDM_MAX_HEAL_ATTEMPTS` (default 2).

```
START → router_agent → search_agent → verifier_agent ─┬─(confirmed)→ END
                ▲                                      │
                └──────────── healer_agent ◀───(retry)─┘
```

Every node appends to a structured `trace`, so each decision is explainable.

### Agent-to-agent communication (MCP)

The agents are also exposed over the **Model Context Protocol** so an
orchestrator (or another agent) can delegate tasks over a standard protocol
rather than importing Python functions:

- `src/mdm/mcp_server.py` — MCP server exposing `route_record`, `search_record`,
  and `verify_record` tools (`python -m src.mdm.mcp_server --http --port 8000`).
- `src/mdm/mcp_client.py` — client wrapper used by the graph.

Set `MDM_USE_MCP=1` (in-process) or `MDM_MCP_ENDPOINT=http://host:8000/mcp`
(networked) to route the graph's search step through MCP.

## Local Setup & Execution

1. **Clone the repository:**

   ```bash
   git clone https://github.com/zerg26/MDM.git
   cd MDM
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment variables:** create a `.env` in the repo root (see
   `.env.example`):

   ```
   OPENAI_API_KEY=your_key_here
   SERPAPI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   TAVILY_URL=...            # optional
   GOOGLE_API_KEY=your_key   # optional, for Knowledge Graph
   REGISTRY_URL=...          # optional
   REGISTRY_BOOST=0.5        # optional
   ```

4. **Run it:**

   ```bash
   # Batch CLI (legacy engine)
   python -m src.cli --input sample_data/query_group1.csv --output out.csv

   # Batch CLI through the self-healing graph (adds mdm_failure_category +
   # mdm_decision_trace columns)
   python -m src.cli --input sample_data/query_group1.csv --output out.csv --engine graph

   # A single record through the self-healing graph
   python -c "from src.mdm.graph import run_row_sync; print(run_row_sync({'name':'Acme Corp'}))"

   # Interactive debugging / human-in-the-loop UI
   python app.py

   # MCP agent server
   python -m src.mdm.mcp_server --http --port 8000
   ```

   Set `MDM_LOG_JSON=1` for structured JSON logs.

## Deployment

- **Docker:** `docker compose up --build` starts the MCP agent server plus a
  pipeline worker that delegates its search step to it.
- **Kubernetes:** `k8s/` contains the MCP `Deployment`/`Service`, a batch `Job`,
  and a secret template (`secret.example.yaml`) for API keys.

## Benchmarks

`benchmark/benchmark.py` measures coordination efficiency (concurrent
orchestration vs. sequential baseline) and manual setup-time reduction
(auto-routing vs. hand-authored routing). Latency is simulated, so it runs
without live keys:

```bash
python benchmark/benchmark.py --records 200 --latency-ms 400 --concurrency 8
```

## Tests

```bash
pytest -q
```

## Known Limitations & Future Work
*(Alpha build validating the multi-agent consensus thesis.)*
* **API Quotas & Resiliency:** heavily dependent on external API availability;
  production needs token-bucket rate limiting tuned to vendor quotas.
* **Heuristic Scoring:** confidence scoring is heuristic; a probabilistic model
  trained on labeled data is a roadmap item.
* **Security:** the Gradio UI has no authentication and is for local debugging
  only. (Structured JSON logging is available via `MDM_LOG_JSON=1`.)

MIT License
