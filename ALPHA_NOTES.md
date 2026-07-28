MDM — Alpha notes

This file captures the recent changes and rationale made during the alpha work.

Highlights
- Agents now load `.env` at call-time so keys can be added/changed without module reload.
- `src/cli` now writes explicit `mdm_verified_{field}` columns alongside original input columns.
- `search_agents.search_serpapi` was tuned to prefer `knowledge_graph` entries and the top organic result (homepage/root domain) to reduce noisy page-title suggestions.
- `run_search_agents` builds a query from common source/name fields (e.g., `SOURCE_NAME`, `SRC_CLEANSED_SOURCE_NAME`) when `name`/`company` are not present.
- `verifier` applies optional registry boosting via `REGISTRY_BOOST` and falls back to a local fuzzy-stub.

Agentic layer (added)
- `src/mdm/graph.py`: LangGraph self-healing state machine
  (router → search → verifier → healer) wrapping the existing agents; emits a
  per-record decision `trace`. `run_row` / `run_row_sync` entrypoints.
  `MDM_MAX_HEAL_ATTEMPTS` bounds retries (default 2).
- `src/cli.py`: `--engine graph` routes rows through the graph and writes
  `mdm_failure_category` + `mdm_decision_trace`; a real batch argparse entrypoint
  (`--input/--output/--engine/--ui`). Also fixed two incomplete-rename bugs from
  commit 5b52be9 (`routing_config`, and the `records` task key) that had made the
  legacy CLI produce empty output.
- `src/mdm/mcp_server.py` + `mcp_client.py`: MCP tools (`route_record`,
  `search_record`, `verify_record`); routed via `MDM_USE_MCP` / `MDM_MCP_ENDPOINT`.
- `app.py`: standalone Gradio entrypoint (extracted from `cli.build_ui`), with an
  explainable self-healing-graph tab and accept/override for human-in-the-loop.
- `src/mdm/logging_config.py`: structured JSON logging via `MDM_LOG_JSON=1`.
- `Dockerfile`, `docker-compose.yml`, `k8s/`: containerized MCP server + worker.
- `benchmark/benchmark.py`: concurrency and auto-routing benchmarks (mocked latency).

Next steps (beta)
- Improve canonical company-name extraction heuristics (prefer short KG titles, infer from domain, or tighten OpenAI prompt).
- Add rate-limiting/backoff tuned to vendor quotas.
- Add CI that runs tests with recorded fixtures rather than live keys.
- Vector-search/RAG entity similarity as an alternative to the registry stub.

