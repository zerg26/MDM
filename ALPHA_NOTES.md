MDM — Alpha notes

This file captures the recent changes and rationale made during the alpha work.

Highlights
- Agents now load `.env` at call-time so keys can be added/changed without module reload.
- `src/cli` now writes explicit `mdm_verified_{field}` columns alongside original input columns.
- `search_agents.search_serpapi` was tuned to prefer `knowledge_graph` entries and the top organic result (homepage/root domain) to reduce noisy page-title suggestions.
- `run_search_agents` builds a query from common source/name fields (e.g., `SOURCE_NAME`, `SRC_CLEANSED_SOURCE_NAME`) when `name`/`company` are not present.
- `verifier` applies optional registry boosting via `REGISTRY_BOOST` and falls back to a local fuzzy-stub.


Next steps (beta)
- Improve canonical company-name extraction heuristics (prefer short KG titles, infer from domain, or tighten OpenAI prompt).
- Add rate-limiting/backoff tuned to vendor quotas.
- Add CI that runs tests with recorded fixtures rather than live keys.

