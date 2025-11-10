# Changelog

All notable changes to this repository will be documented in this file.

## v0.1-alpha (2025-11-10)

- Alpha release of the MDM pipeline.
- Added pipeline outputs produced by a smoke run:
  - `sample_data/output_smoke.csv` — full pipeline output including `mdm_verified_*` columns
  - `sample_data/compact_smoke.csv` — compact review CSV preferring `mdm_verified_*` columns
- Behavior included in this release:
  - Planner → Search agents (OpenAI, SerpAPI, Tavily, registry) → Verifier flow
  - Normalization and deduplication of candidate values
  - `mdm_verified_{field}` columns appended to output for easy inspection
  - Registry adapter and registry-boost logic (configurable via `.env`)
  - Smoke test executed against `sample_data/input.csv`; HTTP responses from SerpAPI/OpenAI noted as 200 OK in logs

Notes:
- This is an alpha tag. For beta, consider adding: stricter name-extraction heuristics, rate-limiting and CI with recorded fixtures, and pushing the tag to the remote repository.
