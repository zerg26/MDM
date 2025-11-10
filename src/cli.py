"""Command-line entrypoint for the MDM pipeline."""
import argparse
import asyncio
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from src.mdm.utils import read_csv, save_csv, rows_from_df, merge_result
from src.mdm.planner import plan_tasks
from src.mdm.search_agents import run_search_agents
from src.mdm.verifier import verify_candidates


def run_pipeline(input_path: str, output_path: str, chunk_size: int = 1, planner_config: dict | None = None) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mdm.cli")

    df = read_csv(input_path)
    rows = rows_from_df(df)
    # planner config may be provided via environment or CLI
    tasks = plan_tasks(rows, chunk_size=chunk_size, config=planner_config)
    results: List[Dict[str, Any]] = []


    async def handle_task(task: Dict[str, Any]):
        task_rows = task.get("rows", [])
        agents = task.get("agents")
        for row in task_rows:
            logger.info("Searching for row id=%s name=%s using agents=%s", row.get("id"), row.get("name"), agents)
            candidates = await run_search_agents(row, agents=agents)
            best = verify_candidates(candidates)
            # Merge original row + verifier result
            merged = merge_result(row, best)
            # Also add explicit mdm_verified_* columns so outputs are unambiguous
            for f_name, val in (best or {}).items():
                try:
                    merged[f"mdm_verified_{f_name}"] = val
                except Exception:
                    # fallback to string coercion if necessary
                    merged[f"mdm_verified_{f_name}"] = str(val) if val is not None else ""
            results.append(merged)

    async def run_all():
        await asyncio.gather(*(handle_task(t) for t in tasks))

    # Use asyncio.run() to create a fresh loop and avoid deprecation warnings
    asyncio.run(run_all())

    # write results
    import pandas as pd

    out_df = pd.DataFrame(results)
    save_csv(out_df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Run MDM multi-agent pipeline")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--planner-config", help="Optional JSON file with planner config mapping fields to agents")
    args = parser.parse_args()

    # load env
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # load planner config if provided
    planner_config = None
    if getattr(args, "planner_config", None):
        import json

        with open(args.planner_config, "r", encoding="utf-8") as fh:
            planner_config = json.load(fh)

    # Call run_pipeline and pass planner_config if provided
    run_pipeline(args.input, args.output, chunk_size=args.chunk_size, planner_config=planner_config)


# run_pipeline_with_planner_config removed; run_pipeline now accepts planner_config


if __name__ == "__main__":
    main()
