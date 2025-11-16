"""Command-line entrypoint for the MDM pipeline."""
import argparse
import asyncio
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from collections import defaultdict
import json

from src.mdm.utils import read_csv, save_csv, rows_from_df, merge_result
from src.mdm.planner import plan_tasks
from src.mdm.search_agents import run_search_agents
from src.mdm.verifier import verify_candidates


def run_pipeline(
    input_path: str, 
    output_path: str, 
    report_path: str | None = None,
    chunk_size: int = 1, 
    planner_config: dict | None = None,
    use_multi_query: bool = True
) -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("mdm.cli")

    df = read_csv(input_path)
    rows = rows_from_df(df)
    tasks = plan_tasks(rows, chunk_size=chunk_size, config=planner_config)
    results: List[Dict[str, Any]] = []
    
    # Track performance metrics for each search mode
    agent_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total_queries": 0,
        "successful_matches": 0,
        "avg_confidence": [],
        "response_times": [],
    })

    async def handle_task(task: Dict[str, Any]):
        task_rows = task.get("rows", [])
        agents = task.get("agents")
        for row in task_rows:
            logger.info("Searching for row id=%s name=%s using agents=%s", 
                       row.get("id"), row.get("company_name", row.get("name")), agents)
            
            candidates = await run_search_agents(
                row, 
                agents=agents, 
                use_multi_query=use_multi_query
            )
            
            # Track agent performance
            for candidate in candidates:
                agent = candidate.get("agent", "unknown")
                agent_performance[agent]["total_queries"] += 1
                conf = float(candidate.get("confidence", 0.5))
                agent_performance[agent]["avg_confidence"].append(conf)
                if conf >= 0.7:
                    agent_performance[agent]["successful_matches"] += 1
            
            best = verify_candidates(candidates, row=row, threshold=0.6)
            merged = merge_result(row, best)
            
            # Add explicit mdm_verified_* columns
            for f_name, val in (best or {}).items():
                try:
                    merged[f"mdm_verified_{f_name}"] = val
                except Exception:
                    merged[f"mdm_verified_{f_name}"] = str(val) if val is not None else ""
            
            results.append(merged)

    async def run_all():
        await asyncio.gather(*(handle_task(t) for t in tasks))

    asyncio.run(run_all())

    # Write results
    import pandas as pd
    out_df = pd.DataFrame(results)
    save_csv(out_df, output_path)
    logger.info(f"Output saved to {output_path}")
    
    # Generate comparative report if requested
    if report_path:
        generate_comparison_report(
            agent_performance, 
            results, 
            report_path, 
            logger
        )


def generate_comparison_report(
    agent_performance: Dict[str, Dict[str, Any]],
    results: List[Dict[str, Any]],
    report_path: str,
    logger
) -> None:
    """Generate a detailed report comparing search mode performance."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MDM PIPELINE - SEARCH MODE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Summary statistics
    total_rows = len(results)
    confirmed_rows = sum(1 for r in results if r.get("mdm_verified_presence_confirmed"))
    
    report_lines.append(f"Total rows processed: {total_rows}")
    report_lines.append(f"Confirmed presences: {confirmed_rows} ({100*confirmed_rows/total_rows:.1f}%)")
    report_lines.append("")
    
    # Per-agent performance
    report_lines.append("-" * 80)
    report_lines.append("SEARCH MODE PERFORMANCE")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for agent_name in sorted(agent_performance.keys()):
        perf = agent_performance[agent_name]
        total = perf["total_queries"]
        if total == 0:
            continue
        
        successes = perf["successful_matches"]
        success_rate = 100 * successes / total if total > 0 else 0
        avg_conf = sum(perf["avg_confidence"]) / len(perf["avg_confidence"]) if perf["avg_confidence"] else 0
        
        report_lines.append(f"Agent: {agent_name.upper()}")
        report_lines.append(f"  Total queries: {total}")
        report_lines.append(f"  Successful matches (conf >= 0.7): {successes} ({success_rate:.1f}%)")
        report_lines.append(f"  Average confidence: {avg_conf:.2f}")
        report_lines.append("")
    
    # Non-match classification breakdown
    report_lines.append("-" * 80)
    report_lines.append("NON-MATCH CLASSIFICATION BREAKDOWN")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    non_match_counts: Dict[str, int] = defaultdict(int)
    for row in results:
        if not row.get("mdm_verified_presence_confirmed"):
            reason = row.get("mdm_verified_non_match_reason", "UNKNOWN")
            non_match_counts[reason] += 1
    
    if non_match_counts:
        for reason in sorted(non_match_counts.keys()):
            count = non_match_counts[reason]
            pct = 100 * count / (total_rows - confirmed_rows)
            report_lines.append(f"{reason:20} : {count:3} ({pct:.1f}%)")
    else:
        report_lines.append("All companies confirmed (no non-matches)")
    report_lines.append("")
    
    # Sample results
    report_lines.append("-" * 80)
    report_lines.append("SAMPLE RESULTS (First 5 rows)")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for i, row in enumerate(results[:5]):
        report_lines.append(f"Row {i+1}: {row.get('company_name', row.get('name', 'Unknown'))}")
        report_lines.append(f"  Address: {row.get('address', 'N/A')} {row.get('state', '')} {row.get('postal_code', '')}")
        report_lines.append(f"  Presence: {row.get('mdm_verified_presence_confirmed')}")
        if row.get("mdm_verified_presence_source"):
            report_lines.append(f"  Source: {row.get('mdm_verified_presence_source')}")
        if not row.get("mdm_verified_presence_confirmed"):
            report_lines.append(f"  Reason: {row.get('mdm_verified_non_match_reason', 'Unknown')}")
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger = __import__('logging').getLogger("mdm.cli")
    logger.info(f"Comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MDM multi-agent pipeline")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--report", help="Optional path for comparison report")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--no-multi-query", action="store_true", 
                       help="Disable LLM-based multi-query generation")
    parser.add_argument("--planner-config", help="Optional JSON file with planner config mapping fields to agents")
    args = parser.parse_args()

    # load env
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    # load planner config if provided
    planner_config = None
    if getattr(args, "planner_config", None):
        with open(args.planner_config, "r", encoding="utf-8") as fh:
            planner_config = json.load(fh)

    # Call run_pipeline with new parameters
    run_pipeline(
        args.input, 
        args.output,
        report_path=args.report,
        chunk_size=args.chunk_size, 
        planner_config=planner_config,
        use_multi_query=not args.no_multi_query
    )


if __name__ == "__main__":
    main()
