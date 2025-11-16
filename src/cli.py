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
    not_confirmed = total_rows - confirmed_rows
    
    report_lines.append(f"Total rows processed:           {total_rows}")
    report_lines.append(f"Companies confirmed at address: {confirmed_rows} ({100*confirmed_rows/total_rows:.1f}%)")
    report_lines.append(f"Companies NOT confirmed:        {not_confirmed} ({100*not_confirmed/total_rows:.1f}%)")
    report_lines.append("")
    
    # Per-agent performance with detailed metrics
    report_lines.append("-" * 80)
    report_lines.append("SEARCH MODE EFFECTIVENESS ANALYSIS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # Calculate per-agent stats
    agent_stats = {}
    for agent_name in sorted(agent_performance.keys()):
        perf = agent_performance[agent_name]
        total_queries = perf["total_queries"]
        if total_queries == 0:
            continue
        
        successes = perf["successful_matches"]
        success_rate = 100 * successes / total_queries
        avg_conf = sum(perf["avg_confidence"]) / len(perf["avg_confidence"]) if perf["avg_confidence"] else 0
        min_conf = min(perf["avg_confidence"]) if perf["avg_confidence"] else 0
        max_conf = max(perf["avg_confidence"]) if perf["avg_confidence"] else 0
        
        agent_stats[agent_name] = {
            "total": total_queries,
            "success": successes,
            "success_rate": success_rate,
            "avg_conf": avg_conf,
            "min_conf": min_conf,
            "max_conf": max_conf,
        }
    
    # Display per-agent statistics
    for agent_name in sorted(agent_stats.keys()):
        stats = agent_stats[agent_name]
        report_lines.append(f"Agent: {agent_name.upper()}")
        report_lines.append(f"  Total queries executed:           {stats['total']}")
        report_lines.append(f"  High-confidence matches (≥0.7):  {stats['success']} ({stats['success_rate']:.1f}%)")
        report_lines.append(f"  Average confidence score:         {stats['avg_conf']:.3f}")
        report_lines.append(f"  Confidence range:                 {stats['min_conf']:.2f} - {stats['max_conf']:.2f}")
        report_lines.append(f"  Coverage (% of total rows):       {100*stats['total']/total_rows:.1f}%")
        report_lines.append("")
    
    # Agent ranking
    report_lines.append("-" * 80)
    report_lines.append("AGENT RANKING BY EFFECTIVENESS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]["success_rate"], reverse=True)
    for rank, (agent_name, stats) in enumerate(sorted_agents, 1):
        report_lines.append(f"{rank}. {agent_name.upper():15} - {stats['success_rate']:6.1f}% success rate | "
                          f"Avg confidence: {stats['avg_conf']:.3f} | "
                          f"Queries: {stats['total']}")
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
        report_lines.append(f"Total non-matches: {not_confirmed}")
        report_lines.append("")
        for reason in sorted(non_match_counts.keys()):
            count = non_match_counts[reason]
            pct = 100 * count / not_confirmed if not_confirmed > 0 else 0
            report_lines.append(f"  {reason:25} : {count:3} ({pct:6.1f}%)")
    else:
        report_lines.append("All companies confirmed (no non-matches)")
    report_lines.append("")
    
    # Search source attribution for confirmed matches
    report_lines.append("-" * 80)
    report_lines.append("SOURCE ATTRIBUTION FOR CONFIRMED MATCHES")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    source_counts: Dict[str, int] = defaultdict(int)
    confirmed_via_source = []
    for row in results:
        if row.get("mdm_verified_presence_confirmed"):
            sources = row.get("mdm_verified_presence_source", "").split("; ")
            for source in sources:
                if source.strip():
                    source_counts[source.strip()] += 1
                    confirmed_via_source.append(source.strip())
    
    if source_counts:
        report_lines.append(f"Total confirmed matches: {confirmed_rows}")
        report_lines.append("")
        for source in sorted(source_counts.keys()):
            count = source_counts[source]
            pct = 100 * count / confirmed_rows
            report_lines.append(f"  {source:25} : {count:3} matches ({pct:6.1f}%)")
    report_lines.append("")
    
    # Sample results
    report_lines.append("-" * 80)
    report_lines.append("SAMPLE RESULTS (First 5 rows)")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for i, row in enumerate(results[:5]):
        report_lines.append(f"Row {i+1}: {row.get('company_name', row.get('name', row.get('SOURCE_NAME', 'Unknown')))}")
        
        # Support both query CSV schema and legacy schema for address
        address = row.get('address') or row.get('SOURCE_ADDRESS') or row.get('SRC_CLEANSED_STREET_ADDRESS') or 'N/A'
        city = row.get('city') or row.get('SOURCE_CITY') or row.get('SRC_CLEANSED_CITY') or ''
        state = row.get('state') or row.get('SOURCE_STATE') or row.get('SRC_CLEANSED_STATE') or ''
        postal = row.get('postal_code') or row.get('SOURCE_POSTAL_CODE') or ''
        
        report_lines.append(f"  Address: {address} {city} {state} {postal}".strip())
        report_lines.append(f"  Presence: {row.get('mdm_verified_presence_confirmed')}")
        if row.get("mdm_verified_presence_source"):
            report_lines.append(f"  Source(s): {row.get('mdm_verified_presence_source')}")
        if not row.get("mdm_verified_presence_confirmed"):
            report_lines.append(f"  Reason: {row.get('mdm_verified_non_match_reason', 'Unknown')}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("-" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    best_agent = sorted_agents[0][0] if sorted_agents else "Unknown"
    best_rate = sorted_agents[0][1]["success_rate"] if sorted_agents else 0
    
    report_lines.append(f"• Best-performing agent: {best_agent.upper()} with {best_rate:.1f}% success rate")
    report_lines.append(f"• Overall confirmation rate: {100*confirmed_rows/total_rows:.1f}%")
    report_lines.append(f"• Most common non-match reason: {max(non_match_counts, key=non_match_counts.get) if non_match_counts else 'N/A'}")
    report_lines.append("")
    report_lines.append("To improve results:")
    report_lines.append("  1. Prioritize multi-query approach (5-7 queries per record improves recall)")
    report_lines.append("  2. Focus on high-performing search agents for production runs")
    report_lines.append("  3. Investigate non-matches with 'AMBIGUOUS' or 'NOT_FOUND' classifications")
    report_lines.append("  4. Consider fuzzy matching to detect spelling/numerical errors in input")
    report_lines.append("")
    
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Comparison report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MDM multi-agent pipeline for query verification")
    parser.add_argument("--input", help="Input CSV path (query_group1.csv or query_test_more_missing_values.csv)")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--report", help="Optional path for comparison report")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--no-multi-query", action="store_true",
                       help="Disable LLM-based multi-query generation")
    parser.add_argument("--planner-config", help="Optional JSON file with planner config mapping fields to agents")
    parser.add_argument("--batch", action="store_true",
                       help="Process both query_group1.csv and query_test_more_missing_values.csv with reports")
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

    # If batch mode, process both query files
    if args.batch:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        query_files = [
            ("sample_data/query_group1.csv", "sample_data/query_group1_cli_output.csv", "sample_data/query_group1_report.txt"),
            ("sample_data/query_test_more_missing_values.csv", "sample_data/query_test_more_missing_values_cli_output.csv", "sample_data/query_test_more_missing_values_report.txt"),
        ]
        
        for input_file, output_file, report_file in query_files:
            input_path = os.path.join(base_dir, input_file)
            output_path = os.path.join(base_dir, output_file)
            report_path = os.path.join(base_dir, report_file)
            
            print(f"\n{'='*80}")
            print(f"Processing: {input_file}")
            print(f"{'='*80}\n")
            
            # Always generate reports in batch mode
            run_pipeline(
                input_path,
                output_path,
                report_path=report_path,
                chunk_size=args.chunk_size,
                planner_config=planner_config,
                use_multi_query=not args.no_multi_query
            )
        
        print(f"\n{'='*80}")
        print("Batch processing complete!")
        print(f"Generated reports:")
        for _, _, report_file in query_files:
            report_path = os.path.join(base_dir, report_file)
            print(f"  - {report_path}")
        print(f"{'='*80}\n")
    else:
        # Standard single file mode
        if not args.input or not args.output:
            parser.error("--input and --output are required when not using --batch mode")
        
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