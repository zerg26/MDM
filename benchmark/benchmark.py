"""Coordination-efficiency benchmark for the MDM pipeline.

Two things are measured, both offline (agent latency is *simulated* with
asyncio.sleep, so no API keys or network are needed):

1. Concurrent orchestration vs. a sequential baseline -> wall-clock speedup.
2. Auto-routing (planner.decide_agents_for_row) vs. hand-authored routing ->
   manual setup-time reduction.

Usage::

    python benchmark/benchmark.py --records 200 --latency-ms 400 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import time
import sys
from pathlib import Path

# Make the project importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mdm.planner import decide_agents_for_row  # noqa: E402

# Assumed minutes a human spends hand-authoring routing rules per record.
MANUAL_SECONDS_PER_RECORD = 20.0


def _sample_records(n: int):
    return [
        {"id": i, "name": f"Company {i}", "company": "", "website": ""}
        for i in range(n)
    ]


async def _agent_call(latency_ms: float) -> None:
    await asyncio.sleep(latency_ms / 1000.0)


async def _run_sequential(records, agents_per_record, latency_ms: float) -> float:
    start = time.perf_counter()
    for _ in records:
        for _ in range(agents_per_record):
            await _agent_call(latency_ms)
    return time.perf_counter() - start


async def _run_concurrent(records, agents_per_record, latency_ms: float, concurrency: int) -> float:
    sem = asyncio.Semaphore(concurrency)

    async def one_call():
        async with sem:
            await _agent_call(latency_ms)

    start = time.perf_counter()
    tasks = [one_call() for _ in records for _ in range(agents_per_record)]
    await asyncio.gather(*tasks)
    return time.perf_counter() - start


def _routing_setup(records):
    """Return (auto_seconds, manual_seconds, avg_agents)."""
    start = time.perf_counter()
    total_agents = 0
    for r in records:
        total_agents += len(decide_agents_for_row(r))
    auto_seconds = time.perf_counter() - start
    manual_seconds = len(records) * MANUAL_SECONDS_PER_RECORD
    avg_agents = total_agents / len(records) if records else 0
    return auto_seconds, manual_seconds, avg_agents


def main() -> None:
    parser = argparse.ArgumentParser(description="MDM coordination benchmark")
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--latency-ms", type=float, default=400.0)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    records = _sample_records(args.records)
    auto_s, manual_s, avg_agents = _routing_setup(records)
    agents_per_record = max(1, round(avg_agents))

    seq = asyncio.run(_run_sequential(records, agents_per_record, args.latency_ms))
    conc = asyncio.run(_run_concurrent(records, agents_per_record, args.latency_ms, args.concurrency))

    speedup = seq / conc if conc > 0 else float("inf")
    setup_reduction = 100.0 * (1 - auto_s / manual_s) if manual_s > 0 else 0.0

    print("=" * 60)
    print("MDM COORDINATION BENCHMARK")
    print("=" * 60)
    print(f"Records:                {args.records}")
    print(f"Agents/record (auto):   {agents_per_record}")
    print(f"Simulated latency:      {args.latency_ms:.0f} ms/agent-call")
    print(f"Concurrency:            {args.concurrency}")
    print("-" * 60)
    print("Orchestration efficiency (concurrent vs sequential)")
    print(f"  Sequential baseline:  {seq:8.2f} s")
    print(f"  Concurrent:           {conc:8.2f} s")
    print(f"  Speedup:              {speedup:8.2f}x")
    print("-" * 60)
    print("Manual setup-time reduction (auto vs hand-authored routing)")
    print(f"  Hand-authored (est):  {manual_s:8.1f} s")
    print(f"  Auto-routing:         {auto_s:8.4f} s")
    print(f"  Reduction:            {setup_reduction:8.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
