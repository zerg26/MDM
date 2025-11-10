import argparse
import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# ensure project root is on sys.path so `src` package can be imported when running script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv()

from src.mdm.utils import read_csv, rows_from_df
from src.mdm.search_agents import run_search_agents
from src.mdm.verifier import verify_candidates


async def process_row(row, agents):
    candidates = await run_search_agents(row, agents=agents)
    best = verify_candidates(candidates)
    return candidates, best


def main():
    parser = argparse.ArgumentParser(description="Debug search agents for a CSV")
    parser.add_argument("input", help="Input CSV path")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows to debug")
    args = parser.parse_args()

    df = read_csv(args.input)
    rows = rows_from_df(df)
    agents = ["openai", "serpapi", "tavily", "registry"]

    async def run():
        for i, row in enumerate(rows[: args.rows]):
            print(f"\n--- ROW {i} ---")
            print(json.dumps(row, indent=2, ensure_ascii=False))
            try:
                candidates, best = await process_row(row, agents)
                print("\nCANDIDATES:")
                print(json.dumps(candidates, indent=2, ensure_ascii=False))
                print("\nBEST:")
                print(json.dumps(best, indent=2, ensure_ascii=False))
            except Exception as e:
                print("Error while processing row:", e)

    asyncio.run(run())


if __name__ == "__main__":
    main()
