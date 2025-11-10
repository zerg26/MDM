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


def main():
    df = read_csv('sample_data/input.csv')
    rows = rows_from_df(df)
    row = rows[0]
    print('Row:', row)
    agents = ['openai', 'serpapi', 'tavily', 'registry']

    async def run():
        candidates = await run_search_agents(row, agents=agents)
        print('\nCANDIDATES:')
        print(json.dumps(candidates, indent=2))
        best = verify_candidates(candidates)
        print('\nBEST:')
        print(json.dumps(best, indent=2))

    asyncio.run(run())


if __name__ == '__main__':
    main()
