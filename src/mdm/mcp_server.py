"""MCP server exposing the MDM agents as tools.

Wraps the existing pipeline functions so an orchestrator (or another agent) can
delegate over the Model Context Protocol instead of importing Python functions:

    - ``route_record``  -> planner.decide_agents_for_row
    - ``search_record`` -> search_agents.run_search_agents
    - ``verify_record`` -> verifier.verify_candidates

Run it::

    python -m src.mdm.mcp_server            # stdio transport (default)
    python -m src.mdm.mcp_server --http     # streamable-http on :8000/mcp

The plain ``_route_record`` / ``_search_record`` / ``_verify_record`` helpers are
also imported directly by ``mcp_client`` for the fast in-process path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .planner import decide_agents_for_row
from .search_agents import run_search_agents
from .verifier import verify_candidates


# --------------------------------------------------------------------------- #
# Core implementations (shared by MCP tools and the in-process client path)
# --------------------------------------------------------------------------- #
def _route_record(row: Dict[str, Any], routing_config: Optional[Dict[str, Any]] = None) -> List[str]:
    return decide_agents_for_row(row, routing_config)


async def _search_record(
    row: Dict[str, Any],
    agents: Optional[List[str]] = None,
    use_multi_query: bool = True,
) -> List[Dict[str, Any]]:
    return await run_search_agents(row, agents=agents, use_multi_query=use_multi_query)


def _verify_record(candidates: List[Dict[str, Any]], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return verify_candidates(candidates, row=row)


# --------------------------------------------------------------------------- #
# MCP server + tool wrappers
# --------------------------------------------------------------------------- #
def build_server(host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    mcp = FastMCP("mdm", host=host, port=port)

    @mcp.tool()
    def route_record(row: Dict[str, Any], routing_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Decide which search agents to run for a record."""
        return _route_record(row, routing_config)

    @mcp.tool()
    async def search_record(
        row: Dict[str, Any],
        agents: Optional[List[str]] = None,
        use_multi_query: bool = True,
    ) -> List[Dict[str, Any]]:
        """Run the selected search agents and return candidate dicts."""
        return await _search_record(row, agents=agents, use_multi_query=use_multi_query)

    @mcp.tool()
    def verify_record(candidates: List[Dict[str, Any]], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reconcile candidates into a verified record + presence/classification."""
        return _verify_record(candidates, row)

    return mcp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MDM MCP agent server")
    parser.add_argument("--http", action="store_true", help="Serve over streamable-http instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = build_server(host=args.host, port=args.port)
    if args.http:
        server.run(transport="streamable-http")
    else:
        server.run()


if __name__ == "__main__":
    main()
