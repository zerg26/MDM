"""Client wrapper for the MDM MCP tools.

Routing is controlled by environment:

    - ``MDM_MCP_ENDPOINT=http://host:8000/mcp`` -> call the networked MCP server.
    - ``MDM_USE_MCP=1`` (and no endpoint)        -> call the tool impls in-process.

The graph's search node imports :func:`search_record` from here. ``route_record``
and ``verify_record`` are provided for symmetry / external orchestrators.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _endpoint() -> Optional[str]:
    return os.getenv("MDM_MCP_ENDPOINT")


def _parse_tool_result(result: Any) -> Any:
    """Extract the Python payload from an MCP CallToolResult.

    FastMCP returns structured output under ``structuredContent`` (dict return
    types) or wraps scalar/list returns as ``{"result": ...}``. Fall back to
    JSON-decoding the first text content block.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        if set(structured.keys()) == {"result"}:
            return structured["result"]
        return structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                return text
    return None


async def _call_remote(tool: str, args: Dict[str, Any]) -> Any:
    """Invoke a tool on the networked MCP server via streamable-http."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    endpoint = _endpoint()
    async with streamablehttp_client(endpoint) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, args)
            return _parse_tool_result(result)


# --------------------------------------------------------------------------- #
# Public call surface (mirrors mcp_server tools)
# --------------------------------------------------------------------------- #
async def search_record(
    row: Dict[str, Any],
    agents: Optional[List[str]] = None,
    use_multi_query: bool = True,
) -> List[Dict[str, Any]]:
    if _endpoint():
        return await _call_remote(
            "search_record",
            {"row": row, "agents": agents, "use_multi_query": use_multi_query},
        )
    from .mcp_server import _search_record

    return await _search_record(row, agents=agents, use_multi_query=use_multi_query)


async def route_record(row: Dict[str, Any], routing_config: Optional[Dict[str, Any]] = None) -> List[str]:
    if _endpoint():
        return await _call_remote("route_record", {"row": row, "routing_config": routing_config})
    from .mcp_server import _route_record

    return _route_record(row, routing_config)


async def verify_record(candidates: List[Dict[str, Any]], row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if _endpoint():
        return await _call_remote("verify_record", {"candidates": candidates, "row": row})
    from .mcp_server import _verify_record

    return _verify_record(candidates, row)
