"""LangGraph self-healing orchestration for the MDM pipeline.

This module wraps the existing agents (planner routing, multi-agent search,
consensus verifier + non-match classification) in an explicit LangGraph state
machine with a self-healing control loop:

    START -> router_agent -> search_agent -> verifier_agent --(confirmed)--> END
                  ^                                            |
                  |                                            |
                  +------------------ healer_agent <--(retry)--+

Each node appends a structured record to ``state["trace"]`` so every decision is
explainable/auditable.

Public API:
    - ``run_row(row)``      : async, runs one record through the graph.
    - ``run_row_sync(row)`` : sync wrapper around ``run_row``.

The search step optionally delegates to an MCP server when ``MDM_USE_MCP=1`` or
``MDM_MCP_ENDPOINT`` is set (see ``mcp_client``); otherwise it calls the search
functions directly.
"""
from __future__ import annotations

import os
import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from .planner import decide_agents_for_row
from .search_agents import run_search_agents
from .verifier import verify_candidates

logger = logging.getLogger("mdm.graph")

# All agents the healer can escalate to when a cheap first pass fails.
ALL_AGENTS = ["serpapi", "google", "tavily", "openai", "registry"]


def _max_heal_attempts() -> int:
    try:
        return max(0, int(os.getenv("MDM_MAX_HEAL_ATTEMPTS", "2")))
    except (TypeError, ValueError):
        return 2


class MDMState(TypedDict, total=False):
    """State threaded through the graph for a single record."""

    row: Dict[str, Any]
    routing_config: Optional[Dict[str, Any]]
    agents: List[str]
    use_multi_query: bool
    candidates: List[Dict[str, Any]]
    verified: Dict[str, Any]
    confirmed: bool
    category: Optional[str]
    attempts: int
    trace: List[Dict[str, Any]]


def _trace(state: MDMState, node: str, decision: str, **detail: Any) -> List[Dict[str, Any]]:
    """Return a new trace list with one entry appended (keeps nodes pure-ish)."""
    entry = {"node": node, "decision": decision, "attempt": state.get("attempts", 0)}
    if detail:
        entry.update(detail)
    return list(state.get("trace", [])) + [entry]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def router_agent(state: MDMState) -> Dict[str, Any]:
    """Pick which agents to run for this record.

    First pass: cheap, heuristic routing via ``decide_agents_for_row``.
    Heal retries (attempts > 0): widen to the full agent set and force
    multi-query for higher recall.
    """
    row = state["row"]
    attempts = state.get("attempts", 0)

    if attempts > 0:
        agents = list(ALL_AGENTS)
        use_multi_query = True
        why = f"heal retry #{attempts}: widened to all agents + multi-query"
    else:
        agents = decide_agents_for_row(row, state.get("routing_config"))
        use_multi_query = True
        why = "heuristic routing from row fields"

    logger.info("router_agent: agents=%s (attempt=%s)", agents, attempts)
    return {
        "agents": agents,
        "use_multi_query": use_multi_query,
        "trace": _trace(state, "router_agent", why, agents=agents),
    }


async def search_agent(state: MDMState) -> Dict[str, Any]:
    """Fan out across the selected search agents and collect candidates.

    Delegates to an MCP server when configured, else calls the search
    functions in-process.
    """
    row = state["row"]
    agents = state.get("agents") or list(ALL_AGENTS)
    use_multi_query = state.get("use_multi_query", True)

    candidates = await _search(row, agents, use_multi_query)

    sources = sorted({c.get("source", c.get("agent", "?")) for c in candidates})
    logger.info("search_agent: %d candidates from %s", len(candidates), sources)
    return {
        "candidates": candidates,
        "trace": _trace(
            state,
            "search_agent",
            f"gathered {len(candidates)} candidates",
            agents=agents,
            multi_query=use_multi_query,
            sources=sources,
        ),
    }


async def _search(row: Dict[str, Any], agents: List[str], use_multi_query: bool) -> List[Dict[str, Any]]:
    """Route the search either through MCP (if enabled) or in-process."""
    if os.getenv("MDM_USE_MCP") == "1" or os.getenv("MDM_MCP_ENDPOINT"):
        try:
            from .mcp_client import search_record  # local import; Phase 2 artifact

            return await search_record(row, agents=agents, use_multi_query=use_multi_query)
        except Exception as exc:  # pragma: no cover - fallback path
            logger.warning("MCP search failed (%s); falling back to in-process", exc)
    return await run_search_agents(row, agents=agents, use_multi_query=use_multi_query)


def verifier_agent(state: MDMState) -> Dict[str, Any]:
    """Reconcile candidates into a verified record + presence/classification."""
    row = state["row"]
    candidates = state.get("candidates", [])
    verified = verify_candidates(candidates, row=row)
    confirmed = bool(verified.get("presence_confirmed"))
    category = verified.get("non_match_reason")

    decision = "confirmed" if confirmed else f"not confirmed ({category})"
    logger.info("verifier_agent: %s", decision)
    return {
        "verified": verified,
        "confirmed": confirmed,
        "category": category,
        "trace": _trace(
            state,
            "verifier_agent",
            decision,
            presence_source=verified.get("presence_source") or None,
        ),
    }


def healer_agent(state: MDMState) -> Dict[str, Any]:
    """Self-healing step: bump the attempt counter before retrying."""
    attempts = state.get("attempts", 0) + 1
    logger.info("healer_agent: escalating, attempt=%s", attempts)
    return {
        "attempts": attempts,
        "trace": _trace(
            {**state, "attempts": attempts},
            "healer_agent",
            "escalating routing and retrying search",
        ),
    }


def _route_after_verify(state: MDMState) -> str:
    """Conditional edge: end on success or exhaustion, else heal and retry."""
    if state.get("confirmed"):
        return END
    if state.get("attempts", 0) >= _max_heal_attempts():
        return END
    return "healer_agent"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph():
    """Construct and compile the MDM self-healing state graph."""
    g = StateGraph(MDMState)
    g.add_node("router_agent", router_agent)
    g.add_node("search_agent", search_agent)
    g.add_node("verifier_agent", verifier_agent)
    g.add_node("healer_agent", healer_agent)

    g.add_edge(START, "router_agent")
    g.add_edge("router_agent", "search_agent")
    g.add_edge("search_agent", "verifier_agent")
    g.add_conditional_edges(
        "verifier_agent",
        _route_after_verify,
        {"healer_agent": "healer_agent", END: END},
    )
    # After healing, re-route (router widens agents) then search again.
    g.add_edge("healer_agent", "router_agent")
    return g.compile()


# Compile once at import; graph is stateless/reusable across records.
_GRAPH = build_graph()


async def run_row(row: Dict[str, Any], routing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a single record through the self-healing graph (async).

    Returns the final state dict (row, verified, confirmed, category, trace, ...).
    """
    initial: MDMState = {
        "row": row,
        "routing_config": routing_config,
        "attempts": 0,
        "trace": [],
    }
    return await _GRAPH.ainvoke(initial)


def run_row_sync(row: Dict[str, Any], routing_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Synchronous convenience wrapper around :func:`run_row`."""
    return asyncio.run(run_row(row, routing_config=routing_config))
