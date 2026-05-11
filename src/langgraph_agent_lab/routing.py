"""Routing functions for conditional edges.

Each function receives AgentState and returns a node name string.
All functions are pure (no side effects) and exhaustive (no silent fallthrough).
"""

from __future__ import annotations

from .state import AgentState, Route


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Unknown routes default to 'answer' (safe degradation — never crash).
    """
    route = state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",  # go directly to retry/tool cycle
    }
    return mapping.get(route, "answer")  # safe default


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry tool or dead-letter after retry node.

    Bounded retry: if attempt >= max_attempts, escalate to dead_letter.
    This ensures the retry loop is always finite (no infinite loops).
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if attempt >= max_attempts:
        return "dead_letter"
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether tool result is satisfactory or needs retry.

    This is the 'done?' check — a key LangGraph advantage over LCEL chains.
    Only two outcomes: retry (loop back) or answer (proceed to response).
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_approval(state: AgentState) -> str:
    """Continue to tool execution only if action was approved.

    If rejected/missing: route to clarify to explain the decision to the user.
    Production extension: add 'edit' outcome that loops back to risky_action.
    """
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    return "clarify"  # rejection → ask for clarification / inform user
