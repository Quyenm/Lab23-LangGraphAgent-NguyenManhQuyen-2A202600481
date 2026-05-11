"""Node implementations for the LangGraph workflow.

Each node:
- Accepts AgentState and returns a *partial* dict (no in-place mutation)
- Emits at least one LabEvent for audit trail
- Is independently unit-testable
"""

from __future__ import annotations

import os

from .state import AgentState, ApprovalDecision, Route, make_event


# ---------------------------------------------------------------------------
# Intake & Classification
# ---------------------------------------------------------------------------

def intake_node(state: AgentState) -> dict:
    """Normalize raw query: strip whitespace, truncate to safe length, tag metadata."""
    raw = state.get("query", "")
    query = raw.strip()
    if len(query) > 500:
        query = query[:500]  # safety truncation — no hallucination on giant inputs
    return {
        "query": query,
        "messages": [f"intake: {query[:60]}"],
        "events": [make_event("intake", "completed", "query normalized", query_len=len(query))],
    }


def classify_node(state: AgentState) -> dict:
    """Classify query into a route using keyword heuristics.

    Production extension: replace with an LLM classify call using structured output.
    Routing policy:
    - risky:        destructive/financial actions (refund, delete, send)
    - tool:         data lookups (status, order, lookup, check, find)
    - missing_info: very short query with ambiguous pronoun
    - error:        explicit failure/timeout signals
    - simple:       everything else
    """
    query = state.get("query", "").lower()
    words = set(query.split())
    clean = {w.strip("?!.,;:") for w in words}

    route = Route.SIMPLE
    risk_level = "low"

    risky_keywords = {"refund", "delete", "send", "cancel", "remove", "transfer", "destroy"}
    tool_keywords = {"status", "order", "lookup", "check", "find", "get", "retrieve", "search"}
    error_keywords = {"timeout", "fail", "failure", "error", "crash", "cannot", "broken", "system"}

    if risky_keywords & clean:
        route = Route.RISKY
        risk_level = "high"
    elif error_keywords & clean:
        route = Route.ERROR
        risk_level = "medium"
    elif tool_keywords & clean:
        route = Route.TOOL
        risk_level = "low"
    elif len(clean) < 6 and ("it" in clean or "this" in clean or "that" in clean):
        route = Route.MISSING_INFO
        risk_level = "low"

    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value} risk={risk_level}")],
    }


# ---------------------------------------------------------------------------
# Simple path
# ---------------------------------------------------------------------------

def answer_node(state: AgentState) -> dict:
    """Produce a grounded final response.

    Grounds answer in tool_results when available; otherwise gives a safe generic answer.
    Post-approval: includes approval confirmation.
    """
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval") or {}
    query = state.get("query", "")

    if tool_results:
        last_result = tool_results[-1]
        reviewer = approval.get("reviewer", "reviewer") if approval else None
        if approval and approval.get("approved") and reviewer:
            answer = (
                f"Action approved by {reviewer} "
                f"and executed. Result: {last_result}"
            )
        else:
            answer = f"Based on lookup results: {last_result}"
    else:
        answer = f"Answer to '{query[:60]}': This is a direct response that does not require tool use."

    return {
        "final_answer": answer,
        "messages": [f"answer: {answer[:80]}"],
        "events": [make_event("answer", "completed", "final answer generated", grounded=bool(tool_results))],
    }


# ---------------------------------------------------------------------------
# Tool & Evaluate path
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with idempotent retry semantics.

    Simulates transient failure for error-route scenarios (attempt < 2).
    In production: replace with actual tool executor; use idempotency keys.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    route = state.get("route", "")

    # Simulate transient failure for error route on first two attempts
    if route == Route.ERROR.value and attempt < 2:
        result = f"ERROR:transient:scenario={scenario_id}:attempt={attempt}"
        return {
            "tool_results": [result],
            "errors": [f"tool transient failure attempt={attempt}"],
            "events": [make_event("tool", "error", f"transient failure attempt={attempt}", attempt=attempt)],
        }

    result = f"tool-ok:scenario={scenario_id}:attempt={attempt}:query={state.get('query', '')[:40]}"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", "tool executed successfully", attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result — the 'done?' check enabling retry loops.

    Sets evaluation_result to 'needs_retry' if latest result contains ERROR,
    otherwise 'success'. This is the gatekeeper of the retry loop.

    Production extension: replace with LLM-as-judge or schema validation.
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""

    if "ERROR" in latest.upper():
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "needs_retry", "tool result indicates failure")],
        }

    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "success", "tool result satisfactory")],
    }


# ---------------------------------------------------------------------------
# Retry / Dead-letter
# ---------------------------------------------------------------------------

def retry_or_fallback_node(state: AgentState) -> dict:
    """Increment attempt counter and record retry event.

    Bounded by max_attempts — route_after_retry checks the counter before calling
    this node again, so the loop is always finite.
    """
    attempt = int(state.get("attempt", 0)) + 1
    return {
        "attempt": attempt,
        "errors": [f"retry:attempt={attempt}"],
        "events": [make_event("retry", "completed", f"retry attempt {attempt}", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Escalate unresolvable failure to dead-letter queue.

    Third tier of error strategy: tool -> evaluate -> retry -> dead_letter.
    Production: persist to SQS dead-letter, alert on-call, create support ticket.
    """
    attempt = state.get("attempt", 0)
    scenario_id = state.get("scenario_id", "unknown")
    return {
        "final_answer": (
            f"Request '{scenario_id}' could not be completed after {attempt} attempts. "
            "Escalated for manual review."
        ),
        "errors": [f"dead_letter:max_retries_exceeded:attempt={attempt}"],
        "events": [make_event(
            "dead_letter", "escalated",
            f"max retries exceeded after {attempt} attempts",
            scenario_id=scenario_id,
        )],
    }


# ---------------------------------------------------------------------------
# Clarification
# ---------------------------------------------------------------------------

def ask_clarification_node(state: AgentState) -> dict:
    """Request specific missing information instead of hallucinating.

    Generates a targeted question by detecting what's missing in the query.
    """
    query = state.get("query", "").lower()
    if "it" in query.split() or "this" in query.split():
        question = "Could you clarify what 'it' refers to? Please provide the order ID, account number, or specific item."
    else:
        question = "Could you provide more context? For example: order ID, account number, or the specific action you need."

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification question sent")],
    }


# ---------------------------------------------------------------------------
# Risky action / HITL approval
# ---------------------------------------------------------------------------

def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action with full justification for approval.

    Captures: proposed action, risk level, evidence, and requestor query.
    This payload is surfaced to the human reviewer.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    return {
        "proposed_action": (
            f"PROPOSED: Execute risky action for query='{query[:80]}' | "
            f"risk={risk_level} | requires human approval before proceeding"
        ),
        "events": [make_event(
            "risky_action", "pending_approval",
            "risky action prepared, awaiting approval",
            risk_level=risk_level,
        )],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for Streamlit/UI demos.
    Default uses mock approval so all tests and CI run offline without input.

    Production extension: add reject/edit decisions and timeout escalation.
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt  # type: ignore[import-untyped]

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
            "query": state.get("query"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        elif isinstance(value, bool):
            decision = ApprovalDecision(approved=value, comment="interrupt decision")
        else:
            decision = ApprovalDecision(approved=True, comment="interrupt: auto-approved")
    else:
        # Mock approval for CI/offline testing — always approved
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="mock approval for lab demo",
        )

    return {
        "approval": decision.model_dump(),
        "events": [make_event(
            "approval", "completed",
            f"approval decision: approved={decision.approved}",
            approved=decision.approved,
            reviewer=decision.reviewer,
        )],
    }


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def finalize_node(state: AgentState) -> dict:
    """Emit final audit event. Marks workflow as complete."""
    route = state.get("route", "unknown")
    attempt = state.get("attempt", 0)
    has_answer = bool(state.get("final_answer") or state.get("pending_question"))
    return {
        "events": [make_event(
            "finalize", "completed",
            "workflow finished",
            route=route,
            attempts=attempt,
            has_answer=has_answer,
        )],
    }
