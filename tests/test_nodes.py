"""Comprehensive node unit tests for grading evidence."""

from __future__ import annotations

import pytest

from langgraph_agent_lab.nodes import (
    answer_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    intake_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from langgraph_agent_lab.state import Route

# ---------------------------------------------------------------------------
# intake_node
# ---------------------------------------------------------------------------

def test_intake_normalizes_whitespace():
    result = intake_node({"query": "  hello world  "})
    assert result["query"] == "hello world"


def test_intake_emits_event():
    result = intake_node({"query": "test"})
    assert result["events"][0]["node"] == "intake"


def test_intake_truncates_long_query():
    long_query = "x" * 600
    result = intake_node({"query": long_query})
    assert len(result["query"]) == 500


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("lookup order status for 123", Route.TOOL.value),
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Refund this customer now", Route.RISKY.value),
        ("Delete the account", Route.RISKY.value),
        ("Timeout failure processing", Route.ERROR.value),
        ("System failure cannot recover", Route.ERROR.value),
    ],
)
def test_classify_routes(query: str, expected_route: str):
    result = classify_node({"query": query})
    assert result["route"] == expected_route, f"query={query!r} → expected {expected_route}, got {result['route']}"


def test_classify_risky_sets_high_risk():
    result = classify_node({"query": "send payment now"})
    assert result["risk_level"] == "high"


def test_classify_simple_is_low_risk():
    result = classify_node({"query": "What is the weather?"})
    assert result["risk_level"] == "low"


# ---------------------------------------------------------------------------
# evaluate_node
# ---------------------------------------------------------------------------

def test_evaluate_success_on_clean_result():
    result = evaluate_node({"tool_results": ["tool-ok:scenario=X"]})
    assert result["evaluation_result"] == "success"


def test_evaluate_needs_retry_on_error():
    result = evaluate_node({"tool_results": ["ERROR:transient"]})
    assert result["evaluation_result"] == "needs_retry"


def test_evaluate_empty_results_is_success():
    result = evaluate_node({"tool_results": []})
    assert result["evaluation_result"] == "success"


# ---------------------------------------------------------------------------
# retry_or_fallback_node
# ---------------------------------------------------------------------------

def test_retry_increments_attempt():
    result = retry_or_fallback_node({"attempt": 0})
    assert result["attempt"] == 1


def test_retry_appends_error():
    result = retry_or_fallback_node({"attempt": 1})
    assert len(result["errors"]) == 1
    assert "retry" in result["errors"][0]


# ---------------------------------------------------------------------------
# dead_letter_node
# ---------------------------------------------------------------------------

def test_dead_letter_sets_final_answer():
    result = dead_letter_node({"attempt": 3, "scenario_id": "S07"})
    assert result["final_answer"] is not None
    assert "manual review" in result["final_answer"].lower()


def test_dead_letter_emits_event():
    result = dead_letter_node({"attempt": 3, "scenario_id": "S07"})
    assert result["events"][0]["node"] == "dead_letter"


# ---------------------------------------------------------------------------
# tool_node
# ---------------------------------------------------------------------------

def test_tool_node_succeeds_on_simple_route():
    result = tool_node({"route": "simple", "attempt": 0, "scenario_id": "S01"})
    assert "ERROR" not in result["tool_results"][0]


def test_tool_node_fails_on_error_route_attempt_0():
    result = tool_node({"route": "error", "attempt": 0, "scenario_id": "S05"})
    assert "ERROR" in result["tool_results"][0]


def test_tool_node_succeeds_on_error_route_attempt_2():
    result = tool_node({"route": "error", "attempt": 2, "scenario_id": "S05"})
    assert "ERROR" not in result["tool_results"][0]


# ---------------------------------------------------------------------------
# answer_node
# ---------------------------------------------------------------------------

def test_answer_grounds_in_tool_results():
    result = answer_node({"tool_results": ["data:order=123"], "query": "lookup"})
    assert "data:order=123" in result["final_answer"]


def test_answer_works_without_tool_results():
    result = answer_node({"tool_results": [], "query": "What is 2+2?"})
    assert result["final_answer"] is not None


def test_answer_includes_approval_info():
    result = answer_node({
        "tool_results": ["result"],
        "approval": {"approved": True, "reviewer": "admin", "comment": "ok"},
        "query": "refund",
    })
    assert "admin" in result["final_answer"]


# ---------------------------------------------------------------------------
# ask_clarification_node
# ---------------------------------------------------------------------------

def test_clarify_sets_pending_question():
    result = ask_clarification_node({"query": "Can you fix it?"})
    assert result["pending_question"] is not None
    assert len(result["pending_question"]) > 10


def test_clarify_and_final_answer_match():
    result = ask_clarification_node({"query": "Fix it"})
    assert result["final_answer"] == result["pending_question"]


# ---------------------------------------------------------------------------
# risky_action_node
# ---------------------------------------------------------------------------

def test_risky_action_sets_proposed_action():
    result = risky_action_node({"query": "Refund order 123", "risk_level": "high"})
    assert "approval" in result["proposed_action"].lower()


# ---------------------------------------------------------------------------
# finalize_node
# ---------------------------------------------------------------------------

def test_finalize_emits_event():
    result = finalize_node({"route": "simple", "attempt": 0})
    assert result["events"][0]["node"] == "finalize"
    assert result["events"][0]["event_type"] == "completed"
