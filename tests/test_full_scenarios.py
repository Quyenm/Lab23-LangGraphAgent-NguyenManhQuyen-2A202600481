"""Full end-to-end scenario tests via graph invocation."""

from __future__ import annotations

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state


@pytest.fixture(scope="module")
def graph():
    return build_graph(checkpointer=build_checkpointer("memory"))


def run(graph, query: str, route: Route, max_attempts: int = 3):
    scenario = Scenario(id="test", query=query, expected_route=route, max_attempts=max_attempts)
    state = initial_state(scenario)
    return graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})


def test_simple_route(graph):
    result = run(graph, "How do I reset my password?", Route.SIMPLE)
    assert result["route"] == Route.SIMPLE.value
    assert result["final_answer"] is not None
    assert len(result["events"]) >= 4


def test_tool_route(graph):
    result = run(graph, "lookup order status 12345", Route.TOOL)
    assert result["route"] == Route.TOOL.value
    assert len(result["tool_results"]) >= 1
    assert result["final_answer"] is not None


def test_missing_info_route(graph):
    result = run(graph, "Can you fix it?", Route.MISSING_INFO)
    assert result["route"] == Route.MISSING_INFO.value
    assert result["pending_question"] is not None


def test_risky_route_with_approval(graph):
    result = run(graph, "Refund this customer", Route.RISKY)
    assert result["route"] == Route.RISKY.value
    assert result["approval"] is not None
    assert result["approval"]["approved"] is True
    approval_events = [e for e in result["events"] if e["node"] == "approval"]
    assert len(approval_events) == 1


def test_error_route_retries_then_succeeds(graph):
    result = run(graph, "Timeout failure on request", Route.ERROR)
    assert result["route"] == Route.ERROR.value
    retry_events = [e for e in result["events"] if e["node"] == "retry"]
    assert len(retry_events) >= 1
    assert result["final_answer"] is not None


def test_dead_letter_on_max_attempts(graph):
    result = run(graph, "System failure cannot recover", Route.ERROR, max_attempts=1)
    assert result["route"] == Route.ERROR.value
    dead_events = [e for e in result["events"] if e["node"] == "dead_letter"]
    assert len(dead_events) == 1
    assert "manual review" in result["final_answer"].lower()


def test_all_paths_produce_final_answer_or_question(graph):
    """Invariant: every run produces final_answer OR pending_question (never both None)."""
    cases = [
        ("How do I reset my password?", Route.SIMPLE),
        ("lookup order status 123", Route.TOOL),
        ("Can you fix it?", Route.MISSING_INFO),
        ("Refund this customer", Route.RISKY),
        ("Timeout failure on request", Route.ERROR),
    ]
    for query, route in cases:
        result = run(graph, query, route)
        has_output = bool(result.get("final_answer") or result.get("pending_question"))
        assert has_output, f"No output for {query!r} → route={route.value}"


def test_events_are_append_only(graph):
    """Verify events accumulate (append reducer) across multiple nodes."""
    result = run(graph, "lookup order 123", Route.TOOL)
    node_names = [e["node"] for e in result["events"]]
    # Must visit all these nodes in order
    assert "intake" in node_names
    assert "classify" in node_names
    assert "tool" in node_names
    assert "evaluate" in node_names
    assert "finalize" in node_names


def test_load_scenarios_minimum_six():
    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    assert len(scenarios) >= 6


def test_all_sample_scenarios_run(graph):
    scenarios = load_scenarios("data/sample/scenarios.jsonl")
    for scenario in scenarios:
        state = initial_state(scenario)
        result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        assert result["route"] == scenario.expected_route.value, (
            f"Scenario {scenario.id}: expected {scenario.expected_route.value}, got {result['route']}"
        )
