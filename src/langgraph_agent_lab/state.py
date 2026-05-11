"""State schema for the Day 08 LangGraph lab.

Design decisions:
- Append-only fields (messages, tool_results, errors, events) use operator.add reducer
  so every node's output accumulates a full audit trail — enabling time-travel debug
  and crash-resume without losing history.
- Scalar fields (route, attempt, evaluation_result, etc.) are overwrite — only the
  latest value matters and accumulating them would bloat the checkpoint.
- State is kept fully JSON-serializable for SQLite/Postgres persistence.
"""

from __future__ import annotations

from enum import StrEnum
from operator import add
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field, field_validator


class Route(StrEnum):
    SIMPLE = "simple"
    TOOL = "tool"
    MISSING_INFO = "missing_info"
    RISKY = "risky"
    ERROR = "error"
    DEAD_LETTER = "dead_letter"
    DONE = "done"


class LabEvent(BaseModel):
    """Append-only audit event for grading and debugging."""

    node: str
    event_type: str
    message: str
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool = False
    reviewer: str = "mock-reviewer"
    comment: str = ""


class AgentState(TypedDict, total=False):
    """LangGraph state.

    Append-only (reducer=add): messages, tool_results, errors, events
      -> These accumulate for audit trail and support time-travel replay.
    Overwrite: route, risk_level, attempt, evaluation_result, final_answer,
               pending_question, proposed_action, approval
      -> Only the current/latest value is needed for routing decisions.
    """

    # Immutable context
    thread_id: str
    scenario_id: str
    query: str

    # Routing & classification (overwrite)
    route: str
    risk_level: str

    # Retry counter (overwrite)
    attempt: int
    max_attempts: int

    # Output fields (overwrite — latest wins)
    final_answer: str | None
    pending_question: str | None
    proposed_action: str | None
    approval: dict[str, Any] | None
    evaluation_result: str | None  # "success" | "needs_retry"

    # Append-only audit trail (reducer=add)
    messages: Annotated[list[str], add]
    tool_results: Annotated[list[str], add]
    errors: Annotated[list[str], add]
    events: Annotated[list[dict[str, Any]], add]


class Scenario(BaseModel):
    id: str
    query: str
    expected_route: Route
    requires_approval: bool = False
    should_retry: bool = False
    max_attempts: int = 3
    tags: list[str] = Field(default_factory=list)

    @field_validator("query")
    @classmethod
    def query_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty")
        return value


def initial_state(scenario: Scenario) -> AgentState:
    """Create a serializable initial state for one scenario."""
    return {
        "thread_id": f"thread-{scenario.id}",
        "scenario_id": scenario.id,
        "query": scenario.query,
        "route": "",
        "risk_level": "unknown",
        "attempt": 0,
        "max_attempts": scenario.max_attempts,
        "final_answer": None,
        "pending_question": None,
        "proposed_action": None,
        "approval": None,
        "evaluation_result": None,
        "messages": [],
        "tool_results": [],
        "errors": [],
        "events": [],
    }


def make_event(node: str, event_type: str, message: str, **metadata: object) -> dict[str, Any]:
    """Create a normalized event payload."""
    return LabEvent(
        node=node,
        event_type=event_type,
        message=message,
        latency_ms=0,
        metadata=metadata,
    ).model_dump()
