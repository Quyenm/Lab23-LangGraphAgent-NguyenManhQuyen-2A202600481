"""Report generation — fills in lab_report.md from metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report markdown from metrics."""
    scenario_rows = "\n".join(
        f"| {m.scenario_id} | {m.expected_route} | {m.actual_route} | "
        f"{'✅' if m.success else '❌'} | {m.retry_count} | "
        f"{m.interrupt_count} | {m.latency_ms}ms |"
        for m in metrics.scenario_metrics
    )

    error_scenarios = [m for m in metrics.scenario_metrics if m.retry_count > 0]
    approval_scenarios = [m for m in metrics.scenario_metrics if m.approval_required]

    retry_text = (
        f"Scenarios demonstrating retry: {', '.join(m.scenario_id for m in error_scenarios)}"
        if error_scenarios
        else "No retry scenarios."
    )
    approval_text = (
        f"Scenarios requiring approval: {', '.join(m.scenario_id for m in approval_scenarios)}"
        if approval_scenarios
        else "No risky scenarios."
    )

    return f"""# Day 08 Lab Report — LangGraph Agent Orchestration

## 1. Student / Team

- **Name**: Nguyễn Mạnh Quyền
- **Student ID**: 2A202600481
- **Lab**: AICB Day 08, Phase 2 Track 3
- **Date**: {date.today().isoformat()}

---

## 2. Architecture

The agent uses a **LangGraph StateGraph** with 11 nodes wired into 6 distinct routes:

```
START → intake → classify → route_after_classify
  simple       → answer → finalize → END
  tool         → tool → evaluate → route_after_evaluate
                  needs_retry → retry → route_after_retry
                    attempt < max → tool (loop)
                    attempt >= max → dead_letter → finalize → END
                  success → answer → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → route_after_approval
                  approved → tool → evaluate → answer → finalize → END
                  rejected → clarify → finalize → END
  error        → retry → tool → evaluate → ... (bounded by max_attempts)
```

### Node responsibilities

| Node | Responsibility |
|------|---------------|
| intake | Normalize query, strip/truncate, emit audit event |
| classify | Route to simple/tool/missing_info/risky/error via keyword heuristics |
| answer | Ground response in tool_results + approval; safe fallback |
| tool | Mock tool call; simulates transient failure for error-route retry demo |
| evaluate | Check latest tool result — "done?" gate for retry loop |
| clarify | Generate targeted clarification question |
| risky_action | Prepare proposed action with risk justification |
| approval | HITL: mock approval in CI; real interrupt() when LANGGRAPH_INTERRUPT=true |
| retry | Increment attempt counter, record error |
| dead_letter | Escalate unresolvable failure, log for manual review |
| finalize | Emit final audit event |

---

## 3. State Schema

| Field | Reducer | Why |
|-------|---------|-----|
| thread_id | overwrite | immutable run identity |
| scenario_id | overwrite | immutable input |
| query | overwrite | normalized once by intake |
| route | overwrite | only current classification needed |
| risk_level | overwrite | only current risk level needed |
| attempt | overwrite | retry counter, latest value drives routing |
| max_attempts | overwrite | configuration, set at start |
| final_answer | overwrite | latest answer supersedes previous |
| pending_question | overwrite | latest clarification question |
| proposed_action | overwrite | single approval workflow per run |
| approval | overwrite | single approval decision |
| evaluation_result | overwrite | latest eval decision drives routing |
| messages | **append** (add) | full conversation audit trail |
| tool_results | **append** (add) | all tool calls preserved for grounding |
| errors | **append** (add) | all errors preserved for failure analysis |
| events | **append** (add) | chronological audit log, enables time-travel |

**Key design insight**: `evaluation_result` is overwrite (not append) because only the
*latest* evaluation drives the routing decision. Appending would require routing to
inspect list position, which is fragile.

---

## 4. Scenario Results

| Scenario | Expected Route | Actual Route | Success | Retries | Interrupts | Latency |
|----------|---------------|--------------|---------|---------|------------|---------|
{scenario_rows}

**Summary**:
- Total scenarios: **{metrics.total_scenarios}**
- Success rate: **{metrics.success_rate:.1%}**
- Average nodes visited: **{metrics.avg_nodes_visited:.1f}**
- Total retries across all runs: **{metrics.total_retries}**
- Total approval/HITL events: **{metrics.total_interrupts}**
- Crash-resume demonstrated: **{metrics.resume_success}**

---

## 5. Failure Analysis

### 5.1 Transient tool failure (retry loop)

{retry_text}

The `tool_node` simulates a transient failure when `route == error AND attempt < 2`.
The retry loop is: `tool → evaluate → retry → tool` and is bounded by `max_attempts`.
After `attempt >= max_attempts`, `route_after_retry` returns `dead_letter` instead of
`tool`, guaranteeing the loop terminates.

**Key safety property**: the retry counter (`attempt`) is incremented in `retry_node`
*before* `route_after_retry` checks it, so the bound is always enforced correctly.

### 5.2 Risky action without approval

{approval_text}

`risky_action_node` prepares the action and `approval_node` blocks execution until
approved. In mock mode (CI), auto-approved. With `LANGGRAPH_INTERRUPT=true`, the graph
suspends at the interrupt, waits for external input, then resumes via `Command`.

**Failure mode**: if `approval_node` returns `approved=False`, the graph routes to
`clarify` — user gets an explanation. No destructive action is taken.

### 5.3 Missing information

Triggered when query is short (<6 words) with an ambiguous pronoun ("it", "this").
`ask_clarification_node` generates a targeted question. This prevents hallucinated
responses when context is insufficient.

### 5.4 Dead-letter escalation

When `attempt >= max_attempts`, `dead_letter_node` sets `final_answer` to an escalation
message and the run exits cleanly. In production, this would persist to an SQS DLQ and
page on-call.

---

## 6. Persistence / Recovery Evidence

The graph uses **MemorySaver** by default (state survives within process).

For SQLite persistence, set `checkpointer: sqlite` in `configs/lab.yaml`. This writes
checkpoints to `outputs/checkpoints.db` and enables:

1. **Crash-resume**: rerun `agent-lab run-scenarios` with the same `thread_id` after
   a simulated crash — the graph resumes from the last checkpoint.
2. **Time-travel debug**: `agent-lab show-history <thread_id>` lists all checkpoints
   and allows replaying from any step with `get_state_history()`.

Each run uses a unique `thread_id = "thread-{{scenario_id}}"` to isolate state.

---

## 7. Extension Work

- **SQLite persistence**: `build_checkpointer(kind="sqlite")` is implemented and tested.
- **Time-travel history**: `agent-lab show-history <thread_id>` command added to CLI.
- **Graph diagram**: see `docs/graph_diagram.md` for ASCII flow + Mermaid diagram.
- **Crash-resume demo**: `cli.py::_demo_crash_resume()` demonstrates same thread_id resuming.

---

## 8. Improvement Plan

If given one more day, priority order:

1. **LLM-as-judge in evaluate_node** — replace ERROR string heuristic with a structured
   LLM call to validate tool output schema and semantic correctness.
2. **Real HITL with Streamlit UI** — build a minimal approval dashboard that receives
   the `interrupt` payload and sends back `Command(resume=...)`.
3. **Observability** — add OpenTelemetry traces so each node emits spans; integrate with
   LangSmith for production monitoring.
4. **Postgres persistence** — swap SQLite for Postgres for horizontal scaling.
5. **Parallel fan-out** — run two mock tools in parallel with `Send` API, merge evidence
   in evaluate_node for richer grounding.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
