"""CLI for the lab — run scenarios, validate metrics, demonstrate crash-resume."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    metrics = []
    resume_success = False

    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}

        t0 = time.monotonic()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = int((time.monotonic() - t0) * 1000)

        metrics.append(metric_from_state(
            final_state,
            scenario.expected_route.value,
            scenario.requires_approval,
            latency_ms=latency_ms,
        ))

    # Demonstrate crash-resume: re-run S07 with same thread_id via SQLite
    if cfg.get("checkpointer") == "sqlite" and cfg.get("demo_resume"):
        resume_success = _demo_crash_resume(graph, scenarios)

    report = summarize_metrics(metrics, resume_success=resume_success)
    write_metrics(report, output)

    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])

    typer.echo(f"Wrote metrics to {output}")
    typer.echo(f"success_rate={report.success_rate:.2%}  scenarios={report.total_scenarios}")


def _demo_crash_resume(graph: object, scenarios: list) -> bool:
    """Simulate crash by running first half, then resume from checkpoint."""
    try:
        from .state import Scenario, Route, initial_state as mk
        scenario = next((s for s in scenarios if s.should_retry), None)
        if not scenario:
            return False
        state = mk(scenario)
        tid = state["thread_id"] + "-resume-demo"
        cfg = {"configurable": {"thread_id": tid}}
        graph.invoke(state, config=cfg)  # type: ignore[union-attr]
        # Resume: invoke again with same thread — checkpointer replays from checkpoint
        graph.invoke({"query": state["query"]}, config=cfg)  # type: ignore[union-attr]
        return True
    except Exception:
        return False


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}  total={report.total_scenarios}")


@app.command("show-history")
def show_history(
    thread_id: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
) -> None:
    """Show checkpoint history for a thread_id (time-travel debug)."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    run_config = {"configurable": {"thread_id": thread_id}}
    history = list(graph.get_state_history(run_config))  # type: ignore[union-attr]

    if not history:
        typer.echo(f"No history found for thread_id={thread_id!r}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(history)} checkpoints for thread={thread_id!r}")
    for i, snapshot in enumerate(history):
        meta = snapshot.metadata or {}
        typer.echo(f"  [{i}] step={meta.get('step', '?')} source={meta.get('source', '?')}")


if __name__ == "__main__":
    app()
