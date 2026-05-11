# Day 08 — LangGraph Agent Orchestration Lab

Production-quality LangGraph agent with typed state, conditional routing, bounded retry loops, HITL approval, and SQLite persistence.

## Quick start

```bash
pip install -e '.[dev]'
make test           # 50 tests, all pass
make run-scenarios  # writes outputs/metrics.json
make grade-local    # validates metrics (success_rate=100%)
```

## Graph overview

```
START → intake → classify → simple/tool/missing_info/risky/error
  simple       → answer → finalize → END
  tool         → tool → evaluate → answer | retry → tool (loop) | dead_letter
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → tool → ... | clarify
  error        → retry → tool → evaluate → ... | dead_letter → finalize → END
```

See `docs/graph_diagram.md` for full Mermaid diagram.

## Scenarios (7 total, 6+ required)

| ID | Route | Tags |
|----|-------|------|
| S01_simple | simple | simple |
| S02_tool | tool | tool |
| S03_missing | missing_info | clarification |
| S04_risky | risky | hitl, risky |
| S05_error | error | retry |
| S06_delete | risky | hitl, destructive |
| S07_dead_letter | error | dead_letter, retry |

## Persistence

Default: `MemorySaver` (in-memory). For SQLite crash-resume:

```yaml
# configs/lab.yaml
checkpointer: sqlite
database_url: outputs/checkpoints.db
```

Time-travel history:
```bash
agent-lab show-history thread-S01_simple --config configs/lab.yaml
```

## HITL approval

Set `LANGGRAPH_INTERRUPT=true` to use real `interrupt()` instead of mock:

```bash
LANGGRAPH_INTERRUPT=true agent-lab run-scenarios --config configs/lab.yaml --output outputs/metrics.json
```

## Environment

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY if using real LLM classify
```

## Lint & type check

```bash
make lint       # ruff check
make typecheck  # mypy
```
