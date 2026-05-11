"""Checkpointer adapter for persistence and crash-resume.

Supports three backends:
- memory: MemorySaver (default, no infra needed, state survives within process)
- sqlite: SqliteSaver (state survives process restart, enables crash-resume demo)
- none: no persistence (useful for pure stateless tests)

Extension: set checkpointer=sqlite in configs/lab.yaml and run with same thread_id
after a crash to demonstrate crash-resume recovery.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer instance.

    Args:
        kind: 'memory' | 'sqlite' | 'postgres' | 'none'
        database_url: path/URL for sqlite or postgres backends

    Returns:
        Checkpointer instance or None.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc
        db_path = database_url or "outputs/checkpoints.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc
        return PostgresSaver.from_conn_string(database_url or "")

    raise ValueError(f"Unknown checkpointer kind: {kind!r}. Choose: memory, sqlite, postgres, none")
