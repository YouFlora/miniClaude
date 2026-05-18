"""SQLite-backed checkpoint storage for LangGraph state.

LangGraph automatically calls checkpointer.put() before / after every
super-step. The thread_id (passed via config={"configurable": {"thread_id": ...}})
becomes the primary key, so different conversations live side by side in the
same database file.

Same abstraction underlies HITL pause/resume (M3) and steering abort/resume (M8).
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


DEFAULT_DB_PATH = Path(__file__).parent / "miniclaude.db"


def open_checkpointer(db_path: str | Path | None = None) -> SqliteSaver:
    """Return a SqliteSaver bound to the given database file."""
    path = Path(db_path or os.getenv("MINICLAUDE_DB", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def list_sessions(db_path: str | Path | None = None) -> list[str]:
    """Return distinct thread_ids found in the checkpoint database."""
    path = Path(db_path or os.getenv("MINICLAUDE_DB", DEFAULT_DB_PATH))
    if not path.exists():
        return []
    with closing(sqlite3.connect(str(path))) as conn:
        try:
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [r[0] for r in rows]
