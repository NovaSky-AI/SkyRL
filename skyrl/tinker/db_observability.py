"""Low-volume diagnostics for Tinker database contention."""

from time import monotonic
from typing import Any

from sqlalchemy import event

from skyrl.utils.log import logger

SLOW_DATABASE_SECONDS = 1.0


def database_pool_status(engine: Any) -> str:
    """Return pool occupancy without exposing database connection details."""
    if engine is None:
        return "unavailable"
    return engine.sync_engine.pool.status()


def enable_database_observability(engine: Any) -> None:
    """Log slow statements and statement failures for an async engine."""
    sync_engine = engine.sync_engine
    if getattr(sync_engine, "_skyrl_database_observability_enabled", False):
        return
    sync_engine._skyrl_database_observability_enabled = True

    @event.listens_for(sync_engine, "before_cursor_execute")
    def record_query_start(connection, cursor, statement, parameters, context, executemany):
        connection.info.setdefault("skyrl_query_started_at", []).append(monotonic())

    @event.listens_for(sync_engine, "after_cursor_execute")
    def log_slow_query(connection, cursor, statement, parameters, context, executemany):
        started = connection.info["skyrl_query_started_at"].pop()
        elapsed = monotonic() - started
        if elapsed < SLOW_DATABASE_SECONDS:
            return
        logger.warning(
            "Slow database statement operation=%s dialect=%s elapsed_seconds=%.3f pool=%s",
            _statement_operation(statement),
            sync_engine.dialect.name,
            elapsed,
            sync_engine.pool.status(),
        )

    @event.listens_for(sync_engine, "handle_error")
    def log_query_failure(exception_context):
        connection = exception_context.connection
        started_stack = connection.info.get("skyrl_query_started_at", []) if connection is not None else []
        elapsed = monotonic() - started_stack.pop() if started_stack else None
        logger.error(
            "Database statement failed operation=%s dialect=%s elapsed_seconds=%s pool=%s error_type=%s",
            _statement_operation(exception_context.statement),
            sync_engine.dialect.name,
            f"{elapsed:.3f}" if elapsed is not None else "unknown",
            sync_engine.pool.status(),
            type(exception_context.original_exception).__name__,
        )


def _statement_operation(statement: str | None) -> str:
    if statement is None:
        return "unknown"
    words = statement.lstrip().split(None, 1)
    return words[0].upper() if words else "unknown"
