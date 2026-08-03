import os
import subprocess
import tempfile
from pathlib import Path

ALEMBIC_CMD_PREFIX = ["uv", "run", "--extra", "dev"]


def test_alembic_migration_generation():
    """Test that Alembic can generate migrations from SQLModel definitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_alembic.db"
        test_db_url = f"sqlite:///{test_db_path}"

        tinker_dir = Path(__file__).parent.parent.parent / "skyrl" / "tinker"

        # Test: alembic upgrade head creates tables
        result = subprocess.run(
            ALEMBIC_CMD_PREFIX + ["alembic", "upgrade", "head"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "SKYRL_DATABASE_URL": test_db_url},
        )

        # Should succeed (even if no migrations exist, it shouldn't error)
        assert result.returncode == 0, f"Alembic upgrade failed: {result.stderr}"

        # Test: alembic current shows version
        result = subprocess.run(
            ALEMBIC_CMD_PREFIX + ["alembic", "current"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "SKYRL_DATABASE_URL": test_db_url},
        )

        assert result.returncode == 0, f"Alembic current failed: {result.stderr}"


def test_alembic_history():
    """Test that Alembic history command works."""
    tinker_dir = Path(__file__).parent.parent.parent / "skyrl" / "tinker"

    # Test: alembic history
    result = subprocess.run(
        ["uv", "run", "alembic", "history"],
        cwd=tinker_dir,
        capture_output=True,
        text=True,
    )

    # Should work even with no migrations
    assert result.returncode == 0, f"Alembic history failed: {result.stderr}"


def test_create_missing_indexes_backfills_existing_database(tmp_path):
    """Indexes added after a database was created must still get built.

    ``SQLModel.metadata.create_all`` skips tables it already finds, so without
    this an existing deployment would silently keep running without the engine's
    covering index.
    """
    from sqlalchemy import inspect
    from sqlmodel import SQLModel, create_engine

    from skyrl.tinker.db_models import FutureDB, create_missing_indexes

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    declared = set(FutureDB.__table__.indexes)
    scan_index = next(index for index in declared if index.name == "ix_futures_pending_scan")

    # Create the schema as it looked before the covering index was declared.
    FutureDB.__table__.indexes.discard(scan_index)
    try:
        SQLModel.metadata.create_all(engine)
    finally:
        FutureDB.__table__.indexes.add(scan_index)

    assert "ix_futures_pending_scan" not in {i["name"] for i in inspect(engine).get_indexes("futures")}

    create_missing_indexes(engine)
    assert "ix_futures_pending_scan" in {i["name"] for i in inspect(engine).get_indexes("futures")}

    # Must be safe to run on every startup.
    create_missing_indexes(engine)
    engine.dispose()


def test_pending_scan_uses_covering_index(tmp_path):
    """The scheduling scan must not touch table rows, which carry the payloads."""
    from sqlalchemy import text
    from sqlmodel import SQLModel, create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'plan.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.connect() as conn:
        plan = conn.execute(
            text(
                "EXPLAIN QUERY PLAN SELECT request_id, model_id, request_type "
                "FROM futures WHERE status='pending' ORDER BY request_id"
            )
        ).fetchall()
    engine.dispose()

    assert any("COVERING INDEX ix_futures_pending_scan" in row[-1] for row in plan), plan
