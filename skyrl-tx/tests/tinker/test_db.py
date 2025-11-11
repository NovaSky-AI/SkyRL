import os
from pathlib import Path
import subprocess
import tempfile


ALEMBIC_CMD_PREFIX = ["uv", "run", "--extra", "dev"]


def test_alembic_migration_generation():
    """Test that Alembic can generate migrations from SQLModel definitions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db_path = Path(tmpdir) / "test_alembic.db"
        test_db_url = f"sqlite:///{test_db_path}"

        tinker_dir = Path(__file__).parent.parent.parent / "tx" / "tinker"

        # Test: alembic upgrade head creates tables
        result = subprocess.run(
            ALEMBIC_CMD_PREFIX + ["alembic", "upgrade", "head"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "TX_DATABASE_URL": test_db_url},
        )

        # Should succeed (even if no migrations exist, it shouldn't error)
        assert result.returncode == 0, f"Alembic upgrade failed: {result.stderr}"

        # Test: alembic current shows version
        result = subprocess.run(
            ALEMBIC_CMD_PREFIX + ["alembic", "current"],
            cwd=tinker_dir,
            capture_output=True,
            text=True,
            env={**os.environ, "TX_DATABASE_URL": test_db_url},
        )

        assert result.returncode == 0, f"Alembic current failed: {result.stderr}"


def test_alembic_history():
    """Test that Alembic history command works."""
    tinker_dir = Path(__file__).parent.parent.parent / "tx" / "tinker"

    # Test: alembic history
    result = subprocess.run(
        ["uv", "run", "alembic", "history"],
        cwd=tinker_dir,
        capture_output=True,
        text=True,
    )

    # Should work even with no migrations
    assert result.returncode == 0, f"Alembic history failed: {result.stderr}"
