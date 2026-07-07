import os
import subprocess
import tempfile
from pathlib import Path

from skyrl.tinker import db_models
from skyrl.tinker.db_models import prepare_sqlite_path

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


def test_prepare_sqlite_path_creates_missing_directories(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "tinker.db"
    prepare_sqlite_path(f"sqlite:///{db_path}")
    assert db_path.parent.is_dir()


def test_prepare_sqlite_path_ignores_non_sqlite_and_memory_urls():
    # None of these should raise or touch the filesystem
    prepare_sqlite_path("postgresql://user:password@localhost:5432/tinker")
    prepare_sqlite_path("sqlite:///:memory:")
    prepare_sqlite_path("sqlite://")
    prepare_sqlite_path("sqlite:///ignored.db?mode=memory")
    assert not Path("ignored.db").exists()


def test_prepare_sqlite_path_warns_on_network_filesystem(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(db_models, "_filesystem_type", lambda path: "nfs4")
    monkeypatch.setattr(db_models.logger, "warning", warnings.append)
    prepare_sqlite_path(f"sqlite:///{tmp_path / 'tinker.db'}")
    assert len(warnings) == 1
    assert "nfs4" in warnings[0]
    assert "database is locked" in warnings[0]


def test_prepare_sqlite_path_no_warning_on_local_filesystem(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(db_models, "_filesystem_type", lambda path: "ext4")
    monkeypatch.setattr(db_models.logger, "warning", warnings.append)
    prepare_sqlite_path(f"sqlite:///{tmp_path / 'tinker.db'}")
    assert warnings == []
