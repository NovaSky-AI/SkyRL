from types import SimpleNamespace

import pytest

from skyrl.tinker.db_models import upgrade_request_status_schema


@pytest.mark.asyncio
async def test_postgres_request_status_schema_adds_dispatched_value():
    class RecordingConnection:
        dialect = SimpleNamespace(name="postgresql")

        def __init__(self):
            self.statements = []

        async def execute(self, statement):
            self.statements.append(str(statement))

    connection = RecordingConnection()
    await upgrade_request_status_schema(connection)

    assert connection.statements == ["ALTER TYPE requeststatus ADD VALUE IF NOT EXISTS 'DISPATCHED'"]


@pytest.mark.asyncio
async def test_sqlite_request_status_schema_needs_no_manual_upgrade():
    class RecordingConnection:
        dialect = SimpleNamespace(name="sqlite")

        async def execute(self, statement):
            raise AssertionError("SQLite status columns must not use PostgreSQL DDL")

    await upgrade_request_status_schema(RecordingConnection())
