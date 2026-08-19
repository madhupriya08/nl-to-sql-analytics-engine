"""Shared fixtures.

Every test runs against a real SQLite database built from the real
synthetic generator -- not an in-memory stub with two rows. The safety
layer's fourth check compiles SQL against the live schema, so a fake
schema would make the most interesting check untestable.
"""

from __future__ import annotations

import pytest

from nl2sql.database import build_database, connect


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    """A real loans database, built once per test session.

    Session-scoped because building 5,000 rows per test is wasteful, and
    because a *shared* database is the stronger test: if any test ever
    mutated the data, later tests would see it. The row-count assertions
    would then fail, which is exactly the alarm we want.
    """
    path = tmp_path_factory.mktemp("nl2sql") / "loans.db"
    build_database(path, overwrite=True)
    return path


@pytest.fixture
def conn(db_path):
    """A fresh connection per test, pointed at the shared database."""
    connection = connect(db_path)
    yield connection
    connection.close()
