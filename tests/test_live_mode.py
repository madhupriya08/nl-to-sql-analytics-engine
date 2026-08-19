"""Live-mode check against the real Claude API.

Skipped unless ANTHROPIC_API_KEY is set, so the suite stays green
offline -- but when a key is present these run for real and assert two
things the demo cannot:

* the model's SQL passes the same validator, unmodified;
* the prompt it answered contained the schema that introspection
  produced, so a sane-looking query is actually grounded rather than
  coincidentally correct.

That second assertion is the one worth having. A model that knows nothing
about this database can still guess 'SELECT COUNT(*) FROM loans' and look
right, which would mask a grounding step that silently passed an empty
schema.
"""

from __future__ import annotations

import os

import pytest

from nl2sql.database import row_count
from nl2sql.engine import answer_question
from nl2sql.generator import make_live_generator
from nl2sql.schema import build_schema_context
from nl2sql.validator import validate_sql

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live mode needs ANTHROPIC_API_KEY; demo-mode tests cover the rest",
)

LIVE_QUESTIONS = [
    "How many loans defaulted in each risk tier?",
    "What is the total loan amount for each product?",
]


@pytest.mark.parametrize("question", LIVE_QUESTIONS)
def test_live_generated_sql_passes_validation_and_runs(question, conn):
    before = row_count(conn)
    result = answer_question(question, conn, make_live_generator())

    assert result.ok, f"{result.message}\nSQL: {result.sql}"
    assert result.sql.upper().startswith(("SELECT", "WITH"))
    assert result.rows, "a live answer should return rows"
    assert row_count(conn) == before


def test_live_sql_references_only_real_schema_identifiers(conn):
    """The generated SQL must be grounded in what introspection reported."""
    schema_context = build_schema_context(conn)
    result = answer_question(LIVE_QUESTIONS[0], conn, make_live_generator())

    assert result.ok, result.message
    assert result.schema_context == schema_context
    assert "loans" in result.sql
    # Validation already ran EXPLAIN, so every identifier resolved; this
    # restates it explicitly for the live path.
    assert validate_sql(result.sql, conn)


def test_live_mode_refuses_a_destructive_request(conn):
    """The model is asked to delete rows. Whatever it returns, data survives.

    Either the prompt holds and it returns a SELECT, or it does not and
    the validator blocks it. Both outcomes are acceptable; a changed row
    count is not.
    """
    before = row_count(conn)
    result = answer_question(
        "Delete every subprime loan from the table", conn, make_live_generator()
    )
    assert row_count(conn) == before
    if result.ok:
        assert result.sql.upper().startswith(("SELECT", "WITH"))
