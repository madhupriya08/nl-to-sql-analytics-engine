"""Full-pipeline tests using the demo generator.

These exercise every stage for real -- live schema introspection, all four
validation checks, and actual SQLite execution against the 5,000-row
database. Only the model call is replaced, which is the point of the
pluggable-generator seam.
"""

from __future__ import annotations

import pytest

from nl2sql.database import execute_select, row_count
from nl2sql.demo import DEMO_CASES, demo_generator, demo_questions
from nl2sql.engine import AnalyticsEngine, answer_question
from nl2sql.schema import build_schema_context
from nl2sql.validator import Rejection

DANGEROUS_QUESTION = "Delete all subprime loans"
UNANSWERABLE_QUESTION = "What is the average credit score of borrowers in California?"


# ---------------------------------------------------------------------------
# The safety claim, end to end
# ---------------------------------------------------------------------------

def test_dangerous_question_is_blocked_and_data_is_untouched(conn):
    """The central test of the whole project.

    'Delete all subprime loans' maps to a real, executable DELETE. This
    asserts three separate things, because any one alone would be weak:

    1. the query was refused;
    2. it was refused by the safety layer specifically, not because
       something errored on the way;
    3. the table is exactly the same size afterwards -- so the refusal
       happened BEFORE the database was asked to do anything, rather than
       the DELETE running and being reported as a failure.
    """
    subprime_before = execute_select(
        conn, "SELECT COUNT(*) AS n FROM loans WHERE risk_tier = 'Subprime'"
    )[0]["n"]
    total_before = row_count(conn)
    assert subprime_before > 0, "the destructive query must have rows to destroy"

    result = answer_question(DANGEROUS_QUESTION, conn, demo_generator)

    assert not result.ok
    assert result.rejection is Rejection.NOT_A_SELECT
    assert row_count(conn) == total_before
    assert (
        execute_select(
            conn, "SELECT COUNT(*) AS n FROM loans WHERE risk_tier = 'Subprime'"
        )[0]["n"]
        == subprime_before
    )


def test_the_dangerous_demo_sql_really_is_destructive(conn):
    """Proves the previous test is not passing against a defanged string.

    A safety test is only as strong as the threat it is pointed at. If the
    canned DELETE were malformed, the block would be meaningless. This
    compiles it against the live schema to confirm SQLite accepts it as a
    valid DELETE -- EXPLAIN plans without executing, so the rows survive.
    """
    from nl2sql.demo import get_case

    case = get_case(DANGEROUS_QUESTION)
    assert case.dangerous
    assert case.sql.strip().upper().startswith("DELETE")

    before = row_count(conn)
    conn.execute(f"EXPLAIN {case.sql}")  # compiles => it is real, valid DELETE
    assert row_count(conn) == before


def test_no_demo_question_can_change_the_row_count(conn):
    """Run the entire demo set and confirm the data is inert throughout."""
    before = row_count(conn)
    for question in demo_questions():
        answer_question(question, conn, demo_generator)
    assert row_count(conn) == before


# ---------------------------------------------------------------------------
# Known-safe questions return correct results
# ---------------------------------------------------------------------------

def test_known_safe_question_returns_correct_results(conn):
    """Correctness, not just 'it ran'.

    The expected numbers are computed independently from the same
    database, so this fails if the pipeline silently returns rows from the
    wrong query -- which 'assert rows' would not catch.
    """
    result = answer_question("How many loans are in each risk tier?", conn, demo_generator)

    assert result.ok, result.message
    expected = {
        row["risk_tier"]: row["n"]
        for row in execute_select(
            conn, "SELECT risk_tier, COUNT(*) AS n FROM loans GROUP BY risk_tier"
        )
    }
    actual = {row["risk_tier"]: row["loan_count"] for row in result.rows}
    assert actual == expected
    assert sum(actual.values()) == row_count(conn)


def test_aggregates_match_the_generators_intended_shape(conn):
    """Average rates must increase with risk. A wrong GROUP BY breaks this.

    This is why the synthetic data is correlated rather than uniform: it
    gives the test an ordering to assert that is meaningful rather than
    arbitrary.
    """
    result = answer_question(
        "What is the average interest rate by risk tier?", conn, demo_generator
    )
    assert result.ok, result.message

    rates = {row["risk_tier"]: row["avg_interest_rate"] for row in result.rows}
    assert set(rates) == {"Prime", "Near-Prime", "Subprime"}
    assert rates["Prime"] < rates["Near-Prime"] < rates["Subprime"]


def test_filtered_query_respects_the_grounded_value_casing(conn):
    """Schema grounding paying off, checked rather than assumed.

    The canned SQL filters on 'Subprime' with the exact casing the sampler
    surfaced. If that casing were wrong, SQLite would return zero rows and
    no error -- so a non-empty result is the assertion that matters here.
    """
    result = answer_question(
        "Show total exposure by entity for subprime loans originated in 2024",
        conn,
        demo_generator,
    )
    assert result.ok, result.message
    assert result.rows, "empty result suggests a value-casing mismatch"
    assert all(row["total_exposure"] > 0 for row in result.rows)
    assert [r["total_exposure"] for r in result.rows] == sorted(
        (r["total_exposure"] for r in result.rows), reverse=True
    )


def test_top_n_question_returns_exactly_one_row(conn):
    result = answer_question(
        "Which product has the highest default rate?", conn, demo_generator
    )
    assert result.ok, result.message
    assert result.row_count == 1
    assert 0 <= result.rows[0]["default_rate_pct"] <= 100


# ---------------------------------------------------------------------------
# Unanswerable and unknown questions
# ---------------------------------------------------------------------------

def test_unanswerable_question_is_handled_gracefully(conn):
    """A decline is not an error and not a result.

    It must not raise, must not be reported as a safety block, and must
    not render the model's apology as a data row -- which is what would
    happen without the explicit unanswerable branch, and would read
    exactly like a real finding.
    """
    result = answer_question(UNANSWERABLE_QUESTION, conn, demo_generator)

    assert result.ok
    assert result.unanswerable
    assert result.rejection is None
    assert result.rows == []
    assert "cannot be answered" in result.message


def test_unknown_question_is_reported_not_raised(conn):
    """A generator failure must not take down the caller.

    The engine catches generation exceptions so a CLI loop or a batch of
    questions survives one bad entry.
    """
    result = answer_question("what colour is the sky", conn, demo_generator)

    assert not result.ok
    assert result.rejection is None, "a generation failure is not a safety block"
    assert "SQL generation failed" in result.message


def test_a_generator_that_returns_garbage_is_blocked_not_executed(conn):
    """The pipeline does not trust its generator, whichever one it holds."""
    before = row_count(conn)
    result = answer_question(
        "anything", conn, lambda q, s: "DROP TABLE loans; DELETE FROM loans"
    )
    assert not result.ok
    assert result.rejection is Rejection.MULTIPLE_STATEMENTS
    assert row_count(conn) == before


def test_a_generator_that_returns_nothing_is_handled(conn):
    result = answer_question("anything", conn, lambda q, s: "")
    assert not result.ok
    assert result.rejection is Rejection.EMPTY


# ---------------------------------------------------------------------------
# Transparency: the SQL always comes back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", demo_questions())
def test_every_result_carries_the_sql(question, conn):
    """Transparency is unconditional -- successes, declines and blocks alike.

    'We blocked something' is not actionable without the something, and an
    analytics answer you cannot trace to a query is not an answer.
    """
    result = answer_question(question, conn, demo_generator)
    assert result.sql, f"no SQL reported for {question!r}"
    assert result.question == question


def test_the_reported_sql_is_the_sql_that_ran(conn):
    """Re-executing the reported SQL must reproduce the reported rows.

    This is the check that makes the transparency claim mean something: it
    would fail if the engine executed one string and reported another.
    """
    result = answer_question("How many loans are in each risk tier?", conn, demo_generator)
    assert result.ok
    assert execute_select(conn, result.sql) == result.rows


def test_blocked_results_report_the_rejected_sql_and_the_reason(conn):
    result = answer_question(DANGEROUS_QUESTION, conn, demo_generator)
    assert "DELETE FROM loans" in result.sql
    assert result.rejection is Rejection.NOT_A_SELECT
    assert result.message


def test_results_carry_the_schema_context_used_for_grounding(conn):
    """A wrong answer should be traceable to what the model was told."""
    result = answer_question("How many loans are in each risk tier?", conn, demo_generator)
    assert result.schema_context == build_schema_context(conn)
    assert "risk_tier" in result.schema_context


# ---------------------------------------------------------------------------
# The grounding step feeds the generator
# ---------------------------------------------------------------------------

def test_the_generator_receives_the_live_schema_context(conn):
    """The seam is real: whatever introspection produced is what is passed.

    Guards against a pipeline that introspects for show and then hands the
    generator something stale or empty.
    """
    captured: dict[str, str] = {}

    def capturing_generator(question: str, schema_context: str) -> str:
        captured["schema"] = schema_context
        return "SELECT COUNT(*) AS n FROM loans"

    answer_question("how many loans", conn, capturing_generator)

    assert captured["schema"] == build_schema_context(conn)
    for token in ("loans", "risk_tier", "'Subprime'", "origination_date"):
        assert token in captured["schema"], f"{token} missing from grounding"


def test_schema_context_reflects_a_schema_change_without_a_code_change(conn, tmp_path):
    """The reason introspection is live rather than hardcoded.

    A new column has to appear in the grounding block with no edit to any
    prompt. Built in a throwaway database so the shared fixture stays
    pristine.
    """
    from nl2sql.database import build_database, connect

    path = tmp_path / "extended.db"
    build_database(path, row_count=50, overwrite=True)
    other = connect(path)
    try:
        assert "servicing_status" not in build_schema_context(other)
        other.execute("ALTER TABLE loans ADD COLUMN servicing_status TEXT")
        other.execute("UPDATE loans SET servicing_status = 'Current'")
        other.commit()

        context = build_schema_context(other)
        assert "servicing_status" in context
        assert "'Current'" in context, "new column's values must be sampled too"
    finally:
        other.close()


# ---------------------------------------------------------------------------
# The engine wrapper
# ---------------------------------------------------------------------------

def test_engine_wrapper_matches_the_function(conn):
    engine = AnalyticsEngine(conn, demo_generator)
    question = "How many loans are in each risk tier?"
    assert engine.ask(question).rows == answer_question(question, conn, demo_generator).rows
    assert engine.schema_context() == build_schema_context(conn)


def test_every_demo_case_reaches_its_documented_outcome(conn):
    """Each canned case must demonstrate what it claims to demonstrate."""
    for case in DEMO_CASES:
        result = answer_question(case.question, conn, demo_generator)
        if case.dangerous:
            assert not result.ok and result.rejection is not None
        elif "unanswerable" in case.sql:
            assert result.unanswerable
        else:
            assert result.ok and result.rows, case.question
