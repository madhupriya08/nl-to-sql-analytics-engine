"""The query pipeline: grounding -> generation -> validation -> execution.

Four stages, in a fixed order, with one rule: **validation sits between
generation and execution and cannot be bypassed**. There is no code path
in this module that reaches :func:`nl2sql.database.execute_select`
without a passing :class:`~nl2sql.validator.ValidationResult` in hand.

The generator is a parameter, not an import. ``answer_question`` takes a
``sql_generator_fn(question, schema_context) -> str`` and does not care
whether it is a Claude call or a dictionary lookup. That seam is what
makes the safety-critical half of the system testable without
credentials, and the README argues it is a design decision rather than a
convenience.

Every answer carries the exact SQL that ran -- and, when a query is
blocked, the exact SQL that was rejected and why. An analytics tool whose
output you cannot trace back to a query is not an analytics tool; it is a
plausible-sounding oracle.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .database import execute_select
from .schema import build_schema_context
from .validator import Rejection, ValidationResult, validate_sql

#: What the engine requires of a SQL generator. Kept as a Protocol so the
#: live generator and the demo generator are interchangeable without
#: either importing the other.
class SqlGenerator(Protocol):
    def __call__(self, question: str, schema_context: str) -> str: ...


#: Marker used by the generation prompt for a question the schema cannot
#: answer. Recognising it lets the engine report 'unanswerable' rather
#: than presenting a one-row table containing an apology as a result.
UNANSWERABLE_MARKER = "question cannot be answered from this schema"


@dataclass(frozen=True)
class QueryResult:
    """The complete, auditable record of one question.

    Deliberately carries the failure detail as well as the success
    detail. A caller should be able to answer 'what did the model write,
    and what did the system do about it?' from this object alone, without
    reading logs.
    """

    question: str
    #: The SQL that ran, or -- when blocked -- the SQL that was rejected.
    #: Never None on a completed run, because "we blocked something" is
    #: not a useful message without the something.
    sql: str
    ok: bool
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: Machine-readable rejection reason, when validation blocked the query.
    rejection: Rejection | None = None
    #: Human-readable explanation of a block or an error.
    message: str = ""
    #: Which validation checks the query passed before it was blocked (or
    #: all four, when it ran). Makes the safety layer's work visible.
    checks_passed: tuple[str, ...] = ()
    #: The schema block that grounded generation, kept so a wrong answer
    #: can be traced to what the model was actually told.
    schema_context: str = ""
    #: True when the model reported the question as unanswerable. Distinct
    #: from ok=False: nothing went wrong, there is simply no answer here.
    unanswerable: bool = False

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return self.ok


def _blocked(
    question: str, validation: ValidationResult, schema_context: str
) -> QueryResult:
    return QueryResult(
        question=question,
        sql=validation.sql or "",
        ok=False,
        rejection=validation.reason,
        message=validation.message,
        checks_passed=validation.checks_passed,
        schema_context=schema_context,
    )


def answer_question(
    question: str,
    conn: sqlite3.Connection,
    sql_generator_fn: SqlGenerator | Callable[[str, str], str],
    max_rows: int | None = 1_000,
) -> QueryResult:
    """Run one question through the full pipeline.

    Stages, in order:

    1. **Ground** -- introspect the live schema for this connection.
    2. **Generate** -- hand the question and schema to ``sql_generator_fn``.
       Its output is treated as untrusted text from here on.
    3. **Validate** -- four whitelist checks, including EXPLAIN against
       this same connection. A failure returns immediately.
    4. **Execute** -- only now, and only the normalised SQL the validator
       returned, not the raw string the generator produced.

    Note that stage 4 executes ``validation.sql`` rather than the
    generator's output. That closes a small but real time-of-check /
    time-of-use gap: if the two could differ, the string that was checked
    would not be the string that runs.
    """
    schema_context = build_schema_context(conn)

    try:
        raw_sql = sql_generator_fn(question, schema_context)
    except Exception as exc:
        # Generation failures (no API key, network error, unknown demo
        # question) are reported as failures of this run, not raised, so a
        # CLI loop or a batch of questions survives one bad question.
        return QueryResult(
            question=question,
            sql="",
            ok=False,
            message=f"SQL generation failed: {exc}",
            schema_context=schema_context,
        )

    validation = validate_sql(raw_sql, conn)
    if not validation:
        return _blocked(question, validation, schema_context)

    if UNANSWERABLE_MARKER in validation.sql:
        # The model declined, and did so via a query that is itself
        # harmless and passed validation. Report the decline rather than
        # rendering the apology as a data row.
        return QueryResult(
            question=question,
            sql=validation.sql,
            ok=True,
            rows=[],
            message=(
                "the model reported that this question cannot be answered "
                "from the available schema"
            ),
            checks_passed=validation.checks_passed,
            schema_context=schema_context,
            unanswerable=True,
        )

    try:
        rows = execute_select(conn, validation.sql, max_rows=max_rows)
    except sqlite3.Error as exc:
        # Validation proved the query compiles; a failure here is a
        # runtime problem (a divide by zero, a locked database), not a
        # safety one, and is reported as such.
        return QueryResult(
            question=question,
            sql=validation.sql,
            ok=False,
            message=f"query failed at execution: {exc}",
            checks_passed=validation.checks_passed,
            schema_context=schema_context,
        )

    return QueryResult(
        question=question,
        sql=validation.sql,
        ok=True,
        rows=rows,
        checks_passed=validation.checks_passed,
        schema_context=schema_context,
    )


class AnalyticsEngine:
    """Convenience wrapper binding a connection to a generator.

    Same pipeline as :func:`answer_question`; this only saves passing the
    two arguments repeatedly from a CLI loop. The function remains the
    real entry point so tests can drive it without constructing anything.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        sql_generator_fn: SqlGenerator | Callable[[str, str], str],
        max_rows: int | None = 1_000,
    ) -> None:
        self.conn = conn
        self.sql_generator_fn = sql_generator_fn
        self.max_rows = max_rows

    def ask(self, question: str) -> QueryResult:
        return answer_question(
            question, self.conn, self.sql_generator_fn, max_rows=self.max_rows
        )

    def schema_context(self) -> str:
        """The grounding block as the model would receive it right now."""
        return build_schema_context(self.conn)
