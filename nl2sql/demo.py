"""Canned question-to-SQL pairs: a real generator with a fixed policy.

The demo generator satisfies the same contract as the live one --
``(question, schema_context) -> str`` -- so the engine cannot tell them
apart. Swapping it in exercises every stage of the pipeline for real:
real schema introspection, real validation, real SQLite execution against
the real 5,000-row database. The only thing replaced is the part that
needs a credit card.

The set is chosen to cover the pipeline's distinct outcomes rather than
to look impressive:

* four ordinary analytical questions that should run and return correct
  numbers;
* one question whose answer is a genuine, executable ``DELETE`` -- the
  case that proves the validator blocks destruction rather than merely
  being tested against toy strings;
* one question the schema cannot answer, so the 'model declined' path is
  exercised too.

The dangerous entry is the important one. A safety layer tested only
against queries that were never going to run anyway proves nothing. The
SQL below is real: paste it into sqlite3 and 1,015 rows disappear. The
test suite runs it through the engine, asserts it is blocked, and asserts
the row count is identical before and after.
"""

from __future__ import annotations

from dataclasses import dataclass

TABLE = "loans"


@dataclass(frozen=True)
class DemoCase:
    """One canned pair, with a note on what it is meant to demonstrate."""

    question: str
    sql: str
    #: What this case exercises. Printed by ``--list-demos`` so the demo
    #: explains itself instead of being a wall of anonymous SQL.
    demonstrates: str
    #: True for the deliberately destructive case.
    dangerous: bool = False


DEMO_CASES: tuple[DemoCase, ...] = (
    DemoCase(
        question="How many loans are in each risk tier?",
        sql=(
            "SELECT risk_tier, COUNT(*) AS loan_count "
            "FROM loans "
            "GROUP BY risk_tier "
            "ORDER BY loan_count DESC"
        ),
        demonstrates=(
            "the baseline happy path: a grouped count that the index on "
            "risk_tier answers directly"
        ),
    ),
    DemoCase(
        question="What is the average interest rate by risk tier?",
        sql=(
            "SELECT risk_tier, ROUND(AVG(interest_rate), 4) AS avg_interest_rate "
            "FROM loans "
            "GROUP BY risk_tier "
            "ORDER BY avg_interest_rate"
        ),
        demonstrates=(
            "aggregation whose correctness is checkable by eye -- rates must "
            "rise Prime < Near-Prime < Subprime, so a wrong GROUP BY shows up"
        ),
    ),
    DemoCase(
        question="Which product has the highest default rate?",
        sql=(
            "SELECT product, "
            "COUNT(*) AS loan_count, "
            "ROUND(AVG(default_flag) * 100, 2) AS default_rate_pct "
            "FROM loans "
            "GROUP BY product "
            "ORDER BY default_rate_pct DESC "
            "LIMIT 1"
        ),
        demonstrates=(
            "a top-N question: LIMIT plus an ordered aggregate, and the "
            "0/1 default_flag being averaged into a rate"
        ),
    ),
    DemoCase(
        question="Show total exposure by entity for subprime loans originated in 2024",
        sql=(
            "SELECT entity, "
            "COUNT(*) AS loan_count, "
            "ROUND(SUM(loan_amount), 2) AS total_exposure "
            "FROM loans "
            "WHERE risk_tier = 'Subprime' "
            "AND origination_date >= '2024-01-01' "
            "AND origination_date <= '2024-12-31' "
            "GROUP BY entity "
            "ORDER BY total_exposure DESC"
        ),
        demonstrates=(
            "schema grounding paying off: 'Subprime' is written with the "
            "exact casing the sampler surfaced, and the ISO date range "
            "works because dates are stored as sortable TEXT"
        ),
    ),
    DemoCase(
        question="Delete all subprime loans",
        # Deliberately real. This is not a mangled or defanged statement --
        # it executes and removes 1,015 rows. The point of the demo is that
        # the validator stops it, and the tests prove the row count is
        # unchanged afterwards. A safety layer demonstrated only against
        # harmless input demonstrates nothing.
        sql="DELETE FROM loans WHERE risk_tier = 'Subprime'",
        demonstrates=(
            "THE SAFETY CASE: a genuinely destructive statement, blocked at "
            "check 2 (does not start with SELECT) before the database is "
            "ever asked to run it"
        ),
        dangerous=True,
    ),
    DemoCase(
        question="What is the average credit score of borrowers in California?",
        sql=(
            "SELECT 'question cannot be answered from this schema' AS unanswerable"
        ),
        demonstrates=(
            "graceful decline: the loan book has neither credit scores nor "
            "geography, so the model reports that rather than inventing a "
            "column -- and the decline is itself a valid, harmless query"
        ),
    ),
)


class UnknownDemoQuestion(KeyError):
    """Raised when a question has no canned answer.

    A KeyError subclass carrying a readable message: the engine catches
    generation exceptions and reports them, so the CLI shows the list of
    known questions instead of a bare traceback.
    """

    def __init__(self, question: str) -> None:
        known = "\n  - ".join(case.question for case in DEMO_CASES)
        super().__init__(
            f"no canned SQL for {question!r}. Demo mode answers only:\n  - {known}"
        )

    def __str__(self) -> str:  # KeyError repr()s its argument otherwise
        return self.args[0]


def _normalise_question(question: str) -> str:
    """Fold case and trailing punctuation so demo lookup is not brittle.

    Only the lookup is lenient. Nothing about matching a question affects
    what the validator does to the SQL that comes back.
    """
    return " ".join(question.lower().replace("?", " ").split())


_LOOKUP = {_normalise_question(case.question): case for case in DEMO_CASES}


def demo_generator(question: str, schema_context: str) -> str:
    """A ``sql_generator_fn`` backed by the canned pairs.

    ``schema_context`` is accepted and ignored -- the signature has to
    match the live generator exactly, because the entire point is that the
    engine cannot distinguish them.
    """
    case = _LOOKUP.get(_normalise_question(question))
    if case is None:
        raise UnknownDemoQuestion(question)
    return case.sql


def demo_questions() -> tuple[str, ...]:
    """Every question demo mode can answer, in order."""
    return tuple(case.question for case in DEMO_CASES)


def get_case(question: str) -> DemoCase:
    """Look up the full case, for callers that want the annotation too."""
    case = _LOOKUP.get(_normalise_question(question))
    if case is None:
        raise UnknownDemoQuestion(question)
    return case
