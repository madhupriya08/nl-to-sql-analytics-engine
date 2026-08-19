"""The safety layer: everything between what the model writes and what runs.

This is the part of the system that has to be right. An LLM's output is
untrusted input -- not because the model is adversarial, but because it is
a text generator with no notion of consequence, and because the question
that produced it came from a human who may be adversarial. Treating
generated SQL as trusted because 'we asked it for a SELECT' is the same
mistake as trusting a form field because the form said 'numbers only'.

The design is a **whitelist**: a query executes only if it is provably one
of the shapes we understand. Everything else is rejected, including
things nobody has thought of yet. The alternative -- blacklisting known
dangerous patterns -- fails structurally, and the README explains why at
length; the short version is that a blacklist is a list of attacks
somebody already thought of, and it is wrong the first time somebody
thinks of a new one.

Four checks run in order, cheapest first, each a precondition for the next:

1. exactly one statement (blocks statement chaining)
2. starts with SELECT or WITH (the only two read-only entry points)
3. no forbidden keyword anywhere, by word-boundary regex (catches
   mutations buried inside subqueries, where check 2 cannot see them)
4. EXPLAIN against the real database (catches hallucinated identifiers
   and syntax errors, without executing anything)

Nothing reaches ``execute()`` without passing all four.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from enum import Enum


class Rejection(str, Enum):
    """Why a query was rejected.

    A machine-readable reason rather than only a message, so the tests can
    assert *which* check fired. A test that merely asserts 'blocked' would
    still pass if a chained DELETE were rejected for being empty -- and
    would keep passing after the chaining check was accidentally removed.
    """

    EMPTY = "empty_query"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_A_SELECT = "not_a_select"
    FORBIDDEN_KEYWORD = "forbidden_keyword"
    DID_NOT_COMPILE = "did_not_compile"


#: Keywords that must not appear anywhere in the query.
#:
#: This list is *not* the security boundary -- checks 1 and 2 are, since
#: they establish that we have a single read-only statement. This is
#: defence in depth for the case those two cannot see: a mutation nested
#: inside a subquery, where the statement still begins with SELECT.
#:
#: The set covers DML (DELETE, INSERT, UPDATE, REPLACE, UPSERT, MERGE),
#: DDL (DROP, CREATE, ALTER, TRUNCATE, RENAME), SQLite's own escape
#: hatches (ATTACH and DETACH, which reach other database files; PRAGMA,
#: which can disable integrity enforcement; VACUUM, which rewrites the
#: file), and privilege and execution verbs that other engines honour
#: (GRANT, REVOKE, EXEC, EXECUTE, CALL) so the validator does not become
#: unsafe the day this is pointed at Postgres.
FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        # Data modification
        "DELETE", "INSERT", "UPDATE", "REPLACE", "UPSERT", "MERGE",
        # Schema modification
        "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
        # SQLite escape hatches
        "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE",
        # Transaction control -- not destructive alone, but a query that
        # opens or ends a transaction is not the single read we asked for.
        "BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE",
        # Privilege and execution verbs from other engines
        "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "DO",
        # Bulk load/unload paths that can touch the filesystem
        "COPY", "LOAD", "OUTFILE", "DUMPFILE", "IMPORT", "EXPORT",
        # Trigger/view/index creation verbs, in case CREATE is ever split
        "TRIGGER",
    }
)

#: One alternation of all forbidden keywords, anchored on word boundaries.
#:
#: The word boundaries are the point. A naive substring search for
#: 'CREATE' matches the column name 'created_at' and blocks a legitimate
#: query; a search for 'DO' matches almost every English word inside a
#: string literal. \b makes the match a whole word, so 'created_at' is
#: fine while 'CREATE TABLE' is not. Compiled once at import: this regex
#: runs on every query.
_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(FORBIDDEN_KEYWORDS)) + r")\b",
    re.IGNORECASE,
)

#: The only two tokens a query may begin with. WITH is included because
#: CTEs are genuinely useful for the analytical questions this answers --
#: but see :func:`_check_starts_with_select` for why WITH needs an extra
#: guard that SELECT does not.
ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH")

#: SQLite permits INSERT/UPDATE/DELETE as the *body* of a CTE:
#:
#:     WITH x AS (DELETE FROM loans RETURNING *) SELECT * FROM x
#:
#: which begins with WITH and would sail past check 2. Check 3 catches it
#: on the DELETE keyword, but relying on a single check for a known,
#: specific bypass is fragile, so the leading-keyword check rejects a
#: WITH statement that does not eventually reach a SELECT.
_WITH_MUST_REACH_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validation. Falsy when the query must not run."""

    is_valid: bool
    #: The normalised SQL that should be executed if valid.
    sql: str = ""
    reason: Rejection | None = None
    message: str = ""
    #: Which of the four checks passed, in order, for transparency and to
    #: let tests assert a query was rejected at the *expected* stage.
    checks_passed: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.is_valid


def _scan(
    sql: str, split_on_semicolon: bool = False, strip_comments: bool = False
) -> list[str]:
    """Single quote-aware left-to-right scan over a SQL string.

    Both comment stripping and statement splitting need to know whether a
    given character sits inside a string literal, and they must agree
    perfectly. Two separate implementations that drift apart by one edge
    case is exactly how a bypass appears: strip comments quote-blind and
    ``WHERE entity = '-- x'`` loses its closing quote; split quote-blind
    and ``WHERE entity = 'A;B'`` looks like two statements. So there is
    one scanner, and both callers share its notion of "inside a string".

    Returns the scanned text as a list of segments, split at top-level
    semicolons when ``split_on_semicolon`` is set; otherwise a
    single-element list.

    Quote handling follows SQL: single quotes delimit string literals,
    double quotes delimit identifiers, and a doubled quote inside either
    is an escaped quote rather than a terminator ('it''s'). If the input
    ends with an unterminated string, the scan simply ends inside it --
    the remaining checks then see a malformed statement and reject it,
    which is the direction we want to fail in.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    n = len(sql)
    while i < n:
        char = sql[i]

        if quote:
            current.append(char)
            if char == quote:
                if i + 1 < n and sql[i + 1] == quote:
                    # Escaped quote: consume both, stay inside the literal.
                    current.append(sql[i + 1])
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        if char in ("'", '"'):
            quote = char
            current.append(char)
            i += 1
            continue

        if strip_comments and sql.startswith("--", i):
            # Line comment: skip to end of line, leaving a space so tokens
            # either side do not fuse into one word.
            end = sql.find("\n", i)
            i = n if end == -1 else end
            current.append(" ")
            continue

        if strip_comments and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            current.append(" ")
            continue

        if split_on_semicolon and char == ";":
            segments.append("".join(current))
            current = []
            i += 1
            continue

        current.append(char)
        i += 1

    segments.append("".join(current))
    return segments


def _strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments.

    Comments are stripped *before* the structural checks, not after,
    because otherwise they are a bypass in both directions: a trailing
    ``-- ;DROP TABLE`` would trip the keyword check on a harmless query,
    and more dangerously a comment can hide a statement separator from a
    naive splitter (``SELECT 1 --\n; DROP TABLE loans``).

    Stripping is quote-aware, so a literal containing ``--`` or ``/*``
    survives intact and a legitimate question about such a value is not
    falsely rejected.
    """
    return _scan(sql, strip_comments=True)[0]


def normalise(sql: str) -> str:
    """Strip comments, collapse whitespace, and drop one trailing semicolon.

    A single trailing semicolon is a formatting habit, not statement
    chaining, so it is removed here rather than rejected -- but only after
    check 1 has confirmed there is nothing after it.
    """
    text = _strip_sql_comments(sql).strip()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(";").strip()


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons that are not inside a string literal.

    Naively splitting on ';' would treat ``WHERE entity = 'A;B'`` as two
    statements and reject a legitimate query, so this delegates to the
    shared quote-aware scanner.

    The bar this has to clear is low but absolute: a *false positive*
    rejects a valid query, which is annoying; a *false negative* lets a
    chained statement through, which is the entire attack. When the input
    ends inside an unterminated literal the scan stops there, which fails
    toward rejection at a later check rather than toward execution.
    """
    return [s.strip() for s in _scan(sql, split_on_semicolon=True) if s.strip()]


def _check_single_statement(sql: str) -> ValidationResult | None:
    """Check 1: exactly one statement.

    This runs first because statement chaining defeats every other check.
    ``SELECT 1; DROP TABLE loans`` starts with SELECT, so check 2 passes
    on the first half; only counting statements catches it.
    """
    statements = _split_statements(sql)
    if len(statements) > 1:
        return ValidationResult(
            is_valid=False,
            reason=Rejection.MULTIPLE_STATEMENTS,
            message=(
                f"query contains {len(statements)} statements; exactly one is "
                f"allowed (statement chaining is how a SELECT smuggles a DROP)"
            ),
        )
    return None


def _check_starts_with_select(sql: str) -> ValidationResult | None:
    """Check 2: the statement is a read.

    SELECT and WITH are the only two forms that can be read-only in
    SQLite. WITH gets an extra condition: it must reach a SELECT
    somewhere, because SQLite allows a data-modifying statement as a CTE
    body.
    """
    leading = re.match(r"\s*([A-Za-z_]+)", sql)
    keyword = (leading.group(1) if leading else "").upper()

    if keyword not in ALLOWED_LEADING_KEYWORDS:
        return ValidationResult(
            is_valid=False,
            reason=Rejection.NOT_A_SELECT,
            message=(
                f"query starts with {keyword or '(nothing)'!r}; only SELECT and "
                f"WITH are allowed"
            ),
        )

    if keyword == "WITH" and not _WITH_MUST_REACH_SELECT.search(sql):
        return ValidationResult(
            is_valid=False,
            reason=Rejection.NOT_A_SELECT,
            message="WITH statement never reaches a SELECT",
        )
    return None


def _check_no_forbidden_keywords(sql: str) -> ValidationResult | None:
    """Check 3: no forbidden keyword anywhere in the statement.

    'Anywhere' is what makes this worth having on top of check 2. A query
    can begin with SELECT and still carry a mutation in a subquery::

        SELECT * FROM loans WHERE loan_id IN (DELETE FROM loans RETURNING loan_id)

    Check 2 sees SELECT and is satisfied. Only scanning the entire string
    finds the DELETE.

    The trade-off, stated plainly: this scans string literals too, so a
    question about a company named 'Drop Inc' would be rejected. That is
    the correct direction to be wrong in. A false rejection is a visible
    error the user can rephrase around; a false acceptance is silent data
    loss. Skipping literals would mean parsing them, and a parser that
    disagrees by one character with SQLite's parser is a bypass.
    """
    match = _FORBIDDEN_PATTERN.search(sql)
    if match:
        return ValidationResult(
            is_valid=False,
            reason=Rejection.FORBIDDEN_KEYWORD,
            message=(
                f"query contains the forbidden keyword {match.group(0).upper()!r} "
                f"at position {match.start()}"
            ),
        )
    return None


def _check_compiles(sql: str, conn: sqlite3.Connection) -> ValidationResult | None:
    """Check 4: ``EXPLAIN <sql>`` against the live database.

    SQLite compiles the statement to bytecode and hands back the program
    without running a single instruction. That gives two things no amount
    of string inspection can:

    * every table and column reference is resolved against the real
      schema, so a hallucinated ``customers`` table or ``fico_score``
      column is caught here rather than at execution;
    * the statement is proven syntactically valid, so the engine is not
      guessing whether a parse error is a safety problem.

    This is last because it is the only check that touches the database,
    and it should only ever see SQL that the three static checks have
    already cleared.
    """
    try:
        conn.execute(f"EXPLAIN {sql}")
    except sqlite3.Error as exc:
        return ValidationResult(
            is_valid=False,
            reason=Rejection.DID_NOT_COMPILE,
            message=f"query failed to compile against the live schema: {exc}",
        )
    return None


def validate_sql(sql: str, conn: sqlite3.Connection | None = None) -> ValidationResult:
    """Run the full whitelist validation.

    ``conn`` is optional so the three static checks can be exercised with
    no database at all; when it is omitted, check 4 is skipped and
    ``checks_passed`` says so. The query engine always passes a
    connection -- a query is never executed on a partial validation.
    """
    if sql is None or not sql.strip():
        return ValidationResult(
            is_valid=False,
            reason=Rejection.EMPTY,
            message="query is empty",
        )

    normalised = normalise(sql)
    if not normalised:
        # Reachable when the input was only comments or semicolons -- e.g.
        # a model that returned '-- I cannot answer that'.
        return ValidationResult(
            is_valid=False,
            reason=Rejection.EMPTY,
            message="query is empty after stripping comments",
        )

    passed: list[str] = []

    # Check 1 runs against the comment-stripped text but *before* the
    # trailing semicolon is dropped, so 'SELECT 1; DROP TABLE loans' is
    # still visibly two statements.
    decommented = re.sub(r"\s+", " ", _strip_sql_comments(sql)).strip()
    for check, name in (
        (lambda: _check_single_statement(decommented), "single_statement"),
        (lambda: _check_starts_with_select(normalised), "starts_with_select"),
        (lambda: _check_no_forbidden_keywords(normalised), "no_forbidden_keywords"),
    ):
        failure = check()
        if failure is not None:
            return ValidationResult(
                is_valid=False,
                sql=normalised,
                reason=failure.reason,
                message=failure.message,
                checks_passed=tuple(passed),
            )
        passed.append(name)

    if conn is not None:
        failure = _check_compiles(normalised, conn)
        if failure is not None:
            return ValidationResult(
                is_valid=False,
                sql=normalised,
                reason=failure.reason,
                message=failure.message,
                checks_passed=tuple(passed),
            )
        passed.append("compiles")

    return ValidationResult(
        is_valid=True,
        sql=normalised,
        checks_passed=tuple(passed),
    )
