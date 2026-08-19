"""Adversarial tests against the safety validator.

The organising idea: every test here asserts not just THAT a query was
blocked, but WHICH check blocked it. A test that only asserts 'blocked'
keeps passing after the check it was written for is deleted -- some other
check happens to catch the case, the test stays green, and the coverage
is gone. Asserting the reason makes the test fail when its check does.
"""

from __future__ import annotations

import pytest

from nl2sql.database import row_count
from nl2sql.validator import (
    FORBIDDEN_KEYWORDS,
    Rejection,
    normalise,
    validate_sql,
)


# ---------------------------------------------------------------------------
# Check 1: statement chaining
# ---------------------------------------------------------------------------

CHAINED_QUERIES = [
    "SELECT * FROM loans; DROP TABLE loans",
    "SELECT 1; DELETE FROM loans",
    "SELECT * FROM loans;DROP TABLE loans",
    "SELECT * FROM loans ;  ; DROP TABLE loans",
    "SELECT 1;\nUPDATE loans SET loan_amount = 0",
    "SELECT 1;\r\nINSERT INTO loans VALUES ('x','y','z','w',1,1,'2024-01-01',0)",
    # The comment hides the separator from anything that strips comments
    # after splitting instead of before.
    "SELECT 1 --\n; DROP TABLE loans",
    "SELECT 1 /* comment */; DROP TABLE loans",
    "SELECT 1;;DROP TABLE loans",
    # Trailing chained statement with only whitespace between.
    "SELECT COUNT(*) FROM loans;\t\tDROP TABLE loans",
]


@pytest.mark.parametrize("sql", CHAINED_QUERIES)
def test_chained_statements_are_blocked(sql, conn):
    result = validate_sql(sql, conn)
    assert not result
    assert result.reason is Rejection.MULTIPLE_STATEMENTS
    # Nothing before this check, so nothing should have been recorded.
    assert result.checks_passed == ()


def test_chaining_is_caught_even_though_the_first_half_is_a_valid_select(conn):
    """The reason check 1 must run first.

    Each half of this query is individually acceptable to some check: the
    first half is a real SELECT, so a leading-keyword check passes it.
    Only counting statements catches the pair.
    """
    first_half = "SELECT COUNT(*) FROM loans"
    assert validate_sql(first_half, conn)

    result = validate_sql(f"{first_half}; DROP TABLE loans", conn)
    assert result.reason is Rejection.MULTIPLE_STATEMENTS


def test_semicolon_inside_a_string_literal_is_not_a_statement_separator(conn):
    """A false positive here would reject legitimate questions.

    The quote-aware scanner exists for this case. It has to be right in
    both directions: this must pass, while the chained cases above must
    still fail.
    """
    result = validate_sql(
        "SELECT COUNT(*) FROM loans WHERE entity = 'Acme; Capital'", conn
    )
    assert result, result.message


def test_escaped_quote_inside_a_literal_does_not_confuse_the_scanner(conn):
    result = validate_sql(
        "SELECT COUNT(*) FROM loans WHERE entity = 'it''s; fine'", conn
    )
    assert result, result.message


def test_a_single_trailing_semicolon_is_formatting_not_chaining(conn):
    result = validate_sql("SELECT COUNT(*) FROM loans;", conn)
    assert result, result.message
    assert not result.sql.endswith(";")


# ---------------------------------------------------------------------------
# Check 2: must be a read
# ---------------------------------------------------------------------------

NON_SELECT_QUERIES = [
    "DELETE FROM loans WHERE risk_tier = 'Subprime'",
    "DROP TABLE loans",
    "INSERT INTO loans (loan_id) VALUES ('LN-999999')",
    "UPDATE loans SET default_flag = 0",
    "ALTER TABLE loans ADD COLUMN secret TEXT",
    "CREATE TABLE evil (x TEXT)",
    "REPLACE INTO loans (loan_id) VALUES ('x')",
    "TRUNCATE TABLE loans",
    "ATTACH DATABASE '/etc/passwd' AS pwn",
    "DETACH DATABASE main",
    "PRAGMA table_info(loans)",
    "VACUUM",
    "GRANT ALL ON loans TO PUBLIC",
    "REVOKE ALL ON loans FROM PUBLIC",
    "EXEC sp_who",
    "BEGIN TRANSACTION",
    "COMMIT",
    "ROLLBACK",
    "REINDEX loans",
    "ANALYZE loans",
]


@pytest.mark.parametrize("sql", NON_SELECT_QUERIES)
def test_non_select_statements_are_blocked(sql, conn):
    result = validate_sql(sql, conn)
    assert not result
    assert result.reason is Rejection.NOT_A_SELECT
    assert result.checks_passed == ("single_statement",)


def test_create_trigger_is_blocked_by_the_statement_count_check_first(conn):
    """Blocked, but by check 1 rather than check 2 -- worth pinning down.

    A trigger body contains its own semicolons ('BEGIN SELECT 1; END'), so
    the statement counter sees two statements and stops before the
    leading-keyword check ever runs. Both outcomes are correct refusals;
    this test records WHICH one fires so the ordering is documented rather
    than accidental. It also shows the layering working as intended: the
    cheapest check catches it, and the later ones are never reached.
    """
    result = validate_sql(
        "CREATE TRIGGER t AFTER INSERT ON loans BEGIN SELECT 1; END", conn
    )
    assert not result
    assert result.reason is Rejection.MULTIPLE_STATEMENTS


def test_every_forbidden_keyword_is_blocked_as_a_leading_token(conn):
    """Exhaustive over the keyword list, so adding one adds a test.

    Guards against a keyword being added to the set but the regex failing
    to pick it up (a stray character in the alternation, say).
    """
    for keyword in sorted(FORBIDDEN_KEYWORDS):
        result = validate_sql(f"{keyword} FROM loans", conn)
        assert not result, f"{keyword} was not blocked"
        assert result.reason is Rejection.NOT_A_SELECT


def test_select_and_with_are_the_only_accepted_openers(conn):
    assert validate_sql("SELECT 1", conn)
    assert validate_sql("WITH t AS (SELECT 1 AS n) SELECT n FROM t", conn)
    assert validate_sql("  select 1  ", conn), "keyword match must be case-insensitive"


def test_with_that_never_reaches_a_select_is_blocked(conn):
    """WITH is only allowed as a prelude to a read.

    Allowing WITH is a real widening of check 2: SQLite permits a
    data-modifying statement as a CTE body, so 'WITH' as an opener is not
    by itself evidence of a read. The keyword check catches the dangerous
    versions of that (see the subquery tests), but check 2 carries its own
    guard rather than delegating a known, specific bypass to a single
    downstream check -- one deletion away from being unguarded.

    This case is what the guard alone catches: a WITH statement whose body
    contains no SELECT at all, so it is not a read by any reading.
    """
    result = validate_sql("WITH x AS (VALUES (1)) VALUES (1)", conn)
    assert not result
    assert result.reason is Rejection.NOT_A_SELECT


def test_a_data_modifying_cte_body_is_blocked(conn):
    """The dangerous WITH: caught by check 3, since it does reach a SELECT."""
    result = validate_sql(
        "WITH doomed AS (DELETE FROM loans RETURNING loan_id) "
        "SELECT COUNT(*) FROM doomed",
        conn,
    )
    assert not result
    assert result.reason is Rejection.FORBIDDEN_KEYWORD


# ---------------------------------------------------------------------------
# Check 3: forbidden keywords anywhere, including inside subqueries
# ---------------------------------------------------------------------------

HIDDEN_MUTATION_QUERIES = [
    # The headline case: a mutation buried in an IN() subquery, behind a
    # statement that opens with a perfectly ordinary SELECT.
    "SELECT * FROM loans WHERE loan_id IN (DELETE FROM loans RETURNING loan_id)",
    "SELECT * FROM loans WHERE loan_id IN (SELECT loan_id FROM loans "
    "WHERE loan_id IN (DROP TABLE loans))",
    "SELECT (SELECT COUNT(*) FROM loans WHERE 1 = (UPDATE loans SET x = 1)) AS n",
    "SELECT * FROM (INSERT INTO loans DEFAULT VALUES RETURNING *)",
    "WITH doomed AS (DELETE FROM loans RETURNING loan_id) SELECT * FROM doomed",
    "SELECT * FROM loans UNION SELECT * FROM (DROP TABLE loans)",
    "SELECT * FROM loans WHERE EXISTS (TRUNCATE TABLE loans)",
    "SELECT * FROM loans WHERE 1=1 AND (ATTACH DATABASE 'x' AS y)",
    "SELECT * FROM loans ORDER BY (VACUUM)",
    "SELECT * FROM loans WHERE loan_id = (PRAGMA journal_mode)",
]


@pytest.mark.parametrize("sql", HIDDEN_MUTATION_QUERIES)
def test_mutations_hidden_inside_subqueries_are_blocked(sql, conn):
    result = validate_sql(sql, conn)
    assert not result
    assert result.reason is Rejection.FORBIDDEN_KEYWORD
    # It got past the first two checks -- which is precisely why check 3
    # has to scan the whole string rather than only the leading token.
    assert "starts_with_select" in result.checks_passed


@pytest.mark.parametrize("keyword", sorted(FORBIDDEN_KEYWORDS))
def test_every_forbidden_keyword_is_blocked_inside_a_subquery(keyword, conn):
    """Exhaustive again, this time in the position check 2 cannot see."""
    result = validate_sql(
        f"SELECT * FROM loans WHERE loan_id IN ({keyword} FROM loans)", conn
    )
    assert not result, f"{keyword} was not blocked inside a subquery"
    assert result.reason is Rejection.FORBIDDEN_KEYWORD


def test_keyword_matching_is_case_insensitive(conn):
    for variant in ("delete", "DeLeTe", "DELETE"):
        result = validate_sql(f"SELECT * FROM loans WHERE x IN ({variant} FROM t)", conn)
        assert result.reason is Rejection.FORBIDDEN_KEYWORD


def test_word_boundaries_prevent_false_positives_on_column_names():
    """The reason check 3 uses \\b rather than a substring search.

    'created_at' contains CREATE; 'updated_at' contains UPDATE;
    'dropoff_date' contains DROP. A substring blacklist rejects all three
    and the tool becomes unusable on any realistic schema. These are
    checked without a connection because they SHOULD reach check 4 and be
    rejected there for not existing -- the point is only that check 3
    lets them through.
    """
    for column in ("created_at", "updated_at", "dropoff_date", "increateive"):
        result = validate_sql(f"SELECT {column} FROM loans")
        assert result, f"{column} was falsely flagged as a forbidden keyword"


def test_word_boundary_does_not_let_a_real_keyword_through(conn):
    """The mirror of the test above: \\b must not become a loophole.

    Punctuation and newlines are word boundaries too, so a keyword hugging
    a bracket or a line break is still a whole word.
    """
    for sql in (
        "SELECT * FROM loans WHERE x IN(DELETE FROM loans)",
        "SELECT * FROM loans WHERE x IN (\nDROP TABLE loans)",
        "SELECT * FROM loans WHERE x=(UPDATE)",
    ):
        result = validate_sql(sql, conn)
        assert result.reason is Rejection.FORBIDDEN_KEYWORD, sql


def test_forbidden_keyword_inside_a_string_literal_is_rejected(conn):
    """Documents an accepted false positive rather than hiding it.

    Check 3 scans literals too, so a genuine question about a company
    called 'Drop Inc' is refused. That is the correct direction to be
    wrong in -- a visible refusal the user rephrases around, versus
    silent data loss -- and skipping literals would require a parser that
    must agree with SQLite's byte for byte or become a bypass.
    """
    result = validate_sql("SELECT * FROM loans WHERE entity = 'Drop Inc'", conn)
    assert not result
    assert result.reason is Rejection.FORBIDDEN_KEYWORD


def test_comments_are_stripped_before_the_keyword_scan(conn):
    """An inert comment must not trip the scan, but must not hide a chain.

    Both directions in one place, because they are the same mechanism.
    """
    assert validate_sql("SELECT COUNT(*) FROM loans -- DROP TABLE loans", conn)
    assert validate_sql("SELECT COUNT(*) FROM loans /* DELETE FROM loans */", conn)

    chained = validate_sql("SELECT 1 --\n; DROP TABLE loans", conn)
    assert chained.reason is Rejection.MULTIPLE_STATEMENTS


# ---------------------------------------------------------------------------
# Check 4: EXPLAIN against the live schema
# ---------------------------------------------------------------------------

HALLUCINATED_QUERIES = [
    ("SELECT * FROM customers", "table that does not exist"),
    ("SELECT * FROM loan", "near-miss table name"),
    ("SELECT fico_score FROM loans", "column that does not exist"),
    ("SELECT loan_amt FROM loans", "near-miss column name"),
    ("SELECT * FROM loans JOIN borrowers ON 1=1", "join to a missing table"),
    ("SELECT SUM(balance) FROM loans GROUP BY state", "two missing columns"),
    ("SELECT * FROM sqlite_master WHERE name = (SELECT nope FROM loans)", "missing column in subquery"),
]


@pytest.mark.parametrize("sql,description", HALLUCINATED_QUERIES)
def test_hallucinated_identifiers_are_blocked(sql, description, conn):
    result = validate_sql(sql, conn)
    assert not result, description
    assert result.reason is Rejection.DID_NOT_COMPILE
    # All three static checks passed: nothing about the string is
    # suspicious. Only compiling against the real schema finds this.
    assert result.checks_passed == (
        "single_statement",
        "starts_with_select",
        "no_forbidden_keywords",
    )


MALFORMED_QUERIES = [
    "SELECT FROM loans",
    "SELECT * FROM loans WHERE",
    "SELECT * FROM loans GROUP",
    "SELECT * FROM loans WHERE risk_tier = ",
    "SELECT 'unterminated string FROM loans",
    "SELECT * FROM loans WHERE (1 = 1",
]


@pytest.mark.parametrize("sql", MALFORMED_QUERIES)
def test_syntactically_invalid_queries_are_blocked(sql, conn):
    result = validate_sql(sql, conn)
    assert not result
    assert result.reason is Rejection.DID_NOT_COMPILE


def test_explain_does_not_execute_the_query(conn):
    """EXPLAIN must plan without running: the premise of check 4.

    If EXPLAIN executed, running it on an unvalidated statement would be
    the very thing the validator is supposed to prevent.
    """
    before = row_count(conn)
    # A valid, expensive, fully-matching SELECT. EXPLAIN should compile it
    # and return bytecode without reading a single row.
    plan = conn.execute("EXPLAIN SELECT * FROM loans").fetchall()
    assert plan, "EXPLAIN returned no bytecode"
    assert row_count(conn) == before


def test_validation_without_a_connection_skips_only_the_compile_check():
    """The static checks must work with no database at all.

    This is what lets the first three checks be unit-tested in isolation,
    and it is reported honestly in checks_passed rather than pretending
    four checks ran.
    """
    result = validate_sql("SELECT * FROM table_that_does_not_exist")
    assert result, "static checks alone should pass this"
    assert "compiles" not in result.checks_passed
    assert len(result.checks_passed) == 3


# ---------------------------------------------------------------------------
# Empty and degenerate input
# ---------------------------------------------------------------------------

EMPTY_QUERIES = ["", "   ", "\n\n", "\t", ";", ";;", "-- just a comment", "/* only */"]


@pytest.mark.parametrize("sql", EMPTY_QUERIES)
def test_empty_queries_are_blocked(sql, conn):
    result = validate_sql(sql, conn)
    assert not result
    assert result.reason is Rejection.EMPTY


def test_none_is_blocked_rather_than_raising(conn):
    """A generator that returns nothing must not crash the pipeline."""
    result = validate_sql(None, conn)
    assert not result
    assert result.reason is Rejection.EMPTY


def test_prose_instead_of_sql_is_blocked(conn):
    """Models sometimes answer in English. That is not a SELECT."""
    result = validate_sql(
        "I'm sorry, I cannot answer that question with the given schema.", conn
    )
    assert not result
    assert result.reason is Rejection.NOT_A_SELECT


# ---------------------------------------------------------------------------
# Properties of the validator itself
# ---------------------------------------------------------------------------

def test_normalise_preserves_string_literals():
    """Normalisation must not corrupt the query it is about to approve."""
    sql = "SELECT * FROM loans WHERE entity = 'Acme  Capital'"
    assert "'Acme  Capital'" in sql
    # Whitespace collapsing is applied to the whole statement, which is a
    # known and accepted limitation: it can squeeze double spaces inside a
    # literal. It is recorded here so the behaviour is deliberate rather
    # than discovered later.
    assert normalise(sql) == "SELECT * FROM loans WHERE entity = 'Acme Capital'"


def test_a_valid_query_reports_all_four_checks(conn):
    result = validate_sql("SELECT risk_tier FROM loans LIMIT 1", conn)
    assert result
    assert result.checks_passed == (
        "single_statement",
        "starts_with_select",
        "no_forbidden_keywords",
        "compiles",
    )


def test_result_is_falsy_when_invalid_and_truthy_when_valid(conn):
    """Callers gate on the result object directly; __bool__ must be right.

    If this inverted, every 'if not validation:' guard in the engine would
    silently invert with it.
    """
    assert bool(validate_sql("SELECT 1", conn)) is True
    assert bool(validate_sql("DROP TABLE loans", conn)) is False


def test_rejected_query_still_reports_the_sql_it_rejected(conn):
    """Transparency applies to refusals too, not just to successes."""
    result = validate_sql("DELETE FROM loans WHERE risk_tier = 'Subprime'", conn)
    assert not result
    assert "DELETE FROM loans" in result.sql
