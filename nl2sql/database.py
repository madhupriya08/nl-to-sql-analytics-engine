"""SQLite loader and the single read-only execution path.

Two responsibilities live here, and they are kept together on purpose:
building the database, and being the *only* place in the codebase that
calls ``cursor.execute()`` on a generated query. Concentrating execution
in one function means there is exactly one line to audit when asking
"can unvalidated SQL reach the database?".
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from .synthetic_data import DEFAULT_ROW_COUNT, DEFAULT_SEED, generate_rows

TABLE_NAME = "loans"

DEFAULT_DB_PATH = Path("data/loans.db")

#: The loan book. REAL for money/rates rather than NUMERIC so aggregate
#: functions behave predictably; dates as TEXT in ISO-8601 because SQLite
#: has no date type and ISO strings sort and compare correctly as text.
CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    loan_id          TEXT    PRIMARY KEY,
    entity           TEXT    NOT NULL,
    product          TEXT    NOT NULL,
    risk_tier        TEXT    NOT NULL,
    loan_amount      REAL    NOT NULL,
    interest_rate    REAL    NOT NULL,
    origination_date TEXT    NOT NULL,
    default_flag     INTEGER NOT NULL CHECK (default_flag IN (0, 1))
)
"""

#: Indexes on the columns natural-language questions actually filter and
#: group on. "Loans by risk tier", "defaults by product", "originations in
#: 2023" are the shapes people ask for, so those are the columns indexed.
#: The two composite indexes cover the most common two-dimensional
#: breakdowns so those queries are answered from the index alone.
INDEX_SQL: tuple[str, ...] = (
    f"CREATE INDEX IF NOT EXISTS idx_loans_risk_tier ON {TABLE_NAME}(risk_tier)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_product ON {TABLE_NAME}(product)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_entity ON {TABLE_NAME}(entity)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_default_flag ON {TABLE_NAME}(default_flag)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_origination_date ON {TABLE_NAME}(origination_date)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_tier_product ON {TABLE_NAME}(risk_tier, product)",
    f"CREATE INDEX IF NOT EXISTS idx_loans_tier_default ON {TABLE_NAME}(risk_tier, default_flag)",
)

INSERT_SQL = f"""
INSERT OR REPLACE INTO {TABLE_NAME} (
    loan_id, entity, product, risk_tier,
    loan_amount, interest_rate, origination_date, default_flag
) VALUES (
    :loan_id, :entity, :product, :risk_tier,
    :loan_amount, :interest_rate, :origination_date, :default_flag
)
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a connection with ``sqlite3.Row`` so results carry column names.

    Results are returned to the caller as dicts keyed by column name, which
    means the engine never has to guess what the model aliased a column to.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def build_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
) -> Path:
    """Create the loans table, its indexes, and load synthetic rows.

    Returns the path to the database. Idempotent: re-running with the same
    seed produces byte-for-byte identical row data.
    """
    path = Path(db_path)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()

    with connect(path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        for statement in INDEX_SQL:
            conn.execute(statement)
        conn.executemany(INSERT_SQL, generate_rows(row_count=row_count, seed=seed))
        conn.commit()
    return path


def ensure_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    row_count: int = DEFAULT_ROW_COUNT,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Build the database only if it does not already exist or is empty."""
    path = Path(db_path)
    if path.exists():
        with connect(path) as conn:
            try:
                existing = conn.execute(
                    f"SELECT COUNT(*) FROM {TABLE_NAME}"
                ).fetchone()[0]
            except sqlite3.Error:
                existing = 0
        if existing:
            return path
    return build_database(path, row_count=row_count, seed=seed)


def row_count(conn: sqlite3.Connection, table: str = TABLE_NAME) -> int:
    """Count rows in ``table``.

    The adversarial tests use this before and after a blocked destructive
    query to prove nothing was mutated. ``table`` is interpolated rather
    than bound because SQLite cannot parameterise identifiers -- it is
    never fed user or model input, only module constants.
    """
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def execute_select(
    conn: sqlite3.Connection, sql: str, max_rows: int | None = 1_000
) -> list[dict[str, Any]]:
    """Execute an *already validated* SELECT and return rows as dicts.

    This is the only ``execute()`` of model-generated SQL in the codebase.
    It performs no validation of its own by design: validation lives in one
    place (:mod:`nl2sql.validator`) and the engine is responsible for
    calling it first. Two mechanisms enforce that rather than trusting a
    convention:

    * ``executescript`` is never used, so even a chained statement that
      somehow got here would raise ``sqlite3.Warning`` instead of running
      both halves;
    * ``max_rows`` caps materialisation, so a query that passes validation
      but matches everything cannot exhaust memory.
    """
    cursor = conn.execute(sql)
    rows = cursor.fetchmany(max_rows) if max_rows else cursor.fetchall()
    return [dict(row) for row in rows]


def explain(conn: sqlite3.Connection, sql: str) -> Sequence[Any]:
    """Run ``EXPLAIN <sql>``: plan the query without executing it.

    SQLite compiles the statement to bytecode and returns that program.
    Nothing in the statement runs, so this is safe to call on SQL that has
    not yet been proven harmless -- and because compilation resolves every
    table and column reference, it rejects hallucinated names that no
    amount of string inspection would catch.
    """
    return conn.execute(f"EXPLAIN {sql}").fetchall()


def iter_tables(conn: sqlite3.Connection) -> Iterable[str]:
    """Yield user table names, skipping SQLite's internal ``sqlite_*`` tables."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [row["name"] for row in rows]
