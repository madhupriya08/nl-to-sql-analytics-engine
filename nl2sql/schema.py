"""Live schema introspection: the grounding step.

The prompt's description of the database is read from the database on
every call, never hardcoded. A hardcoded schema string is a second source
of truth that drifts the moment someone adds a column, and the failure
mode is quiet: the model keeps generating SQL against the schema it was
told about, and the errors surface as 'no such column' at execution time
-- or worse, as answers computed from the wrong column.

Introspection also does something a handwritten description usually will
not: it samples the *actual values* in TEXT columns. Knowing a column is
called ``risk_tier`` does not tell the model whether the value is 'Prime',
'prime' or 'PRIME'. SQLite compares TEXT with ``=`` case-sensitively, so
guessing wrong returns zero rows and no error -- a wrong answer that looks
like a real one. Showing the model the literal values removes the guess.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

#: How many distinct values to sample per TEXT column. Enough to pin down
#: casing and vocabulary for a low-cardinality dimension; capped so a
#: high-cardinality column (loan_id, 5,000 distinct values) cannot flood
#: the prompt. The cap is also why the count is reported alongside: the
#: model needs to know whether it is seeing all values or a sample.
MAX_SAMPLE_VALUES = 12

#: Columns whose sampled values would be noise rather than grounding.
#: Primary keys and free-text identifiers have no shared vocabulary to
#: learn, and listing twelve arbitrary loan_ids only wastes tokens.
SAMPLE_SKIP_SUFFIXES = ("_id",)


@dataclass(frozen=True)
class ColumnInfo:
    """One column, as PRAGMA table_info reports it."""

    name: str
    type: str
    not_null: bool
    primary_key: bool
    #: The column's complete set of distinct values, when it is small
    #: enough to enumerate. Empty for high-cardinality columns.
    sample_values: tuple[str, ...] = ()
    #: Total distinct count, so an enumeration can be labelled complete.
    distinct_count: int | None = None
    #: (min, max) for high-cardinality columns, where the useful grounding
    #: is the range rather than the values.
    value_range: tuple[str, str] | None = None


@dataclass(frozen=True)
class TableInfo:
    """One table and its columns."""

    name: str
    columns: tuple[ColumnInfo, ...] = field(default_factory=tuple)
    row_count: int = 0


def _should_profile(column: ColumnInfo) -> bool:
    """Whether a column is worth profiling at all.

    Primary keys and ``*_id`` columns are skipped: they are unique by
    construction, so neither their values nor their range teach the model
    anything, and enumerating 5,000 loan_ids would swamp the prompt.
    """
    if column.primary_key:
        return False
    return not column.name.lower().endswith(SAMPLE_SKIP_SUFFIXES)


def _profile_column(
    conn: sqlite3.Connection, table: str, column: str, limit: int = MAX_SAMPLE_VALUES
) -> tuple[tuple[str, ...], int, tuple[str, str] | None]:
    """Profile a column as either an enumeration or a range.

    Which one depends on cardinality, because the two carry different
    information:

    * Low cardinality (<= ``limit`` distinct) -- enumerate every value.
      This is the case that matters most. It pins down exact casing and
      spelling, so the model writes ``risk_tier = 'Near-Prime'`` rather
      than ``'near prime'``, which SQLite would match against nothing and
      report as an empty result rather than an error.
    * High cardinality -- report ``MIN``/``MAX`` instead. Twelve arbitrary
      values from a date or amount column teach nothing, but the range
      tells the model what period the book covers, so it can answer 'last
      year' correctly and recognise a question about data that is not there.

    Identifiers are interpolated rather than bound because SQLite cannot
    parameterise identifiers. They come from PRAGMA output -- the
    database's own metadata, never from the model or the user -- and are
    double-quoted so unusual names cannot break the query.
    """
    quoted_table = f'"{table}"'
    quoted_column = f'"{column}"'
    distinct_count = int(
        conn.execute(
            f"SELECT COUNT(DISTINCT {quoted_column}) FROM {quoted_table}"
        ).fetchone()[0]
    )

    if distinct_count <= limit:
        rows = conn.execute(
            f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL "
            f"ORDER BY {quoted_column}"
        ).fetchall()
        return tuple(str(row[0]) for row in rows), distinct_count, None

    bounds = conn.execute(
        f"SELECT MIN({quoted_column}), MAX({quoted_column}) FROM {quoted_table}"
    ).fetchone()
    if bounds is None or bounds[0] is None:
        return (), distinct_count, None
    return (), distinct_count, (str(bounds[0]), str(bounds[1]))


def introspect_table(conn: sqlite3.Connection, table: str) -> TableInfo:
    """Read one table's live structure via ``PRAGMA table_info``."""
    pragma_rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not pragma_rows:
        raise ValueError(f"table {table!r} does not exist in this database")

    count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])

    columns: list[ColumnInfo] = []
    for row in pragma_rows:
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
        column = ColumnInfo(
            name=row["name"],
            type=row["type"],
            not_null=bool(row["notnull"]),
            primary_key=bool(row["pk"]),
        )
        if _should_profile(column):
            values, distinct, bounds = _profile_column(conn, table, column.name)
            column = ColumnInfo(
                name=column.name,
                type=column.type,
                not_null=column.not_null,
                primary_key=column.primary_key,
                sample_values=values,
                distinct_count=distinct,
                value_range=bounds,
            )
        columns.append(column)

    return TableInfo(name=table, columns=tuple(columns), row_count=count)


def introspect_database(conn: sqlite3.Connection) -> tuple[TableInfo, ...]:
    """Introspect every user table in the connected database."""
    from .database import iter_tables

    return tuple(introspect_table(conn, name) for name in iter_tables(conn))


def _format_column(column: ColumnInfo) -> str:
    parts = [f"  - {column.name} ({column.type or 'ANY'}"]
    flags = []
    if column.primary_key:
        flags.append("primary key")
    if column.not_null and not column.primary_key:
        flags.append("not null")
    parts.append(", " + ", ".join(flags) if flags else "")
    parts.append(")")
    line = "".join(parts)

    if column.sample_values:
        rendered = ", ".join(repr(value) for value in column.sample_values)
        # Naming this as the complete set matters: it tells the model that a
        # WHERE clause on any other value is guaranteed to return nothing,
        # so it should say the question is unanswerable rather than invent a
        # category that does not exist in the data.
        line += (
            f"\n      complete set of {len(column.sample_values)} "
            f"distinct values: {rendered}"
        )
    elif column.value_range is not None:
        low, high = column.value_range
        line += (
            f"\n      {column.distinct_count} distinct values, "
            f"ranging from {low!r} to {high!r}"
        )
    return line


def render_schema_context(tables: tuple[TableInfo, ...]) -> str:
    """Render introspected tables as the schema block sent to the model.

    Plain indented text rather than raw DDL or JSON: it is the most
    token-efficient form that still carries types and, critically, the
    literal column values.
    """
    blocks: list[str] = []
    for table in tables:
        header = f"Table: {table.name} ({table.row_count} rows)"
        column_lines = "\n".join(_format_column(c) for c in table.columns)
        blocks.append(f"{header}\n{column_lines}")
    return "\n\n".join(blocks)


def build_schema_context(conn: sqlite3.Connection) -> str:
    """Introspect the live database and render its schema for the prompt."""
    return render_schema_context(introspect_database(conn))
