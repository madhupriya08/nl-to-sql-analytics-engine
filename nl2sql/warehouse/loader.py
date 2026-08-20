"""Build the warehouse: create tables, load rows, index, verify.

Engine-agnostic. Everything here goes through a :class:`Dialect`, so the
same function builds a local DuckDB file or a Snowflake schema depending
only on what it is handed. Airflow calls this; so does the CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import schema_def as S
from .dialects import Dialect, DuckDBDialect, get_dialect
from .synthetic import DEFAULT_SEED, generate_all


@dataclass
class LoadReport:
    """What a build actually did, per table.

    Returned rather than printed so Airflow can push it to XCom and the
    CLI can render it. A load that silently half-succeeds is the failure
    mode worth guarding against, so the row counts are read back from the
    warehouse after loading rather than taken from what was sent.
    """

    dialect: str
    rows_by_table: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    verified: bool = False
    discrepancies: list[str] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(self.rows_by_table.values())


def build_warehouse(
    dialect: Dialect | None = None,
    seed: int = DEFAULT_SEED,
    scale: dict[str, int] | None = None,
    drop_existing: bool = True,
    progress: Callable[[str], None] | None = None,
) -> LoadReport:
    """Create every table and load generated rows into it.

    ``drop_existing`` defaults to True because this is a demo warehouse
    rebuilt from a deterministic seed -- there is no state worth
    preserving, and an append onto an existing table would silently double
    every count and quietly corrupt every eval assertion.
    """
    dialect = dialect or get_dialect()
    say = progress or (lambda _message: None)
    started = time.time()
    report = LoadReport(dialect=dialect.name)

    say(f"generating synthetic data (seed={seed})")
    data = generate_all(seed=seed, scale=scale)

    connection = dialect.connect()
    try:
        for table in S.TABLES:
            rows = data.get(table.name, [])
            if drop_existing:
                dialect.execute(connection, dialect.drop_table_sql(table), max_rows=None)
            dialect.execute(connection, dialect.create_table_sql(table), max_rows=None)
            loaded = dialect.bulk_insert(connection, table, rows)
            report.rows_by_table[table.name] = loaded
            say(f"  {table.name:24} {loaded:>8,} rows")

        say("verifying row counts against the warehouse")
        report.discrepancies = _verify(dialect, connection, report)
        report.verified = not report.discrepancies
    finally:
        dialect.close(connection)

    report.seconds = round(time.time() - started, 2)
    return report


def _verify(dialect: Dialect, connection: Any, report: LoadReport) -> list[str]:
    """Read counts back from the warehouse and compare to what was sent.

    Trusting the loader's own arithmetic would miss the interesting
    failures: a silently rejected batch, a type coercion that dropped
    rows, a transaction that never committed. Counting from the other
    side is the only check that catches those.
    """
    problems: list[str] = []
    for table_name, expected in report.rows_by_table.items():
        actual = dialect.count_rows(connection, table_name)
        if actual != expected:
            problems.append(f"{table_name}: loaded {expected:,} but warehouse holds {actual:,}")
    return problems


#: Indexes for the columns questions actually filter and group on. DuckDB
#: is columnar and needs far fewer indexes than a row store -- it prunes
#: with zone maps automatically -- so these target only the high-selectivity
#: foreign keys used in joins, where an index still pays. Snowflake needs
#: none of this at all (micro-partitions handle it), which is why index
#: creation is skipped for any dialect that is not DuckDB.
DUCKDB_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("idx_legs_trip", "trip_legs", "trip_id"),
    ("idx_legs_mode", "trip_legs", "mode"),
    ("idx_legs_route", "trip_legs", "route_id"),
    ("idx_trips_started", "trips", "started_at"),
    ("idx_trips_origin", "trips", "origin_zone_id"),
    ("idx_trips_rider", "trips", "rider_id"),
    ("idx_payments_trip", "payments", "trip_id"),
    ("idx_feedback_trip", "feedback", "trip_id"),
    ("idx_weather_zone_time", "weather_hourly", "zone_id"),
    ("idx_aq_zone_time", "air_quality_hourly", "zone_id"),
    ("idx_snapshots_station", "station_snapshots", "station_id"),
    ("idx_maint_vehicle", "maintenance_events", "vehicle_id"),
)


def create_indexes(dialect: Dialect, connection: Any) -> int:
    """Create the DuckDB indexes; a no-op on engines that manage their own."""
    if not isinstance(dialect, DuckDBDialect):
        return 0
    created = 0
    for index_name, table, column in DUCKDB_INDEXES:
        connection.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} "
            f"ON {dialect.quote(table)} ({dialect.quote(column)})"
        )
        created += 1
    return created


def introspect(dialect: Dialect, connection: Any) -> dict[str, list[str]]:
    """Read the live schema back, as ``{table: [column, ...]}``.

    Used by the drift check below and by anything that wants to see the
    warehouse as it really is rather than as it was declared.
    """
    return {
        table: [c.name for c in dialect.describe_table(connection, table)]
        for table in dialect.list_tables(connection)
    }


def check_drift(dialect: Dialect, connection: Any) -> list[str]:
    """Compare the live warehouse against the declaration.

    The declaration drives retrieval: descriptions are embedded from it,
    and the model is told those columns exist. If the warehouse and the
    declaration disagree, retrieval confidently hands the model a column
    that is not there -- which surfaces as a "the model hallucinated"
    complaint when it is really a schema-drift bug. This is the check that
    tells the two apart, and Airflow runs it after every load.
    """
    live = introspect(dialect, connection)
    problems: list[str] = []

    for table in S.TABLES:
        if table.name not in live:
            problems.append(f"declared table {table.name!r} is missing from the warehouse")
            continue
        live_columns = {name.lower() for name in live[table.name]}
        for column in table.columns:
            if column.name.lower() not in live_columns:
                problems.append(f"{table.name}.{column.name} is declared but not in the warehouse")
        declared = {c.name.lower() for c in table.columns}
        for name in live_columns - declared:
            problems.append(f"{table.name}.{name} exists in the warehouse but is not declared")

    for table_name in set(live) - {t.name for t in S.TABLES}:
        problems.append(f"table {table_name!r} exists in the warehouse but is not declared")

    return problems
