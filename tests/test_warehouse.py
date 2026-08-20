"""Tests for the generator, the dialect layer and the loader.

The warehouse fixture is session-scoped and built at a reduced trip count:
the full build takes ~30 seconds, and nothing here needs 50,000 trips to
prove a point. The correlation tests use proportions rather than absolute
counts so they hold at any scale.
"""

from __future__ import annotations

import pytest

from nl2sql.warehouse import schema_def as S
from nl2sql.warehouse.dialects import (
    DuckDBDialect,
    SnowflakeDialect,
    get_dialect,
)
from nl2sql.warehouse.loader import build_warehouse, check_drift, create_indexes
from nl2sql.warehouse.synthetic import DEFAULT_SEED, generate_all

SMALL = {"trips": 4_000, "station_snapshots": 6_000, "charging_sessions": 2_000}


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory):
    """A real DuckDB warehouse, built once for the session."""
    path = tmp_path_factory.mktemp("warehouse") / "mobility.duckdb"
    dialect = DuckDBDialect(path)
    report = build_warehouse(dialect=dialect, scale=SMALL)
    connection = dialect.connect()
    create_indexes(dialect, connection)
    yield dialect, connection, report
    dialect.close(connection)


def query(connection, sql):
    cursor = connection.execute(sql)
    columns = [d[0] for d in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generation_is_deterministic():
    """Two runs at the same seed must be identical.

    Everything downstream depends on this: the eval harness asserts exact
    numbers, and a golden question's expected answer is only meaningful
    if the data behind it cannot move.
    """
    first = generate_all(seed=DEFAULT_SEED, scale=SMALL)
    second = generate_all(seed=DEFAULT_SEED, scale=SMALL)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name] == second[name], name


def test_a_different_seed_gives_different_data():
    a = generate_all(seed=1, scale=SMALL)
    b = generate_all(seed=2, scale=SMALL)
    assert a["trips"] != b["trips"]


def test_every_declared_table_is_generated():
    data = generate_all(scale=SMALL)
    assert set(data) == set(S.table_names())
    for name, rows in data.items():
        assert rows, f"{name} generated no rows"


def test_generated_rows_have_exactly_the_declared_columns():
    """A stray or missing key would fail at load time with a worse message."""
    data = generate_all(scale=SMALL)
    for table in S.TABLES:
        declared = {c.name for c in table.columns}
        actual = set(data[table.name][0])
        assert actual == declared, f"{table.name}: {actual ^ declared}"


def test_enum_columns_only_contain_declared_values():
    """The values the model is shown must be the values that exist.

    Retrieval surfaces a column's enum values to the model as the complete
    set. If the generator emitted something outside that set, the model
    would be told a filter is exhaustive when it is not.
    """
    data = generate_all(scale=SMALL)
    for table in S.TABLES:
        for column in table.columns:
            if not column.enum_values:
                continue
            seen = {row[column.name] for row in data[table.name] if row[column.name] is not None}
            unexpected = seen - set(column.enum_values)
            assert not unexpected, f"{table.name}.{column.name}: {unexpected}"


def test_foreign_key_values_resolve():
    """Every FK must point at a row that exists.

    Generated in one pass precisely so this holds; the test is what proves
    the one-pass approach worked.
    """
    data = generate_all(scale=SMALL)
    for table in S.TABLES:
        for column, target in table.foreign_keys:
            target_table, target_column = target.split(".", 1)
            valid = {row[target_column] for row in data[target_table]}
            for row in data[table.name]:
                value = row[column]
                if value is not None:
                    assert value in valid, f"{table.name}.{column}={value} has no {target}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_verifies_row_counts_from_the_warehouse(warehouse):
    _, _, report = warehouse
    assert report.verified
    assert report.discrepancies == []
    assert report.total_rows > 50_000


def test_no_schema_drift_after_load(warehouse):
    """The declaration and the live warehouse must agree exactly.

    This is the check that separates "the model hallucinated a column"
    from "the warehouse does not match what we told the model".
    """
    dialect, connection, _ = warehouse
    assert check_drift(dialect, connection) == []


def test_all_declared_tables_exist_in_the_warehouse(warehouse):
    dialect, connection, _ = warehouse
    assert set(dialect.list_tables(connection)) == set(S.table_names())


def test_introspection_reports_the_declared_columns(warehouse):
    dialect, connection, _ = warehouse
    for table in S.TABLES:
        live = {c.name.lower() for c in dialect.describe_table(connection, table.name)}
        assert live == {c.name.lower() for c in table.columns}, table.name


# ---------------------------------------------------------------------------
# The correlations -- what makes generated answers checkable by eye
# ---------------------------------------------------------------------------

def test_rain_shifts_riders_off_bikes(warehouse):
    """The headline correlation. If this flattens, the data is noise."""
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT CASE WHEN w.precipitation_mm > 0 THEN 'wet' ELSE 'dry' END AS weather,
               100.0 * SUM(CASE WHEN l.mode = 'bike' THEN 1 ELSE 0 END) / COUNT(*) AS bike_pct
        FROM trip_legs l
        JOIN trips t ON t.trip_id = l.trip_id
        JOIN weather_hourly w
          ON w.zone_id = t.origin_zone_id
         AND w.observed_at = date_trunc('hour', t.started_at)
        GROUP BY 1
    """)
    share = {r["weather"]: r["bike_pct"] for r in rows}
    assert share["dry"] > share["wet"] * 2, share


def test_delay_is_worse_at_peak(warehouse):
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT t.is_peak, AVG(l.delay_min) AS avg_delay
        FROM trip_legs l JOIN trips t ON t.trip_id = l.trip_id
        WHERE l.mode IN ('bus', 'metro') GROUP BY 1
    """)
    by_peak = {bool(r["is_peak"]): r["avg_delay"] for r in rows}
    assert by_peak[True] > by_peak[False]


def test_older_models_cost_more_to_maintain(warehouse):
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT vm.year_introduced < 2018 AS old_model, AVG(m.cost_eur) AS avg_cost
        FROM maintenance_events m
        JOIN vehicles v ON v.vehicle_id = m.vehicle_id
        JOIN vehicle_models vm ON vm.model_id = v.model_id
        GROUP BY 1
    """)
    by_age = {bool(r["old_model"]): r["avg_cost"] for r in rows}
    assert by_age[True] > by_age[False]


def test_delays_reduce_ratings(warehouse):
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT d.worst > 8 AS badly_delayed, AVG(f.rating) AS avg_rating
        FROM feedback f
        JOIN (SELECT trip_id, MAX(delay_min) AS worst FROM trip_legs GROUP BY 1) d
          ON d.trip_id = f.trip_id
        GROUP BY 1
    """)
    by_delay = {bool(r["badly_delayed"]): r["avg_rating"] for r in rows}
    assert by_delay[False] > by_delay[True]


def test_industrial_zones_have_worse_air(warehouse):
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT z.zone_type, AVG(a.pm25) AS avg_pm25
        FROM air_quality_hourly a JOIN zones z ON z.zone_id = a.zone_id
        GROUP BY 1 ORDER BY avg_pm25 DESC
    """)
    assert rows[0]["zone_type"] == "industrial"
    assert rows[-1]["zone_type"] == "parkland"


def test_demand_has_commuter_peaks(warehouse):
    """An hour-of-day query should show two humps, not a flat line."""
    _, connection, _ = warehouse
    rows = query(connection, """
        SELECT EXTRACT(hour FROM started_at) AS hour, COUNT(*) AS trips
        FROM trips GROUP BY 1 ORDER BY 1
    """)
    by_hour = {int(r["hour"]): r["trips"] for r in rows}
    assert by_hour[8] > by_hour[3] * 5
    assert by_hour[17] > by_hour[3] * 5


# ---------------------------------------------------------------------------
# Dialect layer
# ---------------------------------------------------------------------------

def test_duckdb_dry_run_does_not_execute(warehouse):
    """The premise of the validator's fourth check, on this engine."""
    dialect, connection, _ = warehouse
    before = dialect.count_rows(connection, "trips")
    dialect.dry_run(connection, "SELECT * FROM trips")
    assert dialect.count_rows(connection, "trips") == before


def test_duckdb_dry_run_rejects_hallucinated_identifiers(warehouse):
    dialect, connection, _ = warehouse
    with pytest.raises(Exception):
        dialect.dry_run(connection, "SELECT * FROM riders_that_do_not_exist")
    with pytest.raises(Exception):
        dialect.dry_run(connection, "SELECT no_such_column FROM trips")


def test_dialects_declare_their_own_forbidden_keywords():
    """Each engine's escape hatches differ; the dialect owns that list.

    Snowflake's stage verbs (COPY, PUT, GET) reach cloud storage; DuckDB's
    INSTALL and LOAD reach the local filesystem. A single shared list would
    either miss one engine's risks or block the other's valid SQL.
    """
    duck = DuckDBDialect()
    snow = SnowflakeDialect()
    assert "INSTALL" in duck.extra_forbidden_keywords
    assert "PUT" in snow.extra_forbidden_keywords
    assert "UNDROP" in snow.extra_forbidden_keywords
    assert duck.extra_forbidden_keywords != snow.extra_forbidden_keywords


def test_snowflake_maps_types_that_differ():
    snow = SnowflakeDialect()
    assert snow.type_for("DOUBLE") == "FLOAT"
    assert snow.type_for("TIMESTAMP") == "TIMESTAMP_NTZ"
    assert snow.type_for("BOOLEAN") == "BOOLEAN"


def test_snowflake_quotes_every_identifier():
    """Snowflake folds unquoted identifiers to UPPERCASE.

    Our tables are declared lowercase, so an unquoted reference would
    resolve to TRIPS and miss. Quoting throughout also keeps result-set
    keys lowercase, so callers index results identically on both engines.
    """
    snow = SnowflakeDialect()
    ddl = snow.create_table_sql(S.TRIPS)
    assert '"trips"' in ddl
    assert '"trip_id"' in ddl
    assert "FLOAT" in ddl  # DOUBLE mapped


def test_snowflake_reports_all_missing_settings_at_once(monkeypatch):
    """Configuring by trial and error is miserable; list everything."""
    for key in SnowflakeDialect.REQUIRED_SETTINGS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PRIVATE_KEY_PATH", raising=False)

    missing = SnowflakeDialect().missing_settings()
    assert set(SnowflakeDialect.REQUIRED_SETTINGS) <= set(missing)
    assert any("PASSWORD" in m for m in missing)
    assert not SnowflakeDialect().is_configured()


def test_snowflake_accepts_key_pair_instead_of_password(monkeypatch):
    """Key-pair auth is the path that survives Snowflake's MFA on passwords."""
    for key in SnowflakeDialect.REQUIRED_SETTINGS:
        monkeypatch.setenv(key, "x")
    monkeypatch.delenv("SNOWFLAKE_PASSWORD", raising=False)
    monkeypatch.setenv("SNOWFLAKE_PRIVATE_KEY_PATH", "/tmp/key.p8")
    assert SnowflakeDialect().is_configured()


def test_get_dialect_resolves_from_the_environment(monkeypatch):
    monkeypatch.delenv("WAREHOUSE", raising=False)
    assert get_dialect().name == "duckdb"
    monkeypatch.setenv("WAREHOUSE", "snowflake")
    assert get_dialect().name == "snowflake"
    with pytest.raises(ValueError):
        get_dialect("postgres")


def test_both_dialects_generate_ddl_for_every_table():
    """The interface has to actually cover the whole schema, not just trips."""
    for dialect in (DuckDBDialect(), SnowflakeDialect()):
        for table in S.TABLES:
            ddl = dialect.create_table_sql(table)
            assert ddl.startswith("CREATE TABLE IF NOT EXISTS")
            for column in table.columns:
                assert dialect.quote(column.name) in ddl
