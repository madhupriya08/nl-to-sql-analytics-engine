"""Warehouse dialects: one interface, several engines.

The engine sits behind this interface so that swapping DuckDB for
Snowflake is a configuration change, not a rewrite. Everything upstream --
retrieval, the agent loop, the safety validator, the API -- talks to a
``Dialect`` and never imports ``duckdb`` or ``snowflake`` directly.

What actually differs between warehouses is smaller than it looks, and
this interface names exactly those differences:

* **connection** -- a file path versus a set of account credentials;
* **identifier quoting** -- DuckDB and Snowflake both use double quotes,
  but Snowflake folds unquoted identifiers to UPPER while DuckDB folds to
  lower, which changes what comes back in result-set keys;
* **type names** -- mostly shared, but ``DOUBLE`` is ``FLOAT`` on
  Snowflake and ``TIMESTAMP`` has different precision defaults;
* **dry-run semantics** -- the validator's fourth check needs a way to
  compile a statement *without running it*. DuckDB has ``EXPLAIN``;
  Snowflake has ``EXPLAIN`` too but it will not plan every statement, so
  the dialect owns that decision rather than the validator guessing;
* **forbidden keywords** -- each engine has its own escape hatches worth
  refusing (Snowflake's ``COPY INTO``, ``PUT``, ``GET``, ``UNDROP``).

Getting that list right is most of what a "supports multiple warehouses"
claim actually means.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import schema_def as S


@dataclass(frozen=True)
class ColumnMeta:
    """A column as the *live* warehouse reports it, not as we declared it.

    Introspection returns this rather than the declaration, because the
    point of introspecting is to catch the case where the warehouse and
    the declaration disagree.
    """

    name: str
    type: str
    nullable: bool


class Dialect(ABC):
    """What the rest of the system needs from a warehouse."""

    #: Short identifier used in config, logs and the UI.
    name: str = "abstract"

    #: Engine-specific verbs the safety validator must refuse on top of the
    #: shared SQL ones. Owned here because only the dialect knows them.
    extra_forbidden_keywords: frozenset[str] = frozenset()

    # -- connection ------------------------------------------------------

    @abstractmethod
    def connect(self) -> Any:
        """Open a connection. Caller owns closing it."""

    @abstractmethod
    def close(self, connection: Any) -> None: ...

    # -- execution -------------------------------------------------------

    @abstractmethod
    def execute(
        self, connection: Any, sql: str, max_rows: int | None = 1_000
    ) -> list[dict[str, Any]]:
        """Run an *already validated* SELECT and return rows as dicts.

        This is the single audited execution path per engine. It performs
        no validation of its own: validation lives in one module and the
        agent is responsible for calling it first.
        """

    @abstractmethod
    def dry_run(self, connection: Any, sql: str) -> None:
        """Compile ``sql`` without executing it; raise if it will not run.

        The validator's fourth check. Must resolve every table and column
        reference against the live warehouse, so hallucinated identifiers
        are caught here rather than at execution time.
        """

    # -- introspection ---------------------------------------------------

    @abstractmethod
    def list_tables(self, connection: Any) -> list[str]: ...

    @abstractmethod
    def describe_table(self, connection: Any, table: str) -> list[ColumnMeta]: ...

    # -- DDL / DML generation --------------------------------------------

    def quote(self, identifier: str) -> str:
        """Quote an identifier. Both engines use double quotes."""
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def type_for(self, declared: str) -> str:
        """Map a declared type to this engine's spelling."""
        return declared

    def create_table_sql(self, table: S.Table) -> str:
        columns = []
        for column in table.columns:
            parts = [self.quote(column.name), self.type_for(column.type)]
            if column.primary_key:
                parts.append("PRIMARY KEY")
            columns.append("  " + " ".join(parts))
        body = ",\n".join(columns)
        return f"CREATE TABLE IF NOT EXISTS {self.quote(table.name)} (\n{body}\n)"

    def drop_table_sql(self, table: S.Table) -> str:
        return f"DROP TABLE IF EXISTS {self.quote(table.name)}"

    def insert_sql(self, table: S.Table) -> str:
        """Positional-parameter INSERT. Both engines use ``?``."""
        columns = ", ".join(self.quote(c.name) for c in table.columns)
        placeholders = ", ".join("?" for _ in table.columns)
        return f"INSERT INTO {self.quote(table.name)} ({columns}) VALUES ({placeholders})"

    @abstractmethod
    def bulk_insert(
        self, connection: Any, table: S.Table, rows: Sequence[dict[str, Any]]
    ) -> int:
        """Load rows. Each engine has a much faster path than row-by-row."""

    def count_rows(self, connection: Any, table: str) -> int:
        result = self.execute(connection, f"SELECT COUNT(*) AS n FROM {self.quote(table)}", max_rows=1)
        return int(next(iter(result[0].values())))


class DuckDBDialect(Dialect):
    """Local analytical warehouse in a single file.

    DuckDB is the offline default for a specific reason: it is columnar
    and speaks close-to-standard analytical SQL, so a query that works
    here is far more likely to work on Snowflake than one written against
    SQLite. Window functions, ``QUALIFY``, ``GROUP BY ALL`` and CTEs all
    behave the same way. Using SQLite locally and Snowflake in production
    would mean the local tests exercise a dialect nobody deploys.
    """

    name = "duckdb"

    #: DuckDB can read and write the local filesystem and install
    #: extensions, so those verbs are refused on top of the shared set.
    extra_forbidden_keywords = frozenset({
        "INSTALL", "LOAD", "ATTACH", "DETACH", "EXPORT", "IMPORT", "COPY",
        "SET", "RESET", "CHECKPOINT", "FORCE",
    })

    def __init__(self, database_path: str | Path = "data/mobility.duckdb", read_only: bool = False):
        self.database_path = Path(database_path)
        self.read_only = read_only

    def connect(self) -> Any:
        import duckdb

        if self.database_path.parent != Path(""):
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.database_path), read_only=self.read_only)

    def close(self, connection: Any) -> None:
        connection.close()

    def execute(self, connection: Any, sql: str, max_rows: int | None = 1_000) -> list[dict[str, Any]]:
        cursor = connection.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows) if max_rows else cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def dry_run(self, connection: Any, sql: str) -> None:
        """``EXPLAIN`` compiles and plans without producing rows."""
        connection.execute(f"EXPLAIN {sql}")

    def list_tables(self, connection: Any) -> list[str]:
        rows = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        return [row[0] for row in rows]

    def describe_table(self, connection: Any, table: str) -> list[ColumnMeta]:
        rows = connection.execute(
            "SELECT column_name, data_type, is_nullable "
            "FROM information_schema.columns WHERE table_name = ? "
            "ORDER BY ordinal_position",
            [table],
        ).fetchall()
        return [ColumnMeta(name=r[0], type=r[1], nullable=r[2] == "YES") for r in rows]

    def bulk_insert(self, connection: Any, table: S.Table, rows: Sequence[dict[str, Any]]) -> int:
        """Load through a staged CSV, not row-by-row.

        The first version of this used ``executemany``, which is the
        obvious choice and was unusably slow -- 378,000 rows had not
        finished after two minutes, because each row round-trips through
        the prepared-statement binder individually.

        DuckDB's CSV reader is vectorised and reads the same data in about
        a second. Staging to a file and issuing one ``COPY`` is also the
        shape every real warehouse load takes -- it is precisely what
        Snowflake's ``PUT`` + ``COPY INTO`` does -- so the local path
        teaches the same pattern rather than a toy one.

        The ``COPY`` here is loader-internal SQL built from the
        declaration, never model output. The safety validator gates what
        the *agent* generates; it was never meant to constrain the
        pipeline that builds the warehouse in the first place.
        """
        if not rows:
            return 0

        import csv
        import tempfile

        names = [c.name for c in table.columns]
        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", newline="", delete=False, encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name) for name in names})
            staged = handle.name

        try:
            columns = ", ".join(self.quote(name) for name in names)
            connection.execute(
                f"COPY {self.quote(table.name)} ({columns}) "
                f"FROM '{staged}' (FORMAT CSV, HEADER, NULLSTR '')"
            )
        finally:
            Path(staged).unlink(missing_ok=True)
        return len(rows)


class SnowflakeDialect(Dialect):
    """Snowflake, configured entirely from the environment.

    NOT EXECUTED in the environment this was written in -- there were no
    warehouse credentials available. The SQL, the connector calls and the
    introspection queries are written against Snowflake's documented
    behaviour, and every dialect-level decision is exercised by the shared
    dialect tests, but the live path is unverified. Treat a first run
    against a real account as debugging, not as a smoke test, and see the
    README's scope note.

    Two Snowflake-specific behaviours are handled here rather than leaking
    upward:

    * **Identifier casing.** Snowflake folds unquoted identifiers to
      UPPERCASE. Our tables are declared lowercase, so every identifier is
      quoted -- otherwise ``SELECT trip_id FROM trips`` resolves to
      ``TRIP_ID`` and fails against a lowercase column.
    * **Result-set keys.** The connector returns column names as written
      in the query. Quoting throughout keeps them lowercase, so a caller
      can index results the same way on both engines.
    """

    name = "snowflake"

    #: Snowflake's stage and file verbs reach outside the warehouse:
    #: ``COPY INTO`` reads and writes cloud storage, ``PUT``/``GET`` move
    #: local files, ``UNDROP`` resurrects dropped objects, and ``USE``
    #: silently changes which database a later statement resolves against.
    extra_forbidden_keywords = frozenset({
        "COPY", "PUT", "GET", "REMOVE", "LIST", "UNDROP", "USE", "CALL",
        "MERGE", "STAGE", "UNLOAD", "SNOWPIPE",
    })

    #: Snowflake spells a few of the declared types differently.
    TYPE_MAP = {
        "DOUBLE": "FLOAT",
        "VARCHAR": "VARCHAR",
        "INTEGER": "NUMBER(38,0)",
        "BOOLEAN": "BOOLEAN",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP_NTZ",
    }

    REQUIRED_SETTINGS = (
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_USER",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
    )

    def __init__(self, **overrides: str):
        self.settings = {key: os.environ.get(key, "") for key in self.REQUIRED_SETTINGS}
        self.settings["SNOWFLAKE_ROLE"] = os.environ.get("SNOWFLAKE_ROLE", "")
        self.settings["SNOWFLAKE_PASSWORD"] = os.environ.get("SNOWFLAKE_PASSWORD", "")
        self.settings["SNOWFLAKE_PRIVATE_KEY_PATH"] = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH", "")
        self.settings.update(overrides)

    def missing_settings(self) -> list[str]:
        """Which required settings are absent.

        Reported as a list rather than raising on the first one, so a user
        configuring this sees everything they still need in one go instead
        of discovering them one failed run at a time.
        """
        missing = [key for key in self.REQUIRED_SETTINGS if not self.settings.get(key)]
        if not self.settings.get("SNOWFLAKE_PASSWORD") and not self.settings.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
            missing.append("SNOWFLAKE_PASSWORD or SNOWFLAKE_PRIVATE_KEY_PATH")
        return missing

    def is_configured(self) -> bool:
        return not self.missing_settings()

    def connect(self) -> Any:
        try:
            import snowflake.connector
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "snowflake-connector-python is required for the Snowflake dialect; "
                "install it with `pip install 'snowflake-connector-python'`"
            ) from exc

        missing = self.missing_settings()
        if missing:
            raise RuntimeError(
                "Snowflake is not configured. Missing: " + ", ".join(missing)
            )

        kwargs: dict[str, Any] = {
            "account": self.settings["SNOWFLAKE_ACCOUNT"],
            "user": self.settings["SNOWFLAKE_USER"],
            "warehouse": self.settings["SNOWFLAKE_WAREHOUSE"],
            "database": self.settings["SNOWFLAKE_DATABASE"],
            "schema": self.settings["SNOWFLAKE_SCHEMA"],
        }
        if self.settings.get("SNOWFLAKE_ROLE"):
            kwargs["role"] = self.settings["SNOWFLAKE_ROLE"]

        # Key-pair auth is the path that survives Snowflake's MFA
        # enforcement on password logins, so it is supported alongside.
        key_path = self.settings.get("SNOWFLAKE_PRIVATE_KEY_PATH")
        if key_path:
            kwargs["private_key_file"] = key_path
        else:
            kwargs["password"] = self.settings["SNOWFLAKE_PASSWORD"]

        return snowflake.connector.connect(**kwargs)

    def close(self, connection: Any) -> None:
        connection.close()

    def type_for(self, declared: str) -> str:
        return self.TYPE_MAP.get(declared.upper(), declared)

    def execute(self, connection: Any, sql: str, max_rows: int | None = 1_000) -> list[dict[str, Any]]:
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows) if max_rows else cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        finally:
            cursor.close()

    def dry_run(self, connection: Any, sql: str) -> None:
        """Compile without executing.

        Snowflake's ``EXPLAIN`` parses and compiles the statement and
        resolves identifiers, without running it or consuming warehouse
        compute for the result. That is exactly the property the
        validator's fourth check needs.
        """
        cursor = connection.cursor()
        try:
            cursor.execute(f"EXPLAIN {sql}")
        finally:
            cursor.close()

    def list_tables(self, connection: Any) -> list[str]:
        rows = self.execute(
            connection,
            "SELECT LOWER(table_name) AS table_name FROM information_schema.tables "
            f"WHERE table_schema = UPPER('{self.settings['SNOWFLAKE_SCHEMA']}') "
            "AND table_type = 'BASE TABLE' ORDER BY 1",
            max_rows=None,
        )
        return [row["TABLE_NAME"] if "TABLE_NAME" in row else row["table_name"] for row in rows]

    def describe_table(self, connection: Any, table: str) -> list[ColumnMeta]:
        rows = self.execute(
            connection,
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            f"WHERE table_schema = UPPER('{self.settings['SNOWFLAKE_SCHEMA']}') "
            f"AND LOWER(table_name) = LOWER('{table}') ORDER BY ordinal_position",
            max_rows=None,
        )
        out = []
        for row in rows:
            values = {k.lower(): v for k, v in row.items()}
            out.append(
                ColumnMeta(
                    name=str(values["column_name"]),
                    type=str(values["data_type"]),
                    nullable=str(values["is_nullable"]).upper() == "YES",
                )
            )
        return out

    def insert_sql(self, table: S.Table) -> str:
        """Snowflake's connector uses ``%s`` placeholders, not ``?``."""
        columns = ", ".join(self.quote(c.name) for c in table.columns)
        placeholders = ", ".join("%s" for _ in table.columns)
        return f"INSERT INTO {self.quote(table.name)} ({columns}) VALUES ({placeholders})"

    def bulk_insert(self, connection: Any, table: S.Table, rows: Sequence[dict[str, Any]]) -> int:
        """Batched ``executemany``.

        The genuinely fast path on Snowflake is ``PUT`` a Parquet file to
        a stage then ``COPY INTO`` -- but those are exactly the verbs the
        validator refuses, and wiring a privileged bypass around the
        safety layer for the loader would undermine the guarantee the
        project exists to make. Batched inserts are slower and keep one
        rule for everyone. Airflow does this load out of band anyway, so
        the latency is not on any user's path.
        """
        if not rows:
            return 0
        names = [c.name for c in table.columns]
        payload = [tuple(row.get(name) for name in names) for row in rows]
        cursor = connection.cursor()
        try:
            batch = 10_000
            for start in range(0, len(payload), batch):
                cursor.executemany(self.insert_sql(table), payload[start:start + batch])
            connection.commit()
        finally:
            cursor.close()
        return len(payload)


def get_dialect(name: str | None = None, **kwargs: Any) -> Dialect:
    """Build a dialect by name, defaulting to the ``WAREHOUSE`` env var.

    One factory so the engine, the CLI, the API and the Airflow DAGs all
    resolve the warehouse the same way.
    """
    resolved = (name or os.environ.get("WAREHOUSE") or "duckdb").lower()
    if resolved == "duckdb":
        return DuckDBDialect(**kwargs)
    if resolved == "snowflake":
        return SnowflakeDialect(**kwargs)
    raise ValueError(f"unknown warehouse {resolved!r}; expected 'duckdb' or 'snowflake'")


DIALECTS: dict[str, type[Dialect]] = {
    "duckdb": DuckDBDialect,
    "snowflake": SnowflakeDialect,
}
