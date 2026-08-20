"""``python -m nl2sql.warehouse.build`` -- build the warehouse from scratch.

Thin CLI over :func:`loader.build_warehouse`. Airflow calls the function
directly; this exists so a person can run the same thing by hand and see
what happened.
"""

from __future__ import annotations

import argparse
import sys

from . import schema_def as S
from .dialects import get_dialect
from .loader import build_warehouse, check_drift, create_indexes
from .synthetic import DEFAULT_SEED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nl2sql.warehouse.build",
        description="Generate and load the mobility warehouse.",
    )
    parser.add_argument(
        "--warehouse",
        default=None,
        choices=["duckdb", "snowflake"],
        help="target engine (default: $WAREHOUSE, else duckdb)",
    )
    parser.add_argument("--db", default=None, help="DuckDB file path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trips",
        type=int,
        default=None,
        help="override the trip count; everything else scales from it",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.db and (args.warehouse or "duckdb") == "duckdb":
        kwargs["database_path"] = args.db

    try:
        dialect = get_dialect(args.warehouse, **kwargs)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Snowflake needs configuration; say exactly what is missing rather
    # than failing at connect() with a connector traceback.
    if hasattr(dialect, "missing_settings"):
        missing = dialect.missing_settings()
        if missing:
            print("Snowflake is not configured. Set these first:", file=sys.stderr)
            for name in missing:
                print(f"  {name}", file=sys.stderr)
            return 2

    scale = {"trips": args.trips} if args.trips else None
    say = (lambda _m: None) if args.quiet else (lambda m: print(m, flush=True))

    say(f"building {dialect.name} warehouse ({len(S.TABLES)} tables)")
    report = build_warehouse(dialect=dialect, seed=args.seed, scale=scale, progress=say)

    connection = dialect.connect()
    try:
        indexes = create_indexes(dialect, connection)
        drift = check_drift(dialect, connection)
    finally:
        dialect.close(connection)

    say("")
    say(f"loaded {report.total_rows:,} rows across {len(report.rows_by_table)} tables "
        f"in {report.seconds}s")
    if indexes:
        say(f"created {indexes} indexes")

    if report.discrepancies:
        print("ROW COUNT MISMATCH:", file=sys.stderr)
        for problem in report.discrepancies:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if drift:
        print("SCHEMA DRIFT:", file=sys.stderr)
        for problem in drift:
            print(f"  {problem}", file=sys.stderr)
        return 1

    say("row counts verified, no schema drift")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
