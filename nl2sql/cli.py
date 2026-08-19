"""Command-line interface.

Two modes over one pipeline. ``--demo`` uses the canned generator and
needs nothing; live mode calls Claude and needs ``ANTHROPIC_API_KEY``.
The mode choice swaps a single function -- every other stage, including
all four validation checks, is identical between them. That is what makes
the demo an honest demonstration rather than a separate happy path.

Output always shows the SQL, whether it ran or was blocked. When a query
is blocked, it shows which checks passed before the block, so the safety
layer's decision is legible rather than an opaque refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from .database import DEFAULT_DB_PATH, connect, ensure_database, row_count
from .demo import DEMO_CASES, demo_generator, demo_questions
from .engine import QueryResult, answer_question
from .schema import build_schema_context

# ANSI colours, suppressed when stdout is not a terminal so piped output
# stays clean.
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def bold(text: str) -> str:
    return _c("1", text)


def green(text: str) -> str:
    return _c("32", text)


def red(text: str) -> str:
    return _c("31", text)


def yellow(text: str) -> str:
    return _c("33", text)


def dim(text: str) -> str:
    return _c("2", text)


def format_table(rows: Sequence[dict[str, Any]], max_rows: int = 25) -> str:
    """Render result rows as a fixed-width table.

    Deliberately plain -- no dependency on a table library for what is
    ultimately a demonstration harness.
    """
    if not rows:
        return dim("(no rows)")

    headers = list(rows[0].keys())
    shown = rows[:max_rows]
    widths = {
        h: max(len(str(h)), *(len(_fmt(r.get(h))) for r in shown)) for h in headers
    }

    line = "  ".join(bold(str(h).ljust(widths[h])) for h in headers)
    rule = "  ".join("-" * widths[h] for h in headers)
    body = [
        "  ".join(_fmt(r.get(h)).ljust(widths[h]) for h in headers) for r in shown
    ]
    out = [line, dim(rule), *body]
    if len(rows) > max_rows:
        out.append(dim(f"... {len(rows) - max_rows} more row(s)"))
    return "\n".join(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_result(result: QueryResult, show_schema: bool = False) -> None:
    """Print one answer, SQL first.

    SQL comes before the results on purpose. The query is the claim; the
    rows are only its consequence. Showing the numbers first invites
    reading them without checking what produced them.
    """
    print()
    print(bold(f"Q: {result.question}"))

    if result.sql:
        label = "SQL (blocked, not executed)" if not result.ok else "SQL executed"
        print(dim(f"{label}:"))
        print(f"  {result.sql}")

    if show_schema and result.schema_context:
        print(dim("Schema context given to the generator:"))
        for line in result.schema_context.splitlines():
            print(dim(f"  {line}"))

    if not result.ok:
        if result.rejection is None:
            # No candidate SQL was produced, or execution failed after a
            # clean validation. Reported as an error rather than a block:
            # conflating the two would make an unset API key look like the
            # safety layer firing, which is exactly the signal that has to
            # stay unambiguous.
            print(red(f"ERROR: {result.message}"))
            return
        print(red(f"BLOCKED [{result.rejection.value}]: {result.message}"))
        if result.checks_passed:
            print(dim(f"  checks passed before block: {', '.join(result.checks_passed)}"))
        else:
            print(dim("  blocked by the first check"))
        return

    if result.unanswerable:
        print(yellow(f"UNANSWERABLE: {result.message}"))
        return

    print(green(f"OK [{result.row_count} row(s)] "
                f"checks passed: {', '.join(result.checks_passed)}"))
    print(format_table(result.rows))


def _run_questions(
    questions: Sequence[str],
    db_path: Path,
    generator,
    show_schema: bool,
    guard_row_count: bool,
) -> int:
    """Answer each question in turn; return a process exit code.

    Exit codes distinguish the three outcomes, because a script wrapping
    this needs to tell them apart:

    * 0 -- every question either answered or was correctly blocked. A
      blocked destructive query is the system working, not a failure, so
      it must not turn into a non-zero exit.
    * 1 -- a question errored: no SQL was generated (missing key, network
      failure, unknown demo question) or execution failed after passing
      validation.
    * 2 -- the row count moved. Nothing in this pipeline can legitimately
      change the data, so this means the safety claim is broken.

    ``guard_row_count`` re-counts rows after the batch. In demo mode that
    is a live assertion of the central safety claim, made by the CLI
    itself rather than only by the test suite: a destructive query was
    submitted and the table is the same size.
    """
    conn = connect(db_path)
    try:
        before = row_count(conn)
        errored = False
        for question in questions:
            result = answer_question(question, conn, generator)
            print_result(result, show_schema=show_schema)
            if not result.ok and result.rejection is None:
                errored = True

        if guard_row_count:
            after = row_count(conn)
            print()
            if before == after:
                print(green(
                    f"Data integrity check: {before:,} rows before, "
                    f"{after:,} after -- unchanged."
                ))
            else:
                print(red(
                    f"DATA INTEGRITY FAILURE: {before:,} rows before, "
                    f"{after:,} after."
                ))
                return 2
        return 1 if errored else 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nl2sql",
        description=(
            "Natural-language analytics over a synthetic loan book, with a "
            "whitelist safety layer between the model and the database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m nl2sql.cli --demo\n"
            "  python -m nl2sql.cli --demo -q 'Delete all subprime loans'\n"
            "  python -m nl2sql.cli -q 'Average loan size by product'\n"
            "  python -m nl2sql.cli --show-schema --list-demos\n"
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="use the canned generator (no API key, no network)",
    )
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        dest="questions",
        metavar="TEXT",
        help="a question to answer; repeat for several",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"path to the SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild the database from the synthetic generator before running",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model for live mode (default: the generator's default)",
    )
    parser.add_argument(
        "--show-schema",
        action="store_true",
        help="print the introspected schema block sent to the generator",
    )
    parser.add_argument(
        "--list-demos",
        action="store_true",
        help="list the canned demo questions and what each demonstrates",
    )
    parser.add_argument(
        "--no-integrity-check",
        action="store_true",
        help="skip the before/after row-count check at the end of a run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_demos:
        print(bold("Demo questions:"))
        for case in DEMO_CASES:
            marker = red(" [DESTRUCTIVE - must be blocked]") if case.dangerous else ""
            print(f"\n  {bold(case.question)}{marker}")
            print(dim(f"    demonstrates: {case.demonstrates}"))
            print(dim(f"    sql: {case.sql}"))
        return 0

    from .database import build_database

    db_path = Path(args.db)
    if args.rebuild:
        build_database(db_path, overwrite=True)
        print(dim(f"Rebuilt {db_path}"))
    else:
        ensure_database(db_path)

    if args.show_schema and not args.questions and not args.demo:
        print(build_schema_context(connect(db_path)))
        return 0

    if args.demo:
        generator = demo_generator
        questions = args.questions or list(demo_questions())
        print(bold(f"Demo mode ({len(questions)} question(s)) -- "
                   f"canned generator, no API key required"))
    else:
        if not args.questions:
            build_parser().error(
                "live mode needs at least one -q/--question (or use --demo)"
            )
        from .generator import DEFAULT_MODEL, make_live_generator

        model = args.model or DEFAULT_MODEL
        generator = make_live_generator(model=model)
        questions = args.questions
        print(bold(f"Live mode -- generating SQL with {model}"))

    return _run_questions(
        questions,
        db_path,
        generator,
        show_schema=args.show_schema,
        # The integrity check is most meaningful in demo mode, where a
        # destructive query is guaranteed to have been submitted, but it
        # is cheap enough to run always.
        guard_row_count=not args.no_integrity_check,
    )


if __name__ == "__main__":
    raise SystemExit(main())
