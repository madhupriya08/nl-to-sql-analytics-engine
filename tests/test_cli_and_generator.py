"""Tests for the CLI surface and the generator module's contract.

The generator's live path needs a key, so what is tested here is
everything around it: the error paths, the prompt construction, and the
fact that importing the module does not require the anthropic package.
"""

from __future__ import annotations

import pytest

from nl2sql import cli, generator
from nl2sql.database import row_count
from nl2sql.demo import demo_generator


# ---------------------------------------------------------------------------
# Generator contract -- everything that does not need a key
# ---------------------------------------------------------------------------

def test_importing_the_package_does_not_require_anthropic():
    """The dependency boundary, asserted rather than assumed.

    nl2sql.generator imports anthropic inside the function. If that import
    moved to module scope, this test would still pass wherever the package
    happens to be installed -- so it also checks the module body directly.
    """
    with open(generator.__file__, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    top_level_imports = [
        line for line in lines if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("anthropic" in line for line in top_level_imports), (
        "anthropic must stay inside generate_sql() so the safety layer is "
        "testable without the package installed"
    )


def test_missing_api_key_raises_a_clear_generation_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(generator.GenerationError) as excinfo:
        generator.generate_sql("how many loans", "Table: loans")
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)
    assert "--demo" in str(excinfo.value), "the error should name the offline route"


def test_code_fences_are_stripped():
    assert generator.strip_code_fences("```sql\nSELECT 1\n```") == "SELECT 1"
    assert generator.strip_code_fences("```\nSELECT 1\n```") == "SELECT 1"
    assert generator.strip_code_fences("SELECT 1") == "SELECT 1"
    assert generator.strip_code_fences("  SELECT 1  ") == "SELECT 1"


def test_the_prompt_carries_the_schema_and_the_question():
    prompt = generator.build_user_prompt("how many loans", "SCHEMA_BLOCK_HERE")
    assert "SCHEMA_BLOCK_HERE" in prompt
    assert "how many loans" in prompt


def test_the_system_prompt_asks_for_a_single_read_only_statement():
    """Prompt hygiene is not a control, but it should still say the right thing.

    The validator is what guarantees safety. This only checks the prompt
    is not working against it.
    """
    system = generator.SYSTEM_PROMPT
    assert "ONE SQL statement" in system
    assert "SELECT or WITH" in system
    for keyword in ("DELETE", "DROP", "UPDATE", "PRAGMA"):
        assert keyword in system


def test_live_generator_factory_matches_the_engine_signature():
    """make_live_generator must produce a drop-in for demo_generator."""
    import inspect

    live = generator.make_live_generator()
    assert list(inspect.signature(live).parameters) == list(
        inspect.signature(demo_generator).parameters
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_demo_run_exits_zero_and_leaves_the_data_untouched(db_path, conn, capsys):
    """The full demo, including the DELETE, driven through the real CLI."""
    before = row_count(conn)
    exit_code = cli.main(["--demo", "--db", str(db_path)])
    captured = capsys.readouterr().out

    assert exit_code == 0, "a correctly blocked query is not a CLI failure"
    assert "BLOCKED" in captured
    assert "DELETE FROM loans" in captured
    assert "unchanged" in captured
    assert row_count(conn) == before


def test_cli_reports_a_block_distinctly_from_an_error(db_path, capsys):
    """A missing key must not look like the safety layer firing."""
    exit_code = cli.main(["--demo", "--db", str(db_path), "-q", "Delete all subprime loans"])
    blocked_output = capsys.readouterr().out
    assert exit_code == 0
    assert "BLOCKED [not_a_select]" in blocked_output
    assert "ERROR" not in blocked_output

    exit_code = cli.main(["--demo", "--db", str(db_path), "-q", "not a known question"])
    error_output = capsys.readouterr().out
    assert exit_code == 1
    assert "ERROR" in error_output
    assert "BLOCKED" not in error_output


def test_cli_prints_the_sql_before_the_results(db_path, capsys):
    cli.main(["--demo", "--db", str(db_path), "-q", "How many loans are in each risk tier?"])
    output = capsys.readouterr().out
    assert output.index("SQL executed") < output.index("risk_tier")


def test_list_demos_flags_the_destructive_case(db_path, capsys):
    exit_code = cli.main(["--list-demos", "--db", str(db_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "DESTRUCTIVE" in output
    assert "DELETE FROM loans WHERE risk_tier = 'Subprime'" in output


def test_show_schema_prints_the_grounding_block(db_path, capsys):
    cli.main(["--show-schema", "--db", str(db_path)])
    output = capsys.readouterr().out
    assert "Table: loans" in output
    assert "'Subprime'" in output


def test_live_mode_requires_a_question(db_path):
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db_path)])


def test_live_mode_without_a_key_errors_cleanly(db_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    exit_code = cli.main(["--db", str(db_path), "-q", "how many loans"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "ANTHROPIC_API_KEY" in output


def test_format_table_handles_empty_results():
    assert "no rows" in cli.format_table([])
