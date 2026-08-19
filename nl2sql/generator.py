"""Claude-backed SQL generation -- the only module that needs an API key.

This is deliberately the thinnest module in the project. It takes a
question and a schema block, asks Claude for one SELECT statement, and
returns the string. It does not validate, does not execute, and does not
decide anything about safety.

That containment is the point. ``anthropic`` is imported inside the
function rather than at module scope, so importing :mod:`nl2sql` -- or
running the entire test suite -- never requires the package to be
installed, let alone a key to be set. Everything downstream of this
function consumes a plain ``str``, which is why the pipeline can be
tested end to end against a canned generator.
"""

from __future__ import annotations

import os
import re

#: Claude model used for generation. Sonnet is the right tier here: the
#: task is short, highly constrained, and heavily grounded by the schema
#: block, so the marginal benefit of a larger model is small relative to
#: its latency on an interactive CLI.
DEFAULT_MODEL = "claude-sonnet-4-5"

#: Generated SQL is short. A low cap bounds cost and, more usefully,
#: bounds how much text the validator has to reason about.
DEFAULT_MAX_TOKENS = 700

SYSTEM_PROMPT = """\
You are a SQL generator for a read-only analytics database (SQLite).

Rules:
- Reply with exactly ONE SQL statement and nothing else.
- No prose, no explanation, no markdown code fences, no trailing semicolon.
- The statement MUST begin with SELECT or WITH. Never write DELETE, DROP,
  INSERT, UPDATE, ALTER, CREATE, REPLACE, TRUNCATE, ATTACH, DETACH,
  PRAGMA or VACUUM. If a question asks you to modify or remove data,
  still reply with a single SELECT that reports on the data instead.
- Use only the tables and columns given in the schema below. Never invent
  a table or column name.
- When a column's complete set of distinct values is listed, use those
  values exactly as written, including capitalisation and punctuation.
- If the question cannot be answered from this schema, reply with exactly:
  SELECT 'question cannot be answered from this schema' AS unanswerable
- Prefer explicit column lists over SELECT *, round money to 2 decimal
  places, and add a LIMIT when a question implies a top-N answer.\
"""

USER_PROMPT_TEMPLATE = """\
Database schema (read live from the database):

{schema_context}

Question: {question}

Reply with one SQL statement only."""


class GenerationError(RuntimeError):
    """Raised when SQL could not be obtained from the model.

    Distinct from a validation failure: this means we never got a
    candidate query, as opposed to getting one and rejecting it. The CLI
    reports the two differently because they need different responses --
    retry versus rephrase.
    """


#: Models wrap SQL in ```sql fences often enough that stripping them is
#: worth doing here rather than letting the validator reject the result.
#: This is formatting cleanup, not sanitisation -- nothing security
#: relevant happens in this module, and the validator sees whatever
#: survives this regardless.
_FENCE_PATTERN = re.compile(r"^\s*```(?:sql)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    match = _FENCE_PATTERN.match(text)
    return (match.group(1) if match else text).strip()


def build_user_prompt(question: str, schema_context: str) -> str:
    """Render the user turn. Exposed so tests can assert on grounding.

    The live-mode check in the test suite uses this to confirm the schema
    block the model actually received is the one introspection produced --
    the failure this guards against is a prompt that quietly falls back to
    a stale or empty schema, which would still produce plausible SQL.
    """
    return USER_PROMPT_TEMPLATE.format(
        schema_context=schema_context, question=question
    )


def generate_sql(
    question: str,
    schema_context: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
) -> str:
    """Ask Claude for one SQL statement answering ``question``.

    Signature matches what :func:`nl2sql.engine.answer_question` expects of
    a ``sql_generator_fn``: ``(question, schema_context) -> str``. The
    engine treats the return value as untrusted text and validates it
    before anything else happens to it.
    """
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise GenerationError(
            "the 'anthropic' package is required for live mode; "
            "install it with `pip install anthropic`, or use --demo"
        ) from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise GenerationError(
            "ANTHROPIC_API_KEY is not set; export it for live mode, "
            "or run with --demo to use the canned generator"
        )

    client = anthropic.Anthropic(api_key=key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_user_prompt(question, schema_context),
                }
            ],
        )
    except Exception as exc:  # anthropic raises several distinct error types
        raise GenerationError(f"Claude API call failed: {exc}") from exc

    text = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    sql = strip_code_fences(text)
    if not sql:
        raise GenerationError("model returned an empty response")
    return sql


def make_live_generator(
    model: str = DEFAULT_MODEL, api_key: str | None = None
):
    """Build a ``sql_generator_fn`` closed over model and key settings.

    The engine's pluggable-generator seam is a two-argument callable, so
    per-run configuration is bound here rather than threaded through the
    engine's signature. The engine stays unaware of which generator it
    holds -- that indifference is what makes the demo generator a drop-in.
    """

    def _generate(question: str, schema_context: str) -> str:
        return generate_sql(
            question, schema_context, model=model, api_key=api_key
        )

    return _generate
