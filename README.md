# Agentic Natural Language-to-SQL Analytics Engine

Ask questions in English about a synthetic loan book; get answers, plus
the exact SQL that produced them.

The interesting engineering here is not the SQL generation — a model does
that in one API call. It is the **safety layer between what the model
writes and what the database runs**, and the design decision that makes
that layer testable without an API key.

```
question ──▶ schema grounding ──▶ generation ──▶ VALIDATION ──▶ execution ──▶ result
             (live PRAGMA)        (Claude, or        (4 checks)    (SELECT      (+ the exact
                                   canned pairs)                    only)        SQL that ran)
```

---

## Quick start

```bash
pip install -r requirements.txt

# Demo mode: no API key, no network. Builds the database on first run.
python -m nl2sql.cli --demo

# See what each canned case is meant to demonstrate
python -m nl2sql.cli --list-demos

# The safety case on its own
python -m nl2sql.cli --demo -q "Delete all subprime loans"

# Live mode
export ANTHROPIC_API_KEY=sk-ant-...
python -m nl2sql.cli -q "Which entity has the largest subprime exposure?"

# Tests
python -m pytest tests/ -q
```

A demo run ends with a row count taken before and after, so the central
claim is asserted by the CLI itself, not only by the test suite:

```
Q: Delete all subprime loans
SQL (blocked, not executed):
  DELETE FROM loans WHERE risk_tier = 'Subprime'
BLOCKED [not_a_select]: query starts with 'DELETE'; only SELECT and WITH are allowed
  checks passed before block: single_statement

Data integrity check: 5,000 rows before, 5,000 after -- unchanged.
```

---

## Running it in an IDE

Two things trip people up in both VS Code and PyCharm, and they are the
same two:

1. **It must run as a module, not a script.** Clicking "run this file" on
   `nl2sql/cli.py` fails with
   `ImportError: attempted relative import with no known parent package`,
   because the package uses relative imports. Use `python -m nl2sql.cli`.
2. **The working directory must be the repo root.** `data/loans.db` is a
   relative path, and the package will not be found from elsewhere.

**VS Code** — `.vscode/launch.json` is committed with configs for demo
mode, `--show-schema`, the blocked-`DELETE` case and live mode, each
already setting `"module"` and `"cwd"` correctly. Pick one from the Run
and Debug panel. Tests are pre-wired to pytest in `.vscode/settings.json`.
For browsing the database file, the *SQLite Viewer* extension opens
`data/loans.db` directly.

**PyCharm** — `Run` → `Edit Configurations` → `+` → Python, then switch
the target dropdown from **Script path** to **Module name** and enter
`nl2sql.cli`, with the repo root as the working directory. Set pytest as
the default runner under `Settings` → `Tools` → `Python Integrated Tools`.
The Database tool window (`View` → `Tool Windows` → `Database`) reads
`data/loans.db`, but note it is **Professional only** — on Community, use
[DB Browser for SQLite](https://sqlitebrowser.org/) or just
`--show-schema`, which shows more than a table browser does anyway.
`.idea/` is gitignored, since PyCharm project files are machine-specific.

**Where to put breakpoints.** `engine.answer_question()` holds the whole
pipeline in one function — inspect `raw_sql` as the generator returns it,
then `validation` immediately after, and the untrusted-string-to-checked-
result transition is visible in one step. `validator.validate_sql()` is
the other one worth watching: run the blocked-`DELETE` config and watch
`passed` accumulate `single_statement` and then stop dead at check 2.

---

## Why whitelist, not blacklist

This is the load-bearing decision in the project, so it is worth being
precise about why.

A **blacklist** asks: *does this query contain anything I know to be
dangerous?* A **whitelist** asks: *is this query provably one of the small
number of shapes I understand?* They sound like two framings of one idea.
They are not, and they fail in opposite directions.

**A blacklist is a list of attacks someone already thought of.** It is
correct only until someone thinks of a new one. Every entry is a patch for
a specific past failure, and the set of things it does not cover is
unbounded and unknowable. You cannot enumerate the complement of your own
imagination.

**The failure modes are asymmetric.** When a whitelist is wrong, it
refuses something harmless — a visible error, a user rephrases, someone
files a bug. When a blacklist is wrong, it *executes something
destructive*, silently, and you find out from the row count. One failure
mode costs a rephrase. The other costs the data.

**A blacklist is stateful about the past; a whitelist is structural.**
`SELECT` is read-only by construction — that fact does not change when
SQLite adds a feature or when someone finds a novel encoding. So the
whitelist keeps holding for cases nobody has enumerated. Concretely, this
validator rejects:

- SQLite-specific escape hatches — `ATTACH` reaches other database files
  on disk, `PRAGMA` can disable integrity enforcement, `VACUUM` rewrites
  the file — *and also* rejects the ones I did not think of, because they
  are not `SELECT`;
- syntax that does not exist yet in SQLite but might in a future version;
- statements valid in a different engine, if this is ever pointed at
  Postgres.

A blacklist would need an entry for each of those. The whitelist needs
none, because none of them is a `SELECT`.

**The honest caveat.** Checks 1–3 in this validator *are* a blacklist —
`FORBIDDEN_KEYWORDS` is literally a list of dangerous words. That is not
a contradiction; it is defence in depth in the one place the whitelist
cannot see. Checks 1 and 2 do the structural work: exactly one statement,
and it begins with `SELECT`. Check 3 exists for what those two structurally
cannot inspect — a mutation nested *inside* a statement that legitimately
begins with `SELECT`. If check 3 were the whole design, this project would
have the flaw it is arguing against. It is the third of four, and the
first two are the boundary.

---

## The four checks

Run in order, cheapest first, each a precondition for the next. Nothing
reaches `execute()` without passing all four.
[`nl2sql/validator.py`](nl2sql/validator.py)

### 1. Exactly one statement

```sql
SELECT * FROM loans; DROP TABLE loans
```

**Blocked: `multiple_statements`.**

This runs first because statement chaining defeats every other check. The
query above *starts with `SELECT`* — a leading-keyword check passes it on
the strength of the first half and never looks at the second. Only
counting statements catches the pair.

Splitting is quote-aware: `WHERE entity = 'Acme; Capital'` is one
statement, not two. It has to be right in both directions — a false
positive rejects a legitimate question, a false negative *is the attack* —
so comment stripping and statement splitting share a single scanner rather
than being two implementations that can drift apart. Quote-blind stripping
mangles `WHERE entity = '-- x'`; quote-blind splitting reads `'A;B'` as a
chain. When input ends inside an unterminated literal, the scan ends there
too, which fails toward rejection rather than execution.

A single *trailing* semicolon is formatting, not chaining, and is stripped
after this check has confirmed nothing follows it.

### 2. Must start with `SELECT` or `WITH`

```sql
DELETE FROM loans WHERE risk_tier = 'Subprime'
```

**Blocked: `not_a_select`.** This is the check that stops the demo's
destructive case — before the database is asked to do anything at all.

`WITH` is allowed because CTEs are genuinely useful for analytical
questions, but it is a real widening: SQLite permits a data-modifying
statement as a CTE body, so `WITH` alone is not evidence of a read.
It therefore carries an extra condition — a `WITH` statement that never
reaches a `SELECT` is rejected here rather than being left to check 3.
Delegating a known, specific bypass to a single downstream check leaves it
one deletion away from being unguarded.

### 3. No forbidden keyword *anywhere*

```sql
SELECT * FROM loans WHERE loan_id IN (DELETE FROM loans RETURNING loan_id)
```

**Blocked: `forbidden_keyword`.** This query passes checks 1 and 2 —
one statement, opens with `SELECT`. The mutation is in a subquery, where
a leading-token check structurally cannot see it. Only scanning the whole
string finds it.

The match is a **word-boundary regex**, and the `\b` is load-bearing in
both directions:

| | substring match | word-boundary match |
|---|---|---|
| `SELECT created_at FROM loans` | ❌ blocked (contains `CREATE`) | ✅ allowed |
| `SELECT dropoff_date FROM t` | ❌ blocked (contains `DROP`) | ✅ allowed |
| `... IN(DELETE FROM loans)` | ✅ blocked | ✅ blocked (`(` is a boundary) |

A substring blacklist rejects `created_at`, `updated_at` and
`dropoff_date` and is unusable on any realistic schema. Word boundaries
are not a loophole either: punctuation and newlines are boundaries too, so
a keyword hugging a bracket is still a whole word.

The keyword set spans DML, DDL, SQLite's escape hatches (`ATTACH`,
`DETACH`, `PRAGMA`, `VACUUM`), transaction control, and verbs other
engines honour (`GRANT`, `EXEC`, `COPY … OUTFILE`) so the validator does
not quietly become unsafe the day this points at a different database.

**Accepted trade-off, stated rather than hidden:** this scans string
literals too, so a real question about a company named *Drop Inc* is
refused. That is the correct direction to be wrong in — a visible refusal
the user rephrases around, versus silent data loss. Skipping literals
means parsing them, and a parser that disagrees with SQLite's by one
character *is* a bypass. There is a test that records this behaviour so it
stays deliberate.

### 4. `EXPLAIN` against the live database

```sql
SELECT fico_score FROM customers
```

**Blocked: `did_not_compile` — `no such table: customers`.**

Nothing about that string is suspicious. It is one statement, it starts
with `SELECT`, it contains no forbidden keyword. It is simply *made up*,
and no amount of string inspection can tell.

`EXPLAIN` makes SQLite compile the statement to bytecode and hand back the
program **without running a single instruction**. Compilation resolves
every table and column reference against the real schema, so hallucinated
identifiers die here — as a free side effect of a syntax check. It runs
last because it is the only check that touches the database, and it should
only ever see SQL the three static checks have already cleared.

That EXPLAIN does not execute is the premise of this whole check, so
there is a test that takes the row count around one.

### One more detail

Stage 4 of the pipeline executes `validation.sql` — the normalised string
the validator *proved safe* — not the raw string the generator returned.
If those two could differ, the thing that was checked would not be the
thing that runs. A small time-of-check/time-of-use gap, but the class of
bug is not small, and closing it costs nothing.

---

## Demo mode is a testability decision, not a shortcut

`answer_question()` takes a `sql_generator_fn(question, schema_context) -> str`
as a **parameter**, not an import. The live generator calls Claude; the
demo generator looks up a dictionary. The engine cannot tell them apart.

The critical property: **swapping the generator does not stub out the
pipeline.** Live schema introspection still runs. All four validation
checks still run. Real SQLite execution against the real 5,000-row
database still runs. The only thing replaced is the single step that is
non-deterministic, credential-requiring, network-dependent and slow — and
that step is the one part of the system that makes no safety decisions.

What is left is everything worth testing, and it runs offline, in CI, in
under two seconds, with byte-identical results every time.

The alternatives are each worse in a specific way:

| approach | what goes wrong |
|---|---|
| mock the `anthropic` client | tests your mock's idea of the API, not the pipeline |
| recorded fixtures / VCR | go stale silently as prompts and models change |
| skip tests when no key is set | the safety layer is untested exactly where no key exists — which includes CI by default |
| require a key to run tests | every contributor needs a credential to verify a security property |

That last row is the real argument. If the safety tests needed an API key,
they would be the tests most likely to be skipped — and they are the tests
that matter most.

**The demo set includes one genuinely dangerous case.** "Delete all
subprime loans" maps to a real, executable `DELETE` that removes 1,015
rows if it ever ran. Not defanged, not commented out, not a lookalike
string. A safety layer demonstrated only against queries that were never
going to run anyway demonstrates nothing — the claim is *destructive SQL
does not reach the database*, and the only honest way to show that is to
point destructive SQL at it. There is even a test asserting the canned
`DELETE` compiles as a valid `DELETE`, because if it were malformed every
other block assertion would be vacuous.

---

## Schema grounding is read live, every time

The schema block in the prompt comes from `PRAGMA table_info` on every
call. It is never hardcoded. [`nl2sql/schema.py`](nl2sql/schema.py)

A hardcoded schema string is a second source of truth, and it drifts
silently: add a column and the model keeps writing SQL against what it was
told; rename one and you get a confidently wrong answer computed from the
wrong field. Introspection cannot drift, because there is only one source.

Introspection also does something a handwritten description almost always
omits — it samples the **actual values**:

```
Table: loans (5000 rows)
  - loan_id (TEXT, primary key)
  - entity (TEXT, not null)
      complete set of 4 distinct values: 'Acme Capital', 'Borealis Bank',
      'Cedar Financial', 'Dunhill Credit'
  - risk_tier (TEXT, not null)
      complete set of 3 distinct values: 'Near-Prime', 'Prime', 'Subprime'
  - loan_amount (REAL, not null)
      4998 distinct values, ranging from '924.46' to '391292.44'
  - origination_date (TEXT, not null)
      1402 distinct values, ranging from '2021-01-01' to '2024-12-31'
  - default_flag (INTEGER, not null)
      complete set of 2 distinct values: '0', '1'
```

Knowing a column is called `risk_tier` does not tell the model whether the
value is `'Prime'`, `'prime'` or `'PRIME'`. SQLite compares TEXT with `=`
case-sensitively, so a wrong guess returns **zero rows and no error** —
the worst failure in the whole pipeline, because it is not a crash, it is
an empty result that reads like a finding ("no subprime loans defaulted").

Profiling is cardinality-aware because the two cases carry different
information. Twelve or fewer distinct values are enumerated and labelled
as the *complete* set — which also tells the model that a filter on any
other value must return nothing, so it should decline rather than invent a
category. Higher-cardinality columns report `MIN`/`MAX` instead: twelve
arbitrary dates teach nothing, but "2021-01-01 to 2024-12-31" lets "last
year" resolve correctly.

---

## The data

~5,000 synthetic loans, generated from a fixed seed with the standard
library only. [`nl2sql/synthetic_data.py`](nl2sql/synthetic_data.py)

| column | type | notes |
|---|---|---|
| `loan_id` | TEXT | primary key, `LN-000001` |
| `entity` | TEXT | 4 booking entities |
| `product` | TEXT | 5 product lines |
| `risk_tier` | TEXT | `Prime` / `Near-Prime` / `Subprime` |
| `loan_amount` | REAL | triangular within a per-product range |
| `interest_rate` | REAL | band determined by risk tier |
| `origination_date` | TEXT | ISO-8601; sortable as text |
| `default_flag` | INTEGER | 0/1, so `AVG()` gives a default rate |

Indexed on `risk_tier`, `product`, `entity`, `default_flag` and
`origination_date` — the columns natural-language questions actually
filter and group on — plus composites on `(risk_tier, product)` and
`(risk_tier, default_flag)` for the common two-dimensional breakdowns.

Distributions are **correlated, not uniform**, and that is a testing
decision. Risk tier drives both the rate band and the default probability,
so `AVG(interest_rate) GROUP BY risk_tier` must come back
Prime < Near-Prime < Subprime. A wrong `GROUP BY` or a swapped aggregate
breaks that ordering visibly. With uniform data every grouped query
returns three near-identical bars and there is nothing to assert.

The seed is fixed so tests can assert exact numbers rather than "roughly"
— weak assertions are how safety regressions slip through.

---

## Modules

| module | role | needs a key? |
|---|---|---|
| `synthetic_data.py` | deterministic loan book | no |
| `database.py` | build, index, and the **single** `execute()` path | no |
| `schema.py` | live `PRAGMA` introspection + value sampling | no |
| `validator.py` | **the four checks** | no |
| `generator.py` | Claude call | **yes** |
| `engine.py` | grounding → generation → validation → execution | no |
| `demo.py` | canned pairs, including the destructive one | no |
| `cli.py` | `--demo` and live mode | only live |

`anthropic` is imported *inside* `generate_sql()`, not at module scope, so
importing `nl2sql` or running the entire test suite never requires the
package to be installed. There is a test that reads the module's source
and fails if that import moves to the top level.

All execution of generated SQL goes through one function,
`database.execute_select()`. One line to audit for "can unvalidated SQL
reach the database?". It never uses `executescript()`, so even a chained
statement that somehow arrived would raise rather than helpfully running
both halves.

---

## Tests

```
$ python -m pytest tests/ -q
156 passed, 4 skipped in 1.4s
```

The four skips are the live-mode tests, which need a key.

Every test asserts not just *that* a query was blocked but **which check
blocked it**. A test that only asserts "blocked" keeps passing after the
check it was written for is deleted — something else happens to catch the
case, the test stays green, and the coverage is gone. Hence `Rejection`
being a typed enum and `checks_passed` being part of every result.

| file | covers |
|---|---|
| `test_validator_adversarial.py` | chaining (incl. comment-hidden separators), every forbidden keyword as a leading token *and* in subquery position, word boundaries in both directions, hallucinated identifiers, malformed SQL, empty/`None`/prose input |
| `test_pipeline.py` | correct results vs. independently computed values, the destructive question blocked with row counts unchanged, graceful decline, reported SQL re-executed to reproduce reported rows, grounding verified by capture and by adding a column at runtime |
| `test_cli_and_generator.py` | exit codes, block-vs-error labelling, the dependency boundary |
| `test_live_mode.py` | real API: generated SQL passes the same validator, and the prompt contained the introspected schema |

Two tests exist to keep the other tests honest: one compiles the canned
`DELETE` to prove it is genuinely destructive, and one takes the row count
around an `EXPLAIN` to prove check 4's premise holds.

---

## Scope and honest limitations

**This handles single-table schemas only.** The grounding step dumps the
whole schema into the prompt, which works because there is one table with
eight columns. It does not generalise.

A real multi-table version would need **embedding-based retrieval of the
relevant tables and columns** rather than dumping everything:

- a warehouse with hundreds of tables blows the context window, and long
  before that, precision falls — the model picks the wrong one of six
  similarly-named `*_facts` tables;
- you would embed table and column descriptions, retrieve the top-k
  relevant to the question, and ground the prompt in that subset;
- retrieval quality then becomes the dominant error source, and it needs
  its own evaluation harness — the right table has to be in the retrieved
  set before generation can possibly be correct.

**That is not built here.** It is a different project, and pretending
otherwise would misrepresent what this demonstrates.

Other limits worth naming:

- **Foreign keys and joins are untested.** The validator does not reason
  about join semantics at all. It would not stop an expensive accidental
  cartesian product — `max_rows` caps materialisation, which is a blunt
  backstop, not a plan-cost limit.
- **No cost or timeout ceiling.** A validated query that scans everything
  still runs to completion. Production would want `sqlite3`'s progress
  handler or a statement timeout.
- **The validator is SQLite-shaped.** The keyword set anticipates other
  engines, but checks 1 and 4 lean on SQLite's tokenizer and `EXPLAIN`
  semantics. Porting to Postgres means re-verifying both.
- **Read-only is enforced in application code, not by the database.**
  A production deployment should *also* connect as a role with no write
  grants, or open the file with SQLite's read-only URI flag. Defence in
  depth: the validator should not be the only thing standing between a
  generated string and the data.
- **No row-level authorisation.** Every question can see every row. A
  multi-tenant version needs the filter applied below this layer, not by
  asking the model nicely to include a `WHERE tenant_id = …`.
- **Semantic correctness is not verified.** The validator proves a query
  is safe and that it compiles. It cannot prove the query answers the
  question that was asked. That is what showing the SQL alongside every
  result is for — the user remains the last check, which is why the SQL is
  printed before the numbers rather than after.
