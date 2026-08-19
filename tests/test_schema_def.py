"""Integrity tests for the warehouse declaration.

The schema declaration is read by three consumers -- the loader, the
retriever and the prompt builder -- so a mistake here does not fail
loudly in one place. It produces a table that cannot be created, or a
retrieval hit whose description no longer matches its columns. These
tests catch that at the declaration, which is the only place all three
agree.
"""

from __future__ import annotations

import pytest

from nl2sql.warehouse import schema_def as S


def test_every_table_has_a_primary_key():
    missing = [t.name for t in S.TABLES if not t.primary_key]
    assert missing == []


def test_every_foreign_key_points_at_something_real():
    """A dangling FK breaks the loader and, worse, misleads retrieval.

    Retrieval expands a candidate set along the foreign-key graph. An FK
    naming a table that does not exist would silently drop that edge, so
    the model would be handed a table it cannot join to anything.
    """
    broken: list[str] = []
    for table in S.TABLES:
        for column, target in table.foreign_keys:
            target_table, target_column = target.split(".", 1)
            if target_table not in S.TABLES_BY_NAME:
                broken.append(f"{table.name}.{column} -> unknown table {target_table}")
                continue
            try:
                S.TABLES_BY_NAME[target_table].column(target_column)
            except KeyError:
                broken.append(f"{table.name}.{column} -> unknown column {target}")
    assert broken == []


def test_subject_areas_are_declared_strings():
    """Guards the collision that actually happened while writing this.

    The subject-area constants were originally bare names (TRIPS =
    "trips"), and the module later rebound TRIPS to the trips Table.
    Every table declared after that point received a Table object as its
    subject_area instead of a string, which broke grouping silently. The
    AREA_ prefix makes the collision impossible; this test makes a
    regression visible.
    """
    for table in S.TABLES:
        assert isinstance(table.subject_area, str), table.name
        assert table.subject_area in S.SUBJECT_AREAS, table.name


def test_table_names_are_unique():
    names = [t.name for t in S.TABLES]
    assert len(names) == len(set(names))
    assert set(names) == set(S.TABLES_BY_NAME)


def test_column_names_are_unique_within_each_table():
    for table in S.TABLES:
        names = [c.name for c in table.columns]
        assert len(names) == len(set(names)), table.name


def test_tables_are_declared_in_dependency_order():
    """The loader creates and populates front to back, so a table's
    referents must already exist. Self-references and the trips/payments
    cycle are the only exceptions, handled by the loader explicitly.
    """
    seen: set[str] = set()
    out_of_order: list[str] = []
    for table in S.TABLES:
        for _, target in table.foreign_keys:
            referent = target.split(".", 1)[0]
            if referent == table.name:
                continue
            if referent not in seen:
                out_of_order.append(f"{table.name} references {referent} before it is declared")
        seen.add(table.name)
    # feedback -> trips and payments -> trips are declared after trips, so
    # the only legitimate forward reference is trips.feedback, which does
    # not exist. Anything reported here is a real ordering bug.
    assert out_of_order == []


def test_pii_columns_all_exist():
    """The sanitiser masks by this list; a typo would silently expose data.

    A misspelled entry does not raise -- it just fails to match any
    column, and the field it was meant to protect is returned in full.
    That is exactly the kind of failure that stays invisible until it
    matters, so it is asserted here.
    """
    for reference in S.PII_COLUMNS:
        table_name, column_name = reference.split(".", 1)
        assert table_name in S.TABLES_BY_NAME, reference
        S.TABLES_BY_NAME[table_name].column(column_name)


def test_enum_values_are_only_on_text_columns():
    for table in S.TABLES:
        for column in table.columns:
            if column.enum_values:
                assert column.type == "VARCHAR", f"{table.name}.{column.name}"


def test_descriptions_are_substantive():
    """Retrieval quality is bounded by description quality.

    A description that merely restates the name embeds to roughly the
    same vector as the name itself and retrieves no better. The length
    floor is crude but catches the real failure -- a placeholder left in.
    """
    for table in S.TABLES:
        assert len(table.description) > 60, table.name
        assert table.grain, table.name
        for column in table.columns:
            assert len(column.description) > 10, f"{table.name}.{column.name}"


def test_the_warehouse_is_large_enough_to_need_retrieval():
    """The premise of the whole RAG layer, asserted rather than assumed.

    If the schema were small enough to dump wholesale into a prompt,
    retrieval would be theatre. This pins the scale that makes it
    necessary -- and fails if someone trims the warehouse to a size where
    the architecture no longer earns its place.
    """
    assert len(S.TABLES) >= 20
    assert sum(len(t.columns) for t in S.TABLES) >= 150


@pytest.mark.parametrize("table", S.TABLES, ids=lambda t: t.name)
def test_retrieval_document_carries_the_searchable_vocabulary(table):
    document = S.retrieval_document(table)
    assert table.name in document
    assert table.description in document
    assert table.grain in document
    for column in table.columns:
        assert column.name in document
        assert column.description in document
    for synonym in table.synonyms:
        assert synonym in document


def test_foreign_key_graph_is_symmetric_and_connected():
    """Retrieval walks this graph, so an island would be unreachable.

    A table no foreign key touches can only ever be retrieved on its own
    text, never by expansion from a neighbour -- which for a warehouse
    this size means it is effectively invisible to multi-table questions.
    """
    graph = S.foreign_key_graph()
    for table, neighbours in graph.items():
        for neighbour in neighbours:
            assert table in graph[neighbour], f"{table}/{neighbour} asymmetric"

    reachable = {"trips"}
    frontier = ["trips"]
    while frontier:
        for neighbour in graph[frontier.pop()]:
            if neighbour not in reachable:
                reachable.add(neighbour)
                frontier.append(neighbour)

    unreachable = set(graph) - reachable
    # vehicle_models reaches the graph through vehicles; every table
    # should be reachable from the central fact.
    assert unreachable == set(), f"unreachable from trips: {sorted(unreachable)}"
