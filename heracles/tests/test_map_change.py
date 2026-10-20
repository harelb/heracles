"""notify_map_change bumps both version counters (console _State + publisher _MapState).

The DB tests need an explicit throwaway instance (HERACLES_TEST_NEO4J_URI); they never
fall back to the default port, because they wipe the version nodes.
"""

import os

import pytest

from heracles import graph_interface as gi


class _CaptureDB:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        return (self.records, None, None)


def test_notify_map_change_is_one_statement_touching_both_counters():
    db = _CaptureDB([{"sv": 5, "mv": 9}])
    assert gi.notify_map_change(db, ["objects"], ["O(3)"]) == (5, 9)
    (query, params), = db.calls
    assert "_State {key: 'global'}" in query and "_MapState" in query
    assert params == {"dl": ["objects"], "ai": ["O(3)"]}


URI = os.environ.get("HERACLES_TEST_NEO4J_URI")
AUTH = (
    os.environ.get("HERACLES_TEST_NEO4J_USER", "neo4j"),
    os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"),
)


@pytest.fixture
def db():
    if not URI:
        pytest.skip("set HERACLES_TEST_NEO4J_URI to a throwaway Neo4j")
    from heracles.query_interface import Neo4jWrapper

    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"No test Neo4j at {URI}: {exc}")
    db.execute("MATCH (n) WHERE n:_State OR n:_MapState DETACH DELETE n")
    yield db
    db.close()


def test_notify_map_change_bumps_both(db):
    assert gi.read_version(db) == 0 and gi.get_map_version(db) == 0
    assert gi.notify_map_change(db, ["objects"], ["O(1)"]) == (1, 1)
    assert gi.bump_map_version(db) == 2  # publisher-only writers keep working
    gi.record_change(db, ["places"], [])  # console-only writers too
    assert gi.notify_map_change(db, ["objects"], ["O(2)"]) == (3, 3)
    assert gi.read_change(db) == (3, ["objects"], ["O(2)"])
    assert gi.get_map_version(db) == 3
    # the ROS-side helpers are the same functions
    rows = db.query("MATCH (m:_MapState) RETURN count(m) AS n")
    assert rows[0]["n"] == 1
