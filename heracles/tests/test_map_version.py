"""The map version: bump_map_version records what changed, read_map_change reads it.

The DB test wipes _MapState, so it needs an explicit throwaway instance
(HERACLES_TEST_NEO4J_URI) and never falls back to a default port.
"""

import os

import pytest

from heracles import map_reload


class _CaptureDB:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        return (self.records, None, None)


def test_bump_records_the_change_in_the_same_statement():
    db = _CaptureDB([{"v": 7}])
    assert map_reload.bump_map_version(db, ["2:0"], ["O3"]) == 7
    assert map_reload.bump_map_version(db) == 7
    (q1, p1), (q2, p2) = db.calls
    assert q1 == q2 == map_reload.BUMP_MAP_VERSION
    assert p1 == {"layers": ["2:0"], "ids": ["O3"]}
    assert p2 == {"layers": None, "ids": None}  # not known: everything


def test_too_many_changed_ids_are_recorded_as_unknown():
    ids = [f"O{i}" for i in range(map_reload.MAX_CHANGED_IDS + 1)]
    assert map_reload.changed_params(["2:0"], ids) == {"layers": ["2:0"], "ids": None}


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
    db.execute("MATCH (m:_MapState) DETACH DELETE m")
    yield db
    db.close()


def test_read_map_change(db):
    assert map_reload.read_map_change(db) == (0, None, None)
    assert map_reload.bump_map_version(db, ["2:0"], ["O1", "O2"]) == 1
    assert map_reload.read_map_change(db) == (1, ["2:0"], ["O1", "O2"])
    assert map_reload.bump_map_version(db) == 2
    assert map_reload.read_map_change(db) == (2, None, None)
    assert map_reload.bump_map_version(db, [], []) == 3
    assert map_reload.read_map_change(db) == (3, [], [])
    # a reload sets the version itself: what changed is then not known
    with map_reload.reloading_map(db, "test reload", log=lambda m: None, full=False):
        pass
    assert map_reload.get_map_version(db) == 4
    assert map_reload.read_map_change(db) == (4, None, None)
    rows = db.query("MATCH (m:_MapState) RETURN count(m) AS n")
    assert rows[0]["n"] == 1
