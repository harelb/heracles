"""map_state's version helpers are heracles.map_reload's, with what changed."""

import os

import pytest
from heracles import map_reload

from heracles_ros import map_state


class _Db:
    def __init__(self):
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        return ([{"v": 3}], None, None)


def test_bump_map_version_is_the_core_one():
    assert map_state.bump_map_version is map_reload.bump_map_version
    assert map_state.read_map_change is map_reload.read_map_change


def test_notify_map_change_records_what_changed_and_bumps_the_console():
    db = _Db()
    assert map_state.notify_map_change(db, ["3:1"], ["P(4)"]) == 3
    (bump, params), (console, _) = db.calls
    assert bump == map_reload.BUMP_MAP_VERSION
    assert params == {"layers": ["3:1"], "ids": ["P(4)"]}
    assert "_State" in console
    db = _Db()
    assert map_state.notify_map_change(db) == 3  # as SGET calls it
    assert db.calls[0][1] == {"layers": None, "ids": None}


def test_notify_map_change_in_map_edit_records_the_change():
    uri = os.environ.get("HERACLES_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("set HERACLES_TEST_NEO4J_URI to a throwaway Neo4j")
    from heracles.query_interface import Neo4jWrapper

    db = Neo4jWrapper(
        uri,
        ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw")),
        atomic_queries=True,
        print_profiles=False,
    )
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"No test Neo4j at {uri}: {exc}")
    try:
        db.execute("MATCH (n) WHERE n:_MapState OR n:_State DETACH DELETE n")
        db.execute("CREATE (:_State {key: 'global', version: 4})")
        with map_state.map_edit(db) as tx:
            assert map_state.notify_map_change(tx, ["2:0"], ["O1"]) == 1
        assert map_state.read_map_change(db) == (1, ["2:0"], ["O1"])
        rows = db.query("MATCH (s:_State {key: 'global'}) RETURN s.version AS v")
        assert rows[0]["v"] == 5  # a console's counter still moves with it
    finally:
        db.execute("MATCH (n) WHERE n:_MapState OR n:_State DETACH DELETE n")
        db.close()
