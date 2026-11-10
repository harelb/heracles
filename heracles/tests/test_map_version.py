"""Map version counters: read_version / notify_map_change.

Requires a disposable Neo4j instance (the version nodes are deleted). Configure
it with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import pytest

from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    wrapper.execute("MATCH (n) WHERE n:_State OR n:_MapState DETACH DELETE n")
    yield wrapper
    wrapper.close()


def _map_version(db):
    records, _, _ = db.execute("MATCH (m:_MapState) RETURN m.version AS v")
    return records[0]["v"] if records else 0


def test_read_version_is_zero_before_any_change(db):
    assert gi.read_version(db) == 0


def test_notify_map_change_bumps_both_counters(db):
    assert gi.notify_map_change(db, ["2:0"], ["O(1)"]) == (1, 1)
    assert gi.read_version(db) == 1
    assert gi.notify_map_change(db) == (2, 2)
    assert gi.read_version(db) == 2
    assert _map_version(db) == 2

    records, _, _ = db.execute(
        "MATCH (s:_State {key: 'global'}) "
        "RETURN s.dirty_layers AS dl, s.affected_ids AS ai"
    )
    assert records[0]["dl"] == [] and records[0]["ai"] == []


def test_notify_map_change_continues_publisher_counter(db):
    # A writer that only bumps _MapState (heracles_ros.map_state) and
    # notify_map_change share one counter.
    db.execute("MERGE (m:_MapState) SET m.version = 5")
    assert gi.notify_map_change(db, ["2:0"], ["O(3)"]) == (1, 6)
    records, _, _ = db.execute(
        "MATCH (s:_State {key: 'global'}) "
        "RETURN s.dirty_layers AS dl, s.affected_ids AS ai"
    )
    assert records[0]["dl"] == ["2:0"] and records[0]["ai"] == ["O(3)"]
