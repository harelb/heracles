"""The indexes an image-store load relies on (needs a scratch Neo4j, which is cleared:
HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD)."""

import os

import pytest

from heracles import constants
from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    try:
        wrapper.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No test Neo4j at {NEO4J_URI}: {exc}")
    gi.initialize_db(wrapper)
    wrapper.query("DROP INDEX subkeyframe_node_symbol IF EXISTS")
    yield wrapper
    wrapper.close()


def _indexed(db, label):
    rows = db.query(
        "SHOW INDEXES YIELD labelsOrTypes, properties RETURN labelsOrTypes, properties"
    )
    return {tuple(r["properties"]) for r in rows if r["labelsOrTypes"] == [label]}  # fmt: skip


def test_subkeyframes_are_indexed_by_node_symbol(db):
    """They are MERGEd by nodeSymbol: without the index each MERGE scanned every
    SubKeyframe, and a rebuild of willow's map took 18 minutes."""
    assert ("nodeSymbol",) not in _indexed(db, constants.SUBKEYFRAMES)
    gi.ensure_provenance_indexes(db)
    assert ("nodeSymbol",) in _indexed(db, constants.SUBKEYFRAMES)
    gi.ensure_provenance_indexes(db)  # again: nothing to do, no error


def test_the_index_is_made_before_subkeyframes_are_written(db, monkeypatch):
    """A rebuild from the prior loads the map without initialize_db, so the load
    itself has to make the index before its first MERGE."""
    seen = []
    monkeypatch.setattr(
        gi, "insert_subkeyframes_to_db",
        lambda db_, subframes: seen.append(("nodeSymbol",) in _indexed(db_, constants.SUBKEYFRAMES)),
    )  # fmt: skip
    monkeypatch.setattr(gi, "insert_edges", lambda *a, **k: None)
    monkeypatch.setattr(gi, "subkeyframe_to_dict", lambda f, pose: {"nodeSymbol": "s1"})

    class Store:
        def body_pose(self, frame, G):
            return (None, None)

    assert (
        gi.add_subkeyframes_from_dsg(None, db, Store(), keyframes=([], [object()])) == 1
    )
    assert seen == [True]
