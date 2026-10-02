"""Scene changes written into Neo4j, and what the published scene graph then says.

Wipes the database it connects to: uses the dedicated test instance
(HERACLES_TEST_NEO4J_URI, default neo4j://127.0.0.1:7687) and skips without it.
"""

import numpy as np
import pytest
import spark_dsg
from heracles.graph_interface import db_to_spark_dsg

from heracles_ros.map_state import get_map_version, seed_db
from heracles_ros.scene_change import (
    ADDED,
    MOVED,
    REMOVED,
    apply_changes,
    object_center,
)
from heracles_ros.scene_change_writer_node import applied_msg

from test_map_state import AUTH, DSG_PATH, ROBOT, URI, _connect  # noqa: F401
from test_map_state import db  # noqa: F401  fixture


def _contained_by(db, sym):
    rows = db.query(
        f"MATCH (p:MeshPlace)-[:CONTAINS]->(o:Object {{nodeSymbol: '{sym}'}}) "
        "RETURN p.nodeSymbol AS p"
    )
    return [r["p"] for r in rows]


def _published_position(db, sym):
    G = db_to_spark_dsg(db)
    for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        if n.id.str(True) == sym:
            return np.array(n.attributes.position)
    return None


def test_moved_object_is_moved_everywhere_the_map_keeps_it(db):
    before = _contained_by(db, "O5")
    applied = apply_changes(db, [(MOVED, "O5", "", (12.0, -8.0, 0.5))])
    assert applied[0]["kind"] == MOVED and applied[0]["old"] is not None
    np.testing.assert_allclose(object_center(db, "O5"), (12.0, -8.0, 0.5))
    row = db.query("MATCH (o:Object {nodeSymbol: 'O5'}) RETURN o.pos_x AS x, o.pos_y AS y")[0]
    assert (row["x"], row["y"]) == (12.0, -8.0)  # what Cypher queries read
    after = _contained_by(db, "O5")
    assert len(after) == 1 and after != before  # one CONTAINS, from the new place
    np.testing.assert_allclose(_published_position(db, "O5"), (12.0, -8.0, 0.5))


def test_removed_object_leaves_the_published_graph(db):
    assert apply_changes(db, [(REMOVED, "O5", "", (0, 0, 0))])[0]["kind"] == REMOVED
    assert _published_position(db, "O5") is None


def test_added_object_gets_a_symbol_and_is_published(db):
    applied = apply_changes(db, [(ADDED, "", "chair", (20.0, -6.0, 0.4))])
    sym = applied[0]["symbol"]
    assert sym == "O21"  # next free after O20
    np.testing.assert_allclose(_published_position(db, sym), (20.0, -6.0, 0.4))
    assert len(_contained_by(db, sym)) == 1


def test_adding_a_known_symbol_moves_it(db):
    applied = apply_changes(db, [(ADDED, "O5", "chair", (13.0, -9.0, 0.5))])
    assert applied[0]["kind"] == MOVED


def test_held_and_unknown_objects_are_skipped(db):
    db.query(
        f"MATCH (r:Robot {{name: '{ROBOT}'}}), (o:Object {{nodeSymbol: 'O4'}}) MERGE (r)-[:HOLDS]->(o)"
    )
    held_at = object_center(db, "O4")
    assert apply_changes(db, [(MOVED, "O4", "", (1.0, 1.0, 0.0)), (MOVED, "O99", "", (1, 1, 0))]) == []
    assert object_center(db, "O4") == held_at


def test_applied_message_carries_the_version():
    msg = applied_msg("change_detection", 7, [
        {"kind": MOVED, "symbol": "O5", "old": (1.0, 2.0, 0.0), "new": (3.0, 4.0, 0.0)},
        {"kind": REMOVED, "symbol": "O6", "old": (5.0, 6.0, 0.0), "new": None},
    ], None or __import__("builtin_interfaces.msg", fromlist=["Time"]).Time())
    assert msg.map_version == 7 and msg.source == "change_detection"
    assert [c.symbol for c in msg.changes] == ["O5", "O6"]
    assert (msg.changes[0].new_position.x, msg.changes[1].old_position.y) == (3.0, 6.0)


def test_seed_resets_versions_then_writes_bump_them(db):
    from heracles_ros.map_state import bump_map_version

    v = get_map_version(db)
    apply_changes(db, [(MOVED, "O5", "", (12.0, -8.0, 0.5))])
    assert bump_map_version(db) == v + 1
    seed_db(URI, AUTH, DSG_PATH)
    assert get_map_version(db) == 1
