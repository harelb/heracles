"""Neo4j as the single map: seeding, holding state, and what the publisher sends.

These tests wipe the database they connect to, so they use a dedicated test
instance (HERACLES_TEST_NEO4J_URI, default neo4j://127.0.0.1:7687 -- the same
one heracles' own tests use) and skip when it is not running.
"""

import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import spark_dsg
from geometry_msgs.msg import Point
from heracles.graph_interface import db_to_spark_dsg
from heracles.query_interface import Neo4jWrapper
from heracles_ros_interfaces.srv import UpdateHoldingState

from heracles_ros.heracles_state_updater_node import HeraclesStateUpdater
from heracles_ros.map_state import (
    METADATA_KEY,
    bump_map_version,
    get_holding,
    get_map_version,
    map_metadata,
    seed_db,
)

URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", "neo4j_pw")
DSG_PATH = os.path.expanduser(
    os.environ.get(
        "HERACLES_TEST_DSG",
        "~/colcon_ws/assets/adt4_output/plan_repair_graph/hydra/backend/dsg.json",
    )
)
ROBOT = "hilbert"


def _connect():
    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"No test Neo4j at {URI}: {exc}")
    return db


@pytest.fixture
def db():
    if not os.path.exists(DSG_PATH):
        pytest.skip(f"No test scene graph at {DSG_PATH}")
    probe = _connect()
    probe.close()
    seed_db(URI, AUTH, DSG_PATH)
    db = _connect()
    db.query(f"MERGE (r:Robot {{name: '{ROBOT}'}}) SET r.position = point({{x: 1.0, y: 2.0, z: 0.5}})")
    yield db
    db.close()


def _holding_call(db, is_holding, object_id, position=None, robot_pose=None):
    """Run the state updater's service callback against the test database."""
    updater = NS(
        dsgdb_conf=NS(
            uri=URI,
            username=NS(get_secret_value=lambda: AUTH[0]),
            password=NS(get_secret_value=lambda: AUTH[1]),
        ),
        _get_robot_pose=lambda name: robot_pose,
        get_logger=lambda: NS(info=lambda *_: None, error=lambda *_: None),
    )
    req = UpdateHoldingState.Request()
    req.is_holding, req.id = is_holding, object_id
    if position is not None:
        req.has_position = True
        req.position = Point(x=position[0], y=position[1], z=position[2])
    return HeraclesStateUpdater.update_holding_state_callback(
        updater, ROBOT, req, UpdateHoldingState.Response()
    ).success


def _object_position(db, symbol):
    G = db_to_spark_dsg(db)
    return np.array(G.get_node(spark_dsg.NodeSymbol(symbol[0], int(symbol[1:]))).attributes.position)


def test_seed_matches_the_file(db):
    original = spark_dsg.DynamicSceneGraph.load(DSG_PATH)
    loaded = db_to_spark_dsg(db)
    for layer in (spark_dsg.DsgLayers.OBJECTS, spark_dsg.DsgLayers.MESH_PLACES):
        assert original.get_layer(layer).num_nodes() == loaded.get_layer(layer).num_nodes()
    # A fresh map is version 1, never 0 ("no map yet"), and holds nothing.
    assert get_map_version(db) == 1
    assert get_holding(db) == {}


def test_reseeding_clears_holding_and_restarts_the_version(db):
    assert _holding_call(db, True, "O4")
    bump_map_version(db)
    seed_db(URI, AUTH, DSG_PATH)
    assert get_holding(db) == {}
    assert get_map_version(db) == 1


def test_pick_and_place_at_the_given_point(db):
    v0 = get_map_version(db)
    assert _holding_call(db, True, "O4")
    assert get_holding(db) == {ROBOT: ["O4"]}
    assert get_map_version(db) == v0 + 1

    put_at = (12.5, -3.25, 0.4)
    assert _holding_call(db, False, "O4", position=put_at)
    assert get_holding(db) == {}
    assert get_map_version(db) == v0 + 2
    np.testing.assert_allclose(_object_position(db, "O4"), put_at)


def test_place_without_a_point_falls_back_to_the_robot(db):
    assert _holding_call(db, True, "O4")
    assert _holding_call(db, False, "O4", robot_pose=(3.0, 4.0, 0.2, 1, 0, 0, 0))
    np.testing.assert_allclose(_object_position(db, "O4"), (3.0, 4.0, 0.2))


def test_place_with_no_point_and_no_pose_fails_without_a_version_bump(db):
    assert _holding_call(db, True, "O4")
    v = get_map_version(db)
    assert not _holding_call(db, False, "O4")
    assert get_map_version(db) == v
    assert get_holding(db) == {ROBOT: ["O4"]}  # still held: nothing was written


def test_published_graph_carries_version_and_holding(db):
    assert _holding_call(db, True, "O4")
    # What the publisher sends, decoded the way omniplanner's subscriber does.
    G = db_to_spark_dsg(db)
    G.metadata.add(map_metadata(db))
    received = spark_dsg.DynamicSceneGraph.from_binary(G.to_binary(False))
    state = received.metadata.get()[METADATA_KEY]
    assert state == {"map_version": get_map_version(db), "holding": {ROBOT: ["O4"]}}

    # A later update replaces the metadata rather than merging into it, so a
    # released object does not linger in `holding`.
    assert _holding_call(db, False, "O4", position=(0.0, 0.0, 0.0))
    G = db_to_spark_dsg(db)
    G.metadata.add(map_metadata(db))
    received.update_from_binary(G.to_binary(False))
    assert received.metadata.get()[METADATA_KEY]["holding"] == {}
