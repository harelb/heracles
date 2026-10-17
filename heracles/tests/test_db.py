import datetime
import json
import os
import tempfile
from unittest.mock import patch


import neo4j
import numpy as np
import pytest
import spark_dsg

import heracles
from heracles.graph_interface import (
    add_buildings_from_dsg,
    add_edges_from_dsg,
    add_mesh_places_from_dsg,
    add_objects_from_dsg,
    add_places_from_dsg,
    add_rooms_from_dsg,
    obj_to_dict,
)
from heracles.query_interface import Neo4jWrapper
from image_store_fixture import build_store
from hydra_image_store import ImageStore
from heracles.graph_interface import add_agents_from_dsg, add_subkeyframes_from_dsg


def try_drop_index(db, index_name):
    try:
        db.execute(f"DROP INDEX {index_name}")
    except neo4j.exceptions.DatabaseError:
        print(f"No index `{index_name}`")


def add_dsg_metadata(G):
    # Inject a minimal labelspace using spark_dsg's native API.
    # spark_dsg_to_db() extracts these via extract_labelspaces_from_dsg().
    obj_ls = spark_dsg.Labelspace({0: "unknown", 1: "tree", 2: "box"})
    G.set_labelspace(obj_ls, 2, 0)

    layers = {
        2: "Object",
        5: "Building",
        20: "MeshPlace",
        3: "Place",
        4: "Room",
    }
    G.metadata.add({"LayerIdToHeraclesLayerStr": layers})


def build_test_dsg():
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "a", spark_dsg.DsgLayers.AGENTS)
    G.add_layer(3, "p", spark_dsg.DsgLayers.PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, "B", spark_dsg.DsgLayers.BUILDINGS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)

    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([0, 0, 0])
    room.semantic_label = 0

    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 0).value, room)

    place1 = spark_dsg.PlaceNodeAttributes()
    place1.position = np.array([-1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 0).value, place1)
    place2 = spark_dsg.PlaceNodeAttributes()
    place2.position = np.array([1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 1).value, place2)

    place1_2d = spark_dsg.PlaceNodeAttributes()
    place1_2d.position = np.array([-1.1, 0, 0])
    place1_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 0).value, place1_2d
    )
    place2_2d = spark_dsg.PlaceNodeAttributes()
    place2_2d.position = np.array([1.1, 0, 0])
    place2_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 1).value, place2_2d
    )

    obj1 = spark_dsg.ObjectNodeAttributes()
    obj1.position = np.array([-1.5, 0, 0])
    obj1.semantic_label = 34  # box
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("o", 0).value, obj1)
    obj2 = spark_dsg.PlaceNodeAttributes()
    obj2.position = np.array([1.5, 0, 0])
    obj2.semantic_label = 15  # rock
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("o", 1).value, obj2)

    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("o", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 1).value, spark_dsg.NodeSymbol("o", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 0).value, spark_dsg.NodeSymbol("P", 1).value
    )

    agent_attrs = spark_dsg.AgentNodeAttributes()
    agent_attrs.position = [4.0, 5.0, 6.0]
    agent_attrs.world_R_body = spark_dsg.Quaternion(0.5, 0.5, 0.5, 0.5)
    # Agents must live in a NON-ZERO partition keyed by the 'a' prefix
    # (ord('a')==97). The string-layer add_node overload always inserts at
    # partition 0, which _collect_agents deliberately skips — so use
    # the (layer_id:int, node_id, attrs, partition:int) overload explicitly.
    G.add_node(2, spark_dsg.NodeSymbol("a", 0), agent_attrs, ord("a"))

    # Second agent, used purely as a sub-keyframe anchor. It MUST have an
    # identity world_R_body (unlike a0, which Phase 1 gave a non-identity
    # rotation (0.5,0.5,0.5,0.5)) so the composed sub-keyframe world position
    # below is a simple translation and the test's expected value stays clean.
    anchor_attrs = spark_dsg.AgentNodeAttributes()
    anchor_attrs.position = [4.0, 5.0, 6.0]
    anchor_attrs.world_R_body = spark_dsg.Quaternion(1.0, 0.0, 0.0, 0.0)
    G.add_node(2, spark_dsg.NodeSymbol("a", 1), anchor_attrs, ord("a"))

    return G


@pytest.fixture(scope="module")
def populated_db():
    G = build_test_dsg()
    add_dsg_metadata(G)

    # IP / Port for database (override to point at a throwaway instance; the
    # fixture WIPES the database).
    URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
    # Database name / password for database
    AUTH = (
        os.environ.get("HERACLES_TEST_NEO4J_USER", "neo4j"),
        os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"),
    )
    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    db.connect()

    db.execute(
        "MATCH (n) DETACH DELETE n",
    )

    try_drop_index(db, "object_node_symbol")
    try_drop_index(db, "place_node_symbol")
    try_drop_index(db, "mesh_place_node_symbol")
    try_drop_index(db, "room_node_symbol")

    db.execute(
        """
    CREATE INDEX object_node_symbol FOR (n:Object) ON (n.nodeSymbol)
    """
    )

    db.execute(
        """
    CREATE INDEX place_node_symbol FOR (n:Place) ON (n.nodeSymbol)
    """
    )

    db.execute(
        """
    CREATE INDEX mesh_place_node_symbol FOR (n:MeshPlace) ON (n.nodeSymbol)
    """
    )

    db.execute(
        """
    CREATE INDEX room_node_symbol FOR (n:Room) ON (n.nodeSymbol)
    """
    )
    
    # Image store: o0 has an observation; a0/a1 have keyframes; two
    # sub-keyframes (one anchored to a1 with offset [0.5, 0, 0], one unanchored).
    with tempfile.TemporaryDirectory() as temp_dir:
        o0 = spark_dsg.NodeSymbol("o", 0).value
        o1 = spark_dsg.NodeSymbol("o", 9).value  # merged child, not in the DSG
        store_dir = build_store(temp_dir, obj1=o0, obj2=o1)
        store = ImageStore.open(store_dir)
        object_ls = {"0": "unknown", "4": "ground", "15": "rock", "34": "box"}

        add_objects_from_dsg(G, db, store, object_labelspace=object_ls)
        add_places_from_dsg(G, db)
        add_mesh_places_from_dsg(G, db, object_labelspace=object_ls)
        add_rooms_from_dsg(G, db)
        add_buildings_from_dsg(G, db)
        add_edges_from_dsg(G, db)
        add_agents_from_dsg(G, db, store)
        add_subkeyframes_from_dsg(G, db, store)

        yield db

    # db.close() # Clean up at end if needed, but yield handles it usually. 
    # The original code closed it after yield.
    db.close()


def test_observations(populated_db):
    # Two store observations (own track + merged child's track), both on o0.
    q = populated_db.query("""MATCH (obs:Observation) RETURN obs ORDER BY obs.score DESC""")
    assert len(q) == 2
    obs = q[0]["obs"]
    assert obs["timestamp_ns"] == 100
    assert obs["bbox_2d_min_x"] == 1 and obs["bbox_2d"] == [1, 2, 5, 6]

    q = populated_db.query(
        """MATCH (o:Object {nodeSymbol: "o0"})-[:HAS_OBSERVATION]->(obs:Observation) RETURN obs"""
    )
    assert len(q) == 2
    assert {r["obs"]["nodeSymbol"] for r in q} == {"o0_100", "o0_101"}


def test_rooms(populated_db):
    q = populated_db.query(
        """MATCH (r: Room {nodeSymbol: "R0"}) RETURN r.nodeSymbol as ns, r.center as center"""
    )
    assert len(q) == 1
    assert q[0]["ns"] == "R0"
    assert np.all(np.isclose(q[0]["center"], np.array([0, 0, 0])))


def test_places(populated_db):
    q = populated_db.query("""MATCH (p: Place {nodeSymbol: "p0"}) RETURN p""")
    assert np.all(np.isclose(np.array([-1, 0, 0]), q[0]["p"]["center"]))

    q = populated_db.query("""MATCH (p: Place) RETURN p""")
    assert len(q) == 2


def test_mesh_places(populated_db):
    q = populated_db.query("""MATCH (p: MeshPlace) RETURN p""")
    assert len(q) == 2

    q = populated_db.query("""MATCH (p: MeshPlace {nodeSymbol: "P0"}) RETURN p""")
    assert np.all(np.isclose(np.array([-1.1, 0, 0]), q[0]["p"]["center"]))
    assert q[0]["p"]["class"] == "ground"


def test_objects(populated_db):
    q = populated_db.query("""MATCH (o: Object) RETURN o""")
    assert len(q) == 2

    q = populated_db.query("""MATCH (o: Object {class: "box"}) RETURN o""")
    assert len(q) == 1
    assert q[0]["o"]["nodeSymbol"] == "o0"


def test_edges(populated_db):
    q = populated_db.query(
        """MATCH (r: Room {nodeSymbol: "R0"})-[:CONTAINS*]->(o: Object) RETURN o"""
    )

    assert len(q) == 2
    assert q[0]["o"]["class"] in ["box", "rock"]
    assert q[1]["o"]["class"] in ["box", "rock"]


def test_agents(populated_db):
    q = populated_db.query(
        """MATCH (a:Agent {nodeSymbol: "a0"})
           RETURN a.center AS center, a.rot_w AS rw, a.rot_x AS rx,
                  a.rot_y AS ry, a.rot_z AS rz, a.rgb_path AS rgb"""
    )
    assert len(q) == 1
    row = q[0]
    assert np.all(np.isclose(row["center"], np.array([4.0, 5.0, 6.0])))
    assert np.isclose(row["rw"], 0.5)
    assert np.isclose(row["rx"], 0.5)
    assert np.isclose(row["ry"], 0.5)
    assert np.isclose(row["rz"], 0.5)
    assert row["rgb"].endswith("kf0.jpg") or "_rgb" in row["rgb"]


def test_subkeyframes(populated_db):
    # anchor is a1 ([4,5,6], identity rotation) + stored offset [0.5,0,0]
    # -> composed world center = [4.5, 5, 6].
    q = populated_db.query(
        """MATCH (s:SubKeyframe)-[:ANCHORED_TO]->(a:Agent {nodeSymbol:"a1"})
           RETURN s.center AS center, s.timestamp_ns AS ts, s.rgb_path AS rgb,
                  s.frame_id AS fid, a.nodeSymbol AS ns"""
    )
    assert len(q) == 1
    assert np.all(np.isclose(q[0]["center"], np.array([4.5, 5.0, 6.0])))
    assert q[0]["ts"] == 150
    assert q[0]["rgb"].endswith("150_rgb.jpg")

    # One SubKeyframe per store sub-keyframe, anchored or not.
    n = populated_db.query("MATCH (s:SubKeyframe) RETURN count(s) AS n")
    assert n[0]["n"] == 2


def test_object_observations(populated_db):
    q = populated_db.query(
        """MATCH (:Object {nodeSymbol:"o0"})-[:HAS_OBSERVATION]->(o:Observation)
           RETURN o ORDER BY o.score DESC"""
    )
    # o0's own track plus the merged child o1's track.
    assert len(q) == 2
    best = q[0]["o"]
    assert best["kept"] is True and best["bbox_2d"] == [1, 2, 5, 6]
    assert best["mask_file"].endswith(".png") and best["mask_file"] == os.path.basename(best["mask_path"])
    assert best["rgb_path"].endswith("100_rgb.jpg")
    assert best["nodeSymbol"] == "o0_100"
    assert q[1]["o"]["kept"] is False and q[1]["o"].get("mask_path") is None
