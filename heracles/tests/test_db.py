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
    agent_attrs.image_folder = "agent_777"
    # Agents must live in a NON-ZERO partition keyed by the 'a' prefix
    # (ord('a')==97). The string-layer add_node overload always inserts at
    # partition 0, which _collect_keyframe_agents deliberately skips — so use
    # the (layer_id:int, node_id, attrs, partition:int) overload explicitly.
    G.add_node(2, spark_dsg.NodeSymbol("a", 0), agent_attrs, ord("a"))

    return G


@pytest.fixture(scope="module")
def populated_db():
    G = build_test_dsg()
    add_dsg_metadata(G)

    # IP / Port for database
    URI = "neo4j://127.0.0.1:7687"
    # Database name / password for database
    AUTH = ("neo4j", "neo4j_pw")
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
    
    # Create temp dir for images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy image folder for object "o0"
        obj_folder = os.path.join(temp_dir, "O_0")
        os.makedirs(obj_folder, exist_ok=True)
        
        # Create dummy meta file
        meta_data = {
            "timestamp_ns": 123456789,
            "mask_file": "frame_1_mask.png",
            "bbox_2d": {
                "min_x": 10, "min_y": 20, "max_x": 100, "max_y": 200
            }
        }
        with open(os.path.join(obj_folder, "frame_1_meta.json"), 'w') as f:
            json.dump(meta_data, f)
            
        # Custom obj_to_dict wrapper to inject image_folder
        original_obj_to_dict = obj_to_dict
        def mock_obj_to_dict(node_classes, obj):
            d = original_obj_to_dict(node_classes, obj)
            # Inject image_folder for o0 to point to our temp folder name (basename)
            if d["nodeSymbol"] == "o0":
                d["image_folder"] = "O_0"
            return d

        with patch("heracles.graph_interface.obj_to_dict", side_effect=mock_obj_to_dict):
            # We pass temp_dir as image_folder_root
            add_objects_from_dsg(G, temp_dir, db)
            
        add_places_from_dsg(G, db)
        add_mesh_places_from_dsg(G, db)
        add_rooms_from_dsg(G, db)
        add_buildings_from_dsg(G, db)
        add_edges_from_dsg(G, db)

        from heracles.graph_interface import add_agents_from_dsg
        add_agents_from_dsg(G, temp_dir, db)

        yield db
    
    # db.close() # Clean up at end if needed, but yield handles it usually. 
    # The original code closed it after yield.
    db.close()


def test_observations(populated_db):
    # Verify Observation created
    q = populated_db.query(
        """MATCH (obs:Observation) RETURN obs"""
    )
    assert len(q) == 1
    obs = q[0]["obs"]
    assert obs["timestamp_ns"] == 123456789
    assert obs["mask_file"] == "frame_1_mask.png"
    assert obs["bbox_2d_min_x"] == 10
    
    # Verify connection to object
    q = populated_db.query(
        """MATCH (o:Object {nodeSymbol: "o0"})-[:HAS_OBSERVATION]->(obs:Observation) RETURN obs"""
    )
    assert len(q) == 1
    assert q[0]["obs"]["nodeSymbol"] == "o0_123456789"


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
                  a.rot_y AS ry, a.rot_z AS rz, a.image_folder AS img"""
    )
    assert len(q) == 1
    row = q[0]
    assert np.all(np.isclose(row["center"], np.array([4.0, 5.0, 6.0])))
    assert np.isclose(row["rw"], 0.5)
    assert np.isclose(row["rx"], 0.5)
    assert np.isclose(row["ry"], 0.5)
    assert np.isclose(row["rz"], 0.5)
    assert row["img"] == "agent_777"
