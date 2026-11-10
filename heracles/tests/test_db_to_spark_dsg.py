"""db_to_spark_dsg call forms.

Requires a disposable Neo4j instance (its contents are cleared). Configure it
with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import numpy as np
import pytest
import spark_dsg

from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))

LAYER_IDS = {2: "Object", 3: "Place", 4: "Room", 5: "Building", 20: "MeshPlace"}
LAYER_NAMES = {**LAYER_IDS, "3[1]": "MeshPlace"}
OBJ1 = spark_dsg.NodeSymbol("O", 1).value


def build_dsg():
    """An object in a place in a room, with object and room labelspaces."""
    G = spark_dsg.DynamicSceneGraph()
    G.clear(True)
    G.add_layer(2, 0, spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, 0, spark_dsg.DsgLayers.PLACES)
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    G.add_layer(4, 0, spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, 0, spark_dsg.DsgLayers.BUILDINGS)
    G.set_labelspace(spark_dsg.Labelspace({0: "unknown", 1: "chair"}), 2, 0)
    G.set_labelspace(spark_dsg.Labelspace({0: "hallway"}), 4, 0)

    obj = spark_dsg.ObjectNodeAttributes()
    obj.position = np.array([1.0, 2.0, 3.0])
    obj.semantic_label = 1
    G.add_node(spark_dsg.DsgLayers.OBJECTS, OBJ1, obj)

    place = spark_dsg.PlaceNodeAttributes()
    place.position = np.array([1.0, 2.0, 0.0])
    place_id = spark_dsg.NodeSymbol("p", 0).value
    G.add_node(spark_dsg.DsgLayers.PLACES, place_id, place)
    G.insert_edge(place_id, OBJ1)

    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([1.0, 2.0, 1.0])
    room.semantic_label = 0
    room_id = spark_dsg.NodeSymbol("R", 0).value
    G.add_node(spark_dsg.DsgLayers.ROOMS, room_id, room)
    G.insert_edge(room_id, place_id)

    G.metadata.add({"LayerIdToHeraclesLayerStr": LAYER_NAMES})
    return G


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    gi.initialize_db(wrapper)
    gi.spark_dsg_to_db(build_dsg(), wrapper)
    yield wrapper
    wrapper.close()


def _summary(G):
    nodes = sorted(
        (n.id.str(True), getattr(n.attributes, "semantic_label", None)) for n in G.nodes
    )
    edges = sorted((e.source, e.target) for e in G.edges)
    return nodes, edges


def test_one_and_four_argument_forms_agree(db):
    one = gi.db_to_spark_dsg(db)
    four = gi.db_to_spark_dsg(db, LAYER_IDS, {"chair": 7}, {"hallway": 9})
    assert _summary(one) == _summary(four)
    assert one.get_node(OBJ1).attributes.semantic_label == 1  # the DB's labelspace


def test_labelspace_arguments_fill_in_a_missing_db_labelspace(db):
    db.execute("MATCH (m:_Labelspace) DETACH DELETE m")
    G = gi.db_to_spark_dsg(db, LAYER_IDS, {"unknown": 0, "chair": 5}, {})
    assert G.get_node(OBJ1).attributes.semantic_label == 5
    assert G.get_labelspace(2, 0).get_category(5) == "chair"
    assert isinstance(G.get_node(OBJ1).attributes, spark_dsg.ObjectNodeAttributes)
