"""A tiny scene graph with objects, places, rooms and two agent (keyframe) nodes."""

import datetime

import numpy as np
import spark_dsg

AGENT0 = spark_dsg.NodeSymbol("a", 0).value
AGENT1 = spark_dsg.NodeSymbol("a", 1).value
OBJ1 = spark_dsg.NodeSymbol("O", 1).value
OBJ2 = spark_dsg.NodeSymbol("O", 2).value  # not in the graph (merged into O1)

LAYER_NAMES = {
    2: "Object",
    3: "Place",
    4: "Room",
    5: "Building",
    20: "MeshPlace",
    "3[1]": "MeshPlace",
}


def build_dsg():
    G = spark_dsg.DynamicSceneGraph()
    G.clear(True)
    G.add_layer(2, 0, spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(2, ord("a"), spark_dsg.DsgLayers.AGENTS)
    G.add_layer(3, 0, spark_dsg.DsgLayers.PLACES)
    G.add_layer(3, 1, spark_dsg.DsgLayers.MESH_PLACES)
    G.add_layer(4, 0, spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, 0, spark_dsg.DsgLayers.BUILDINGS)
    G.set_labelspace(spark_dsg.Labelspace({0: "unknown", 1: "chair"}), 2, 0)
    G.set_labelspace(spark_dsg.Labelspace({0: "hallway"}), 4, 0)

    obj = spark_dsg.ObjectNodeAttributes()
    obj.position = np.array([1.0, 2.0, 3.0])
    obj.semantic_label = 1
    obj.name = "chair"
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

    for nid, pos, stamp_us in (
        (AGENT0, (0.0, 0.0, 0.0), 0),
        (AGENT1, (4.0, 5.0, 6.0), 2),
    ):
        a = spark_dsg.AgentNodeAttributes()
        a.position = np.array(pos)
        a.world_R_body = spark_dsg.Quaternion(1.0, 0.0, 0.0, 0.0)
        a.timestamp = datetime.timedelta(microseconds=stamp_us)
        G.add_node(2, nid, a, ord("a"))
    G.insert_edge(AGENT0, AGENT1)  # the robot path
    G.insert_edge(place_id, AGENT1)  # the place the agent is in

    G.metadata.add({"LayerIdToHeraclesLayerStr": LAYER_NAMES})
    return G
