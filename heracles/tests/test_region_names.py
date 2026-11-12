"""Hydra open-vocabulary regions get their class and a name from the room labelspace."""

import spark_dsg

from heracles.graph_interface import name_unnamed_regions, node_to_dict

ROOM_LS = {"0": "unknown", "7": "sidewalk", "9": "lawn"}


def _region_graph(labels):
    G = spark_dsg.DynamicSceneGraph()
    for i, label in enumerate(labels):
        attrs = spark_dsg.SemanticNodeAttributes()
        attrs.semantic_label = label
        G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", i), attrs)
    return G


def test_open_vocab_region_class_from_room_labelspace():
    G = _region_graph([9])
    (node,) = list(G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes)
    d = node_to_dict(node, room_labelspace=ROOM_LS)
    assert d["semantic_label"] == 9
    assert d["class"] == "lawn"
    assert "name" not in d


def test_open_vocab_region_without_labelspace_has_no_class():
    G = _region_graph([9])
    (node,) = list(G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes)
    assert "class" not in node_to_dict(node, room_labelspace={})


def test_unnamed_regions_take_their_class_numbered_in_symbol_order():
    G = _region_graph([7, 9, 7, 9, 9])
    nodes = [node_to_dict(n, room_labelspace=ROOM_LS) for n in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes]
    sym = {i: spark_dsg.NodeSymbol("R", i).str(True) for i in range(5)}
    named = {n["nodeSymbol"]: n["name"] for n in name_unnamed_regions(nodes)}
    assert named == {
        sym[0]: "sidewalk",
        sym[1]: "lawn",
        sym[2]: "sidewalk 2",
        sym[3]: "lawn 2",
        sym[4]: "lawn 3",
    }


def test_existing_names_are_kept_and_not_reused():
    nodes = [
        {"nodeSymbol": "R(0)", "attr_type": "SemanticNodeAttributes", "class": "lawn", "name": "lawn"},
        {"nodeSymbol": "R(1)", "attr_type": "SemanticNodeAttributes", "class": "lawn"},
        {"nodeSymbol": "R(2)", "attr_type": "RoomNodeAttributes", "class": "kitchen"},
    ]
    out = {n["nodeSymbol"]: n.get("name") for n in name_unnamed_regions(nodes)}
    assert out == {"R(0)": "lawn", "R(1)": "lawn 2", "R(2)": None}
