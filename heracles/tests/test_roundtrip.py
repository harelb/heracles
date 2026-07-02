"""
Round-trip tests: spark_dsg → Neo4j → spark_dsg.

Verifies that loading a DSG into Neo4j via heracles and reconstructing it
preserves node data for both old and new scene graph formats.

These tests require:
- Neo4j running on localhost:7687 with credentials neo4j/neo4j_pw
- The old example DSG at heracles/examples/scene_graphs/example_dsg.json
- Optionally, the new DSG (set NEW_DSG_PATH env var or skip)
"""

import os

import neo4j
import numpy as np
import pytest
import spark_dsg

from heracles import constants
from heracles.graph_interface import (
    db_record_to_spark_attrs,
    db_to_spark_dsg,
    get_layer_nodes,
    initialize_db,
    spark_dsg_to_db,
)
from heracles.query_interface import Neo4jWrapper

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OLD_DSG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "examples",
    "scene_graphs",
    "example_dsg.json",
)

NEW_DSG_PATH = os.environ.get(
    "HERACLES_TEST_NEW_DSG",
    os.path.expanduser(
        "~/software/mit/awesome-dcist-t4/scene_graphs/2026-04-02/"
        "2026_04_02_b10_prior_map_2/hydra/backend/dsg_with_mesh.json"
    ),
)

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = ("neo4j", "neo4j_pw")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _try_drop_index(db, index_name):
    try:
        db.execute(f"DROP INDEX {index_name}")
    except neo4j.exceptions.DatabaseError:
        pass


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    yield wrapper
    wrapper.close()


def _load_and_push(db, dsg_path):
    """Load a DSG file, push to Neo4j. Returns the original DSG.

    Labelspaces are read from the DSG's embedded metadata by spark_dsg_to_db().
    """
    original = spark_dsg.DynamicSceneGraph.load(dsg_path)
    original.metadata.add(
        {
            "LayerIdToHeraclesLayerStr": {
                2: "Object",
                3: "Place",
                4: "Room",
                5: "Building",
                20: "MeshPlace",
                "3[1]": "MeshPlace",
            }
        }
    )
    initialize_db(db)
    spark_dsg_to_db(original, db, source_file_path=dsg_path)

    # Extract labelspaces for callers that need the reverse mappings.
    from heracles.utils import extract_labelspaces_from_dsg

    obj_ls, room_ls = extract_labelspaces_from_dsg(original)
    obj_labels = {name: int(sid) for sid, name in obj_ls.items()} if obj_ls else {}
    room_labels = {name: int(sid) for sid, name in room_ls.items()} if room_ls else {}
    return original, obj_labels, room_labels


# ---------------------------------------------------------------------------
# Old DSG round-trip tests
# ---------------------------------------------------------------------------


class TestOldDsgRoundtrip:
    def test_node_counts_preserved(self, db):
        """Node counts per layer match after round-trip."""
        original, obj_labels, room_labels = _load_and_push(db, OLD_DSG_PATH)

        for layer_label, layer_enum in [
            (constants.OBJECTS, spark_dsg.DsgLayers.OBJECTS),
            (constants.MESH_PLACES, spark_dsg.DsgLayers.MESH_PLACES),
            (constants.ROOMS, spark_dsg.DsgLayers.ROOMS),
        ]:
            orig_count = original.get_layer(layer_enum).num_nodes()
            db_nodes, _, _ = get_layer_nodes(db, layer_label)
            assert len(db_nodes) == orig_count, (
                f"{layer_label}: expected {orig_count}, got {len(db_nodes)}"
            )

    def test_attr_type_stored(self, db):
        """Every node in Neo4j has an attr_type property."""
        _load_and_push(db, OLD_DSG_PATH)

        for label in [constants.OBJECTS, constants.MESH_PLACES, constants.ROOMS]:
            nodes, _, _ = get_layer_nodes(db, label)
            for node in nodes:
                assert "attr_type" in node, (
                    f"Node {node.get('nodeSymbol')} missing attr_type"
                )

    def test_object_properties_preserved(self, db):
        """Object nodes preserve class, name, position, and bounding box."""
        original, obj_labels, room_labels = _load_and_push(db, OLD_DSG_PATH)

        nodes, _, _ = get_layer_nodes(db, constants.OBJECTS)
        assert len(nodes) > 0

        sample = nodes[0]
        assert "class" in sample
        assert "center" in sample
        # insert_nodes_to_db converts flat bbox_x/y/z/l/w/h to native Point3D
        # bbox_center / bbox_dim and removes the flat keys.
        assert "bbox_center" in sample
        assert "bbox_dim" in sample
        assert sample["attr_type"] == "ObjectNodeAttributes"
        assert "color_r" in sample
        assert "registered" in sample
        # New schema additions (Phase 0.5): explicit layer/partition ints.
        assert sample.get("layer") == 2
        assert sample.get("partition") == 0

    def test_room_properties_preserved(self, db):
        """Room nodes preserve class and position."""
        _load_and_push(db, OLD_DSG_PATH)

        nodes, _, _ = get_layer_nodes(db, constants.ROOMS)
        assert len(nodes) > 0

        sample = nodes[0]
        assert "class" in sample
        assert sample["attr_type"] == "RoomNodeAttributes"

    def test_reconstruction(self, db):
        """Reconstructed DSG has the same node symbols as original."""
        original, obj_labels, room_labels = _load_and_push(db, OLD_DSG_PATH)

        reconstructed = db_to_spark_dsg(db)

        # Compare object node symbols.
        orig_objs = {n.id.str(True) for n in original.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes}
        recon_objs = {n.id.str(True) for n in reconstructed.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes}
        assert orig_objs == recon_objs

    def test_source_file_path_stored(self, db):
        """The source file path is stored as _GraphMetadata."""
        _load_and_push(db, OLD_DSG_PATH)

        result = db.query(
            "MATCH (m:_GraphMetadata {key: 'source'}) RETURN m.file_path AS path"
        )
        assert len(result) == 1
        assert result[0]["path"].endswith("example_dsg.json")


# ---------------------------------------------------------------------------
# New DSG round-trip tests (TravNodeAttributes)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(NEW_DSG_PATH),
    reason=f"New DSG not found at {NEW_DSG_PATH}",
)
class TestNewDsgRoundtrip:
    def test_trav_node_properties(self, db):
        """TravNodeAttributes store radii, states, and timestamps."""
        _load_and_push(db, NEW_DSG_PATH)

        nodes, _, _ = get_layer_nodes(db, constants.MESH_PLACES)
        assert len(nodes) > 0

        sample = nodes[0]
        assert sample["attr_type"] == "TravNodeAttributes"
        assert "radii" in sample
        assert "states" in sample
        assert "min_radius" in sample
        assert "max_radius" in sample
        assert "first_observed_ns" in sample
        assert isinstance(sample["radii"], list)
        assert isinstance(sample["states"], list)

    def test_khronos_object_properties(self, db):
        """KhronosObjectAttributes store bbox, color, timestamps."""
        _load_and_push(db, NEW_DSG_PATH)

        nodes, _, _ = get_layer_nodes(db, constants.OBJECTS)
        assert len(nodes) > 0

        sample = nodes[0]
        assert sample["attr_type"] == "KhronosObjectAttributes"
        assert "bbox_x" in sample
        assert "color_r" in sample
        assert "first_observed_ns" in sample
        # KhronosObjectAttributes has list timestamps
        assert isinstance(sample["first_observed_ns"], list)

    def test_node_counts(self, db):
        """Correct node counts for the new DSG."""
        _load_and_push(db, NEW_DSG_PATH)

        objects, _, _ = get_layer_nodes(db, constants.OBJECTS)
        mesh_places, _, _ = get_layer_nodes(db, constants.MESH_PLACES)
        assert len(objects) == 14
        assert len(mesh_places) == 160


# ---------------------------------------------------------------------------
# Attr type required test
# ---------------------------------------------------------------------------


class TestAttrTypeRequired:
    def test_missing_attr_type_raises(self, db):
        """Reconstruction raises ValueError if attr_type is missing."""
        # Manually insert a node without attr_type.
        db.execute("MATCH (n) DETACH DELETE n")
        db.execute(
            "CREATE (:Object {nodeSymbol: 'test_no_type', "
            "center: point({x: 0, y: 0, z: 0})})"
        )

        records, _, _ = get_layer_nodes(db, constants.OBJECTS)
        assert len(records) == 1

        with pytest.raises(ValueError, match="no attr_type"):
            db_record_to_spark_attrs(records[0], {}, {})


# ---------------------------------------------------------------------------
# Agent-partition selection (no live Neo4j required — uses a capturing fake db)
# ---------------------------------------------------------------------------


class _CaptureDB:
    """Minimal db stub: records the ``agents`` kwarg passed to db.execute()."""

    def __init__(self):
        self.captured_agents = None

    def execute(self, query, **params):
        if "agents" in params:
            self.captured_agents = params["agents"]
        return ([], None, None)


def _build_object_and_agent_graph():
    """A DSG with one Object (layer 2, partition 0) and one Agent (layer 2,
    partition ord('a')). Both share layer id 2 — agents are distinguished only
    by partition, which is the crux of the bug under test."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, 0, spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(2, ord("a"), spark_dsg.DsgLayers.AGENTS)

    obj = spark_dsg.KhronosObjectAttributes()
    obj.position = np.array([1.0, 2.0, 3.0])
    obj.image_folder = "/crops/O5"
    G.add_node(2, spark_dsg.NodeSymbol("O", 5).value, obj, 0)

    agent = spark_dsg.AgentNodeAttributes()
    agent.position = np.array([10.0, 20.0, 30.0])
    agent.image_folder = "/abs/agents/agent_123"
    G.add_node(2, spark_dsg.NodeSymbol("a", 0).value, agent, ord("a"))
    return G


class TestAgentPartition:
    """add_agents_from_dsg must read the agent partition (layer 2, partition
    != 0), NOT G.get_layer(DsgLayers.AGENTS) which resolves to layer 2 /
    partition 0 == the OBJECTS layer."""

    def test_agents_read_from_correct_partition(self):
        from heracles.graph_interface import add_agents_from_dsg

        G = _build_object_and_agent_graph()
        cap = _CaptureDB()
        # image_folder_root is the OBJECT-crop dir; agents must NOT be rebased onto it.
        add_agents_from_dsg(G, "/crops", cap)

        assert cap.captured_agents is not None, "no agents were written"
        syms = [a["nodeSymbol"] for a in cap.captured_agents]
        # Exactly the real agent; the object O5 must NOT be written as an Agent.
        assert syms == ["a0"], f"expected only the agent a0, got {syms}"

    def test_agent_image_folder_not_rebased(self):
        from heracles.graph_interface import add_agents_from_dsg

        G = _build_object_and_agent_graph()
        cap = _CaptureDB()
        add_agents_from_dsg(G, "/crops", cap)

        agent = cap.captured_agents[0]
        # The agent's image_folder is already an absolute agents/ prefix; it must
        # be preserved verbatim, not joined onto the object-crop root.
        assert agent["image_folder"] == "/abs/agents/agent_123"

    def test_object_not_mislabeled_as_agent(self):
        from heracles.graph_interface import add_agents_from_dsg

        G = _build_object_and_agent_graph()
        cap = _CaptureDB()
        add_agents_from_dsg(G, "/crops", cap)

        syms = [a["nodeSymbol"] for a in (cap.captured_agents or [])]
        assert "O5" not in syms, "object O5 was mislabeled as an Agent"


def _build_frontend_like_graph(folders):
    """A DSG with agent nodes a0..a{N-1}, each with the given image_folder (or '')."""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, ord("a"), spark_dsg.DsgLayers.AGENTS)
    for i, folder in enumerate(folders):
        a = spark_dsg.AgentNodeAttributes()
        a.position = np.array([float(i), 0.0, 0.0])
        a.image_folder = folder
        G.add_node(2, spark_dsg.NodeSymbol("a", i).value, a, ord("a"))
    return G


class TestMergeAgentImageFolders:
    """merge_agent_image_folders upserts image_folder by symbol (no rebasing),
    so backend agent nodes that lost image_folder during pose-graph optimization
    can be filled from the sibling frontend DSG."""

    def test_merges_all_nonempty_folders(self):
        from heracles.graph_interface import merge_agent_image_folders

        # frontend-like: a0,a1,a2 all have folders (backend would have dropped a1).
        G = _build_frontend_like_graph(["/agents/agent_0", "/agents/agent_1", "/agents/agent_2"])
        cap = _CaptureDB()
        n = merge_agent_image_folders(G, cap)

        assert n == 3
        rows = {a["nodeSymbol"]: a["image_folder"] for a in cap.captured_agents}
        assert rows == {
            "a0": "/agents/agent_0",
            "a1": "/agents/agent_1",
            "a2": "/agents/agent_2",
        }

    def test_skips_empty_folders(self):
        from heracles.graph_interface import merge_agent_image_folders

        G = _build_frontend_like_graph(["/agents/agent_0", "", "/agents/agent_2"])
        cap = _CaptureDB()
        n = merge_agent_image_folders(G, cap)

        assert n == 2
        syms = {a["nodeSymbol"] for a in cap.captured_agents}
        assert syms == {"a0", "a2"}
