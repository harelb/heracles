"""The evidence-admission gate's schema + the prior-map ingest stamp (G1).

Objects arriving through ``add_objects_from_dsg`` come from a COMMITTED scene
graph, so they are *trusted priors*: readers downstream refuse to plan on an
:Object that carries no explicit admission and looks detector-written, and this
stamp is what keeps an ingested map plannable by declaration rather than by a
reader's default.

No Neo4j required: ``insert_nodes_to_db`` is stubbed so the assertions are
about the node dicts the ingest hands it.
"""
import numpy as np
import pytest
import spark_dsg

from heracles import constants
from heracles.graph_interface import add_objects_from_dsg


def _dsg_with_one_object():
    G = spark_dsg.DynamicSceneGraph()
    attrs = spark_dsg.ObjectNodeAttributes()
    attrs.position = np.array([1.0, 2.0, 3.0])
    attrs.semantic_label = 2
    attrs.name = "box"
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 0).value, attrs)
    return G


@pytest.fixture()
def captured(monkeypatch):
    """Capture the node dicts the object ingest would insert."""
    seen = []

    def fake_insert(db, layer_label, node_dicts):
        seen.append((layer_label, node_dicts))

    monkeypatch.setattr("heracles.graph_interface.insert_nodes_to_db", fake_insert)
    return seen


def test_ingested_objects_are_stamped_trusted_prior(captured):
    add_objects_from_dsg(_dsg_with_one_object(), None, db=None,
                         object_labelspace={"2": "box"})

    assert len(captured) == 1
    layer_label, nodes = captured[0]
    assert layer_label == constants.OBJECTS
    assert len(nodes) == 1
    assert nodes[0][constants.ADMISSION_STATUS] == constants.TRUSTED_PRIOR
    assert (nodes[0][constants.ADMISSION_POLICY_VERSION]
            == constants.PRIOR_MAP_POLICY_VERSION)


def test_ingest_does_not_overwrite_an_existing_admission(captured):
    """setdefault, not assignment: a node dict that already carries a decision
    (e.g. a re-ingest of a graph built from admitted detections) keeps it."""
    import heracles.graph_interface as gi

    original = gi.node_to_dict

    def tagged(node, **kw):
        d = original(node, **kw)
        d[constants.ADMISSION_STATUS] = constants.ADMITTED
        return d

    gi.node_to_dict = tagged
    try:
        add_objects_from_dsg(_dsg_with_one_object(), None, db=None,
                             object_labelspace={"2": "box"})
    finally:
        gi.node_to_dict = original

    _, nodes = captured[0]
    assert nodes[0][constants.ADMISSION_STATUS] == constants.ADMITTED


def test_admission_schema_constants_are_stable():
    """These strings are a persisted schema and are mirrored by
    ``agentic_navigation.evidence.admission`` (which drift-guards against this
    module). Changing a value is a migration, not an edit."""
    assert constants.ADMISSION_STATUS == "admission_status"
    assert constants.ADMISSION_POLICY_VERSION == "admission_policy_version"
    assert constants.ADMISSION_OBSERVATIONS == "admission_observation_ids"
    assert constants.ADMISSION_REASON == "admission_reason"
    assert constants.TRUSTED_PRIOR == "trusted_prior"
    assert constants.CANDIDATE == "candidate"
    assert constants.ADMITTED == "admitted"
    assert constants.REJECTED == "rejected"
    assert constants.PRIOR_MAP_POLICY_VERSION == "prior_map_legacy_v0"
