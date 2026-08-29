"""The evidence-admission gate's schema + the prior-map ingest stamp (G1).

Objects arriving through ``add_objects_from_dsg`` come from a COMMITTED scene
graph, so they are *trusted priors*: readers downstream refuse to plan on an
:Object that carries no explicit admission and looks detector-written, and this
stamp is what keeps an ingested map plannable by declaration rather than by a
reader's default.

The stamp must not be able to *undo* a grading decision, so the round trip is
checked against a real Neo4j (skipped when none is configured); the rest is
stubbed at ``insert_nodes_to_db`` and asserts on what the ingest hands it.
"""

import os

import numpy as np
import pytest
import spark_dsg

from heracles import constants
from heracles.graph_interface import add_objects_from_dsg

_TESTCONTAINERS_AVAILABLE = False
try:
    from testcontainers.neo4j import Neo4jContainer  # type: ignore

    _TESTCONTAINERS_AVAILABLE = True
except ImportError:
    pass

_NEO4J_ENV_URI = os.environ.get("NEO4J_URI", "")


def _dsg_with_objects(*indices):
    G = spark_dsg.DynamicSceneGraph()
    for i in indices:
        attrs = spark_dsg.ObjectNodeAttributes()
        attrs.position = np.array([1.0 + i, 2.0, 3.0])
        attrs.semantic_label = 2
        attrs.name = "box"
        G.add_node(
            spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", i).value, attrs
        )
    return G


def _dsg_with_one_object():
    return _dsg_with_objects(0)


@pytest.fixture()
def captured(monkeypatch):
    """Capture the node dicts the object ingest would insert."""
    seen = []

    def fake_insert(db, layer_label, node_dicts, default_properties=None):
        seen.append((layer_label, node_dicts, default_properties))

    monkeypatch.setattr("heracles.graph_interface.insert_nodes_to_db", fake_insert)
    return seen


def test_ingested_objects_are_stamped_trusted_prior(captured):
    add_objects_from_dsg(_dsg_with_one_object(), None, db=None,
                         object_labelspace={"2": "box"})

    assert len(captured) == 1
    layer_label, nodes, defaults = captured[0]
    assert layer_label == constants.OBJECTS
    assert len(nodes) == 1
    assert defaults[constants.ADMISSION_STATUS] == constants.TRUSTED_PRIOR
    assert (defaults[constants.ADMISSION_POLICY_VERSION]
            == constants.PRIOR_MAP_POLICY_VERSION)


def test_the_stamp_is_not_written_into_the_node_payload(captured):
    """The stamp is a DEFAULT, not a property of the ingested node.

    It used to be setdefault-ed onto a dict node_to_dict had just built, so it
    was unconditionally present and rode along in insert_nodes_to_db's
    "SET n += node" -- which overwrites. Keeping it out of the payload is what
    makes the coalesce in insert_nodes_to_db reachable.
    """
    add_objects_from_dsg(_dsg_with_one_object(), None, db=None,
                         object_labelspace={"2": "box"})

    _, nodes, defaults = captured[0]
    assert constants.ADMISSION_STATUS not in nodes[0]
    assert constants.ADMISSION_POLICY_VERSION not in nodes[0]


# ---------------------------------------------------------------------------
# against a real Neo4j: re-ingest must not undo a grading decision
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    from heracles.query_interface import Neo4jWrapper

    if _TESTCONTAINERS_AVAILABLE:
        container = Neo4jContainer("neo4j:5.18")
        container.start()
        wrapper = Neo4jWrapper(
            container.get_connection_url(),
            ("neo4j", container.NEO4J_ADMIN_PASSWORD),
            atomic_queries=True,
        )
        wrapper.connect()
        yield wrapper
        wrapper.close()
        container.stop()
        return
    if not _NEO4J_ENV_URI:
        pytest.skip(
            "needs Neo4j: set NEO4J_URI (and NEO4J_AUTH) or install "
            "testcontainers-neo4j"
        )
    auth_str = os.environ.get("NEO4J_AUTH", "neo4j/neo4j")
    user, password = auth_str.split("/", 1)
    wrapper = Neo4jWrapper(_NEO4J_ENV_URI, (user, password), atomic_queries=True)
    wrapper.connect()
    yield wrapper
    wrapper.close()


@pytest.fixture()
def clean_db(db):
    db.execute("MATCH (n) DETACH DELETE n")
    yield db


def _admission(db, node_symbol):
    records, _, _ = db.execute(
        f"""
        MATCH (n:{constants.OBJECTS} {{nodeSymbol: $sym}})
        RETURN n.{constants.ADMISSION_STATUS} AS status,
               n.{constants.ADMISSION_POLICY_VERSION} AS policy,
               n.{constants.ADMISSION_REASON} AS reason
        """,
        sym=node_symbol,
    )
    assert len(records) == 1, f"expected exactly one {node_symbol}"
    return records[0]


def test_reingest_does_not_overwrite_a_graded_admission(clean_db):
    """A node an authority REJECTED must not come back trusted_prior.

    "MERGE ... SET n += node" overwrites, and the stamp was unconditionally in
    that payload, so re-ingesting the prior map silently re-admitted a rejected
    object -- while leaving its admission_reason attached, so the node ended up
    claiming to be a trusted prior *and* carrying the reason it was refused.
    """
    db = clean_db
    db.execute(
        f"""
        CREATE (n:{constants.OBJECTS} {{
            nodeSymbol: $sym,
            {constants.ADMISSION_STATUS}: $status,
            {constants.ADMISSION_REASON}: $reason
        }})
        """,
        sym="O0",
        status=constants.REJECTED,
        reason="failed the pre-spawn probe",
    )

    # O0 is the graded node; O1 is new to the database.
    add_objects_from_dsg(
        _dsg_with_objects(0, 1), None, db=db, object_labelspace={"2": "box"}
    )

    graded = _admission(db, "O0")
    assert graded["status"] == constants.REJECTED
    assert graded["reason"] == "failed the pre-spawn probe"
    assert graded["policy"] == constants.PRIOR_MAP_POLICY_VERSION, (
        "a node that carried no policy version should still be backfilled"
    )

    fresh = _admission(db, "O1")
    assert fresh["status"] == constants.TRUSTED_PRIOR
    assert fresh["policy"] == constants.PRIOR_MAP_POLICY_VERSION
    assert fresh["reason"] is None


def test_fresh_ingest_stamps_trusted_prior_in_the_db(clean_db):
    db = clean_db
    add_objects_from_dsg(
        _dsg_with_one_object(), None, db=db, object_labelspace={"2": "box"}
    )
    stamped = _admission(db, "O0")
    assert stamped["status"] == constants.TRUSTED_PRIOR
    assert stamped["policy"] == constants.PRIOR_MAP_POLICY_VERSION


def test_reingest_is_idempotent_for_a_trusted_prior(clean_db):
    db = clean_db
    for _ in range(2):
        add_objects_from_dsg(
            _dsg_with_one_object(), None, db=db, object_labelspace={"2": "box"}
        )
    stamped = _admission(db, "O0")
    assert stamped["status"] == constants.TRUSTED_PRIOR


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
