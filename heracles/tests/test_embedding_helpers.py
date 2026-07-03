"""Tests for embedding indices, version counter, and TrajectoryFrame helpers.

These tests require a running Neo4j instance. They use testcontainers-neo4j when
available, or fall back to the NEO4J_URI / NEO4J_AUTH environment variables.

Skip cleanly when neither is available.
"""

import os
import time

import pytest

# ---------------------------------------------------------------------------
# Availability guards — skip at collection time if nothing is usable
# ---------------------------------------------------------------------------

_TESTCONTAINERS_AVAILABLE = False
try:
    from testcontainers.neo4j import Neo4jContainer  # type: ignore

    _TESTCONTAINERS_AVAILABLE = True
except ImportError:
    pass

_NEO4J_ENV_URI = os.environ.get("NEO4J_URI", "")

if not _TESTCONTAINERS_AVAILABLE and not _NEO4J_ENV_URI:
    pytest.skip(
        "Skipping: testcontainers-neo4j is not installed and NEO4J_URI env var is not set.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Imports from heracles (always available inside the package)
# ---------------------------------------------------------------------------

from heracles.graph_interface import (
    bump_version,
    create_vector_indexes,
    insert_frame_edges,
    insert_trajectory_frames_to_db,
    query_similar_nodes,
    read_version,
    set_node_embedding,
)
from heracles.query_interface import Neo4jWrapper


# ---------------------------------------------------------------------------
# Fixture: Neo4j connection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db():
    """Provide a connected Neo4jWrapper for the test module.

    Uses testcontainers-neo4j if available, otherwise falls back to
    NEO4J_URI / NEO4J_AUTH environment variables.
    """
    if _TESTCONTAINERS_AVAILABLE:
        container = Neo4jContainer("neo4j:5.18")
        container.start()
        uri = container.get_connection_url()
        auth = ("neo4j", container.NEO4J_ADMIN_PASSWORD)
        wrapper = Neo4jWrapper(uri, auth, atomic_queries=True)
        wrapper.connect()
        yield wrapper
        wrapper.close()
        container.stop()
    else:
        uri = _NEO4J_ENV_URI
        auth_str = os.environ.get("NEO4J_AUTH", "neo4j/neo4j")
        user, password = auth_str.split("/", 1)
        auth = (user, password)
        wrapper = Neo4jWrapper(uri, auth, atomic_queries=True)
        wrapper.connect()
        yield wrapper
        wrapper.close()


@pytest.fixture(autouse=True)
def clean_db(db):
    """Wipe the database before each test to ensure isolation."""
    db.execute("MATCH (n) DETACH DELETE n")
    # Drop vector indexes to allow re-creation with potentially different dims
    for idx in ("object_embedding", "observation_embedding", "trajectoryframe_embedding"):
        try:
            db.execute(f"DROP INDEX {idx} IF EXISTS")
        except Exception:
            pass
    yield


# ---------------------------------------------------------------------------
# Helper: wait for a vector index to become online
# ---------------------------------------------------------------------------


def _await_index(db, index_name: str, timeout: float = 10.0) -> None:
    """Poll until the named index is ONLINE or timeout is reached.

    Uses ``SHOW INDEXES`` without parameters (parameter binding is not
    supported for ``SHOW`` in older Neo4j versions), then filters in Python.
    Falls back to a hard sleep on error.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            records, _, _ = db.execute("SHOW INDEXES")
            for row in records:
                if row.get("name") == index_name and row.get("state") == "ONLINE":
                    return
        except Exception:
            pass
        time.sleep(0.2)
    # Last-resort hard wait
    time.sleep(1.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_vector_indexes_and_query(db):
    """Insert Object nodes, create vector indexes, set a known embedding, query top-1."""
    import random

    rng = random.Random(42)

    # Insert 5 Object nodes
    for i in range(5):
        vec = [rng.gauss(0, 1) for _ in range(4)]
        db.execute(
            "CREATE (n:Object {nodeSymbol: $ns, embedding: $v})",
            ns=f"O({i})",
            v=vec,
        )

    # Create vector indexes (idempotent)
    create_vector_indexes(db, dim=4, model_name="test")

    # Wait for the object_embedding index to become ONLINE
    _await_index(db, "object_embedding")

    # Plant a known vector for O(1) and query for that exact vector
    known_vec = [1.0, 0.0, 0.0, 0.0]
    set_node_embedding(db, "O(1)", known_vec, model_name="test")

    results = query_similar_nodes(db, "Object", known_vec, k=5)
    assert len(results) > 0, "Expected at least one result from vector query"
    top_symbol, top_score = results[0]
    assert top_symbol == "O(1)", f"Expected O(1) as top-1, got {top_symbol}"
    assert top_score > 0.99, f"Expected near-perfect cosine similarity, got {top_score}"


def test_create_vector_indexes_idempotent(db):
    """Calling create_vector_indexes twice should not raise."""
    create_vector_indexes(db, dim=4, model_name="test")
    create_vector_indexes(db, dim=4, model_name="test")  # should be a no-op


def test_bump_version_monotonic(db):
    """bump_version should return 1, 2, 3, ... on successive calls."""
    assert read_version(db) == 0, "Version should be 0 before any bump"
    for expected in range(1, 6):
        v = bump_version(db)
        assert v == expected, f"Expected version {expected}, got {v}"
    assert read_version(db) == 5


def test_insert_trajectory_frames(db):
    """insert_trajectory_frames_to_db should create the expected number of nodes."""
    frames = [
        {
            "nodeSymbol": f"TF({i})",
            "timestamp_ns": 1_000_000_000 + i,
            "path": f"/images/frame_{i:04d}.jpg",
            "pos_x": float(i),
            "pos_y": 0.0,
            "pos_z": 0.0,
            "pose_qw": 1.0,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "place_id": None,
            "embedding": None,
            "embedding_dim": None,
            "embedding_model": None,
        }
        for i in range(7)
    ]
    n = insert_trajectory_frames_to_db(db, frames)
    assert n == 7, f"Expected 7 frames written, got {n}"

    # Verify with a direct count query
    records, _, _ = db.execute("MATCH (t:TrajectoryFrame) RETURN count(t) AS c")
    assert records[0]["c"] == 7


def test_insert_trajectory_frames_idempotent(db):
    """Inserting the same frames twice should not create duplicates."""
    frames = [
        {
            "nodeSymbol": "TF(0)",
            "timestamp_ns": 1_000_000_000,
            "path": "/images/frame_0000.jpg",
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "pose_qw": 1.0,
            "pose_qx": 0.0,
            "pose_qy": 0.0,
            "pose_qz": 0.0,
            "place_id": None,
            "embedding": None,
            "embedding_dim": None,
            "embedding_model": None,
        }
    ]
    insert_trajectory_frames_to_db(db, frames)
    insert_trajectory_frames_to_db(db, frames)
    records, _, _ = db.execute("MATCH (t:TrajectoryFrame) RETURN count(t) AS c")
    assert records[0]["c"] == 1, "MERGE should be idempotent"


def test_insert_frame_edges(db):
    """insert_frame_edges should create OBSERVED_AT and DEPICTS edges."""
    # Create a TrajectoryFrame, a Place, and an Object node
    db.execute("CREATE (f:TrajectoryFrame {nodeSymbol: 'TF(0)'})")
    db.execute("CREATE (p:Place {nodeSymbol: 'p(0)'})")
    db.execute("CREATE (o:Object {nodeSymbol: 'O(0)'})")

    insert_frame_edges(
        db,
        frame_to_place_edges=[("TF(0)", "p(0)")],
        frame_to_object_edges=[("TF(0)", "O(0)")],
    )

    # Verify OBSERVED_AT edge
    records, _, _ = db.execute(
        "MATCH (f:TrajectoryFrame)-[:OBSERVED_AT]->(p:Place) "
        "RETURN f.nodeSymbol AS f_ns, p.nodeSymbol AS p_ns"
    )
    assert len(records) == 1
    assert records[0]["f_ns"] == "TF(0)"
    assert records[0]["p_ns"] == "p(0)"

    # Verify DEPICTS edge
    records, _, _ = db.execute(
        "MATCH (f:TrajectoryFrame)-[:DEPICTS]->(o:Object) "
        "RETURN f.nodeSymbol AS f_ns, o.nodeSymbol AS o_ns"
    )
    assert len(records) == 1
    assert records[0]["f_ns"] == "TF(0)"
    assert records[0]["o_ns"] == "O(0)"


def test_insert_frame_edges_idempotent(db):
    """Calling insert_frame_edges twice should not create duplicate edges."""
    db.execute("CREATE (f:TrajectoryFrame {nodeSymbol: 'TF(0)'})")
    db.execute("CREATE (p:Place {nodeSymbol: 'p(0)'})")

    insert_frame_edges(db, frame_to_place_edges=[("TF(0)", "p(0)")], frame_to_object_edges=[])
    insert_frame_edges(db, frame_to_place_edges=[("TF(0)", "p(0)")], frame_to_object_edges=[])

    records, _, _ = db.execute(
        "MATCH (f:TrajectoryFrame)-[:OBSERVED_AT]->(p:Place) RETURN count(*) AS c"
    )
    assert records[0]["c"] == 1, "MERGE on edges should be idempotent"
