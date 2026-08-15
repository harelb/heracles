"""Tests for evidence-provenance helpers: Observation -> frame linking,
Agent timestamp backfill, and CameraCalib materialization.

These tests require a running Neo4j instance (same guards/fixture pattern as
test_embedding_helpers.py): testcontainers-neo4j when available, else the
NEO4J_URI / NEO4J_AUTH environment variables.
"""

import json
import os

import pytest

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

from heracles.graph_interface import (
    attach_camera_calibs,
    backfill_agent_timestamps,
    insert_observations_to_db,
    link_observations_to_frames,
)
from heracles.query_interface import Neo4jWrapper


@pytest.fixture(scope="module")
def db():
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
        wrapper = Neo4jWrapper(uri, (user, password), atomic_queries=True)
        wrapper.connect()
        yield wrapper
        wrapper.close()


@pytest.fixture(autouse=True)
def clean_db(db):
    db.execute("MATCH (n) DETACH DELETE n")
    yield


def _count(db, cypher, **kw):
    records, _, _ = db.execute(cypher, **kw)
    return records[0]["n"]


# ---------------------------------------------------------------------------
# link_observations_to_frames
# ---------------------------------------------------------------------------


def test_link_observations_to_frames_joins_by_timestamp(db):
    db.execute(
        "CREATE (:Agent {nodeSymbol: 'a0', timestamp_ns: 111}),"
        "       (:SubKeyframe {nodeSymbol: 's0', timestamp_ns: 222}),"
        "       (:TrajectoryFrame {nodeSymbol: 'F(1)', timestamp_ns: 111})"
    )
    insert_observations_to_db(
        db,
        [
            {
                "nodeSymbol": "O(1)_111",
                "timestamp_ns": 111,
                "mask_file": "m.png",
                "bbox_2d_min_x": 0,
                "bbox_2d_min_y": 0,
                "bbox_2d_max_x": 5,
                "bbox_2d_max_y": 5,
                "score": 0.87,
                "detector": "sam3",
                "mechanism": "sam3_query",
            },
            {
                "nodeSymbol": "O(1)_222",
                "timestamp_ns": 222,
                "mask_file": "m2.png",
                "bbox_2d_min_x": 0,
                "bbox_2d_min_y": 0,
                "bbox_2d_max_x": 5,
                "bbox_2d_max_y": 5,
            },
            {
                "nodeSymbol": "O(1)_999",
                "timestamp_ns": 999,  # no matching frame anywhere
                "mask_file": "",
                "bbox_2d_min_x": 0,
                "bbox_2d_min_y": 0,
                "bbox_2d_max_x": 1,
                "bbox_2d_max_y": 1,
            },
        ],
    )

    n = link_observations_to_frames(db)
    assert n == 3  # a0 + F(1) for ts 111, s0 for ts 222

    assert (
        _count(
            db,
            "MATCH (:Observation {nodeSymbol:'O(1)_111'})-[:OBSERVED_IN]->(f:Agent {nodeSymbol:'a0'}) "
            "RETURN count(f) AS n",
        )
        == 1
    )
    assert (
        _count(
            db,
            "MATCH (:Observation {nodeSymbol:'O(1)_222'})-[:OBSERVED_IN]->(f:SubKeyframe) "
            "RETURN count(f) AS n",
        )
        == 1
    )
    assert (
        _count(
            db,
            "MATCH (:Observation {nodeSymbol:'O(1)_999'})-[:OBSERVED_IN]->(f) RETURN count(f) AS n",
        )
        == 0
    )

    # Idempotent: rerunning must not duplicate relationships.
    link_observations_to_frames(db)
    assert _count(db, "MATCH ()-[r:OBSERVED_IN]->() RETURN count(r) AS n") == 3


def test_observation_score_survives_scoreless_reinsert(db):
    obs = {
        "nodeSymbol": "O(2)_5",
        "timestamp_ns": 5,
        "mask_file": "m.png",
        "bbox_2d_min_x": 0,
        "bbox_2d_min_y": 0,
        "bbox_2d_max_x": 2,
        "bbox_2d_max_y": 2,
        "score": 0.5,
        "detector": "sam3",
        "mechanism": "sam3_query",
    }
    insert_observations_to_db(db, [obs])
    # Re-insert the same observation without provenance fields (khronos-style
    # dict) — coalesce must keep the previously written values.
    insert_observations_to_db(db, [{k: v for k, v in obs.items() if k not in ("score", "detector", "mechanism")}])

    records, _, _ = db.execute(
        "MATCH (o:Observation {nodeSymbol:'O(2)_5'}) RETURN o.score AS s, o.detector AS d, o.mechanism AS m"
    )
    assert records[0]["s"] == 0.5
    assert records[0]["d"] == "sam3"
    assert records[0]["m"] == "sam3_query"


# ---------------------------------------------------------------------------
# backfill_agent_timestamps
# ---------------------------------------------------------------------------


def test_backfill_agent_timestamps(db):
    db.execute(
        "CREATE (:Agent {nodeSymbol: 'a0', image_folder: '/data/agents/agent_123456789'}),"
        "       (:Agent {nodeSymbol: 'a1', image_folder: '/data/agents/not_numeric'}),"
        "       (:Agent {nodeSymbol: 'a2', image_folder: '/data/agents/agent_42', timestamp_ns: 99})"
    )
    n = backfill_agent_timestamps(db)
    assert n == 1  # a1 unparseable, a2 already has one

    records, _, _ = db.execute(
        "MATCH (a:Agent) RETURN a.nodeSymbol AS ns, a.timestamp_ns AS ts ORDER BY ns"
    )
    by_ns = {r["ns"]: r["ts"] for r in records}
    assert by_ns == {"a0": 123456789, "a1": None, "a2": 99}


# ---------------------------------------------------------------------------
# attach_camera_calibs
# ---------------------------------------------------------------------------


def _write_calib(dirpath):
    calib = {
        "fx": 380.0,
        "fy": 380.0,
        "cx": 320.0,
        "cy": 240.0,
        "width": 640,
        "height": 480,
        "depth_scale": 1e-3,
        "body_T_sensor": [[1, 0, 0, 0.1], [0, 1, 0, 0.0], [0, 0, 1, 0.2], [0, 0, 0, 1]],
    }
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, "camera_calib.json"), "w") as f:
        json.dump(calib, f)


def test_attach_camera_calibs(db, tmp_path):
    agents_dir = str(tmp_path / "agents")
    _write_calib(agents_dir)
    db.execute(
        "CREATE (:Agent {nodeSymbol: 'a0', image_folder: $p1}),"
        "       (:SubKeyframe {nodeSymbol: 's0', image_folder: $p2}),"
        "       (:TrajectoryFrame {nodeSymbol: 'F(1)', path: $p3}),"
        "       (:Agent {nodeSymbol: 'a_nocalib', image_folder: '/nonexistent/agent_7'})",
        p1=os.path.join(agents_dir, "agent_1"),
        p2=os.path.join(agents_dir, "subkf_2"),
        p3=os.path.join(agents_dir, "agent_1_rgb.jpg"),
    )

    n_edges = attach_camera_calibs(db)
    assert n_edges == 3  # a_nocalib has no calib file -> skipped

    # One shared CameraCalib node (same dir/content hash) with full properties.
    records, _, _ = db.execute(
        "MATCH (c:CameraCalib) RETURN c.fx AS fx, c.width AS w, c.depth_scale AS ds, "
        "c.body_T_sensor AS bts, count(c) AS n"
    )
    assert records[0]["n"] == 1
    assert records[0]["fx"] == 380.0
    assert records[0]["w"] == 640
    assert records[0]["ds"] == 1e-3
    assert len(records[0]["bts"]) == 16

    assert _count(db, "MATCH ()-[r:HAS_CALIB]->(:CameraCalib) RETURN count(r) AS n") == 3
    assert (
        _count(
            db,
            "MATCH (:Agent {nodeSymbol:'a_nocalib'})-[r:HAS_CALIB]->() RETURN count(r) AS n",
        )
        == 0
    )

    # Idempotent: node and edge counts stable on rerun.
    attach_camera_calibs(db)
    assert _count(db, "MATCH (c:CameraCalib) RETURN count(c) AS n") == 1
    assert _count(db, "MATCH ()-[r:HAS_CALIB]->() RETURN count(r) AS n") == 3
