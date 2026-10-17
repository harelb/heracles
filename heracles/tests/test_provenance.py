"""Tests for evidence-provenance helpers: Observation -> frame linking,
and CameraCalib materialization from an image store.

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
# attach_camera_calibs / full ingest from an image store
# ---------------------------------------------------------------------------


def test_attach_camera_calibs(db, tmp_path):
    from hydra_image_store import ImageStore
    from image_store_fixture import build_store

    store = ImageStore.open(build_store(tmp_path))
    db.execute(
        "CREATE (:Agent {nodeSymbol: 'a0'}), (:Agent {nodeSymbol: 'a1'}),"
        "       (:SubKeyframe {nodeSymbol: $s_anchored}),"
        "       (:SubKeyframe {nodeSymbol: $s_loose})",
        s_anchored=_subkf_symbols(store)[0],
        s_loose=_subkf_symbols(store)[1],
    )

    n_edges = attach_camera_calibs(db, store)
    assert n_edges == 3  # 2 agents + the anchored sub-keyframe (unanchored has no frame->agent)

    records, _, _ = db.execute(
        "MATCH (c:CameraCalib) RETURN c.sensor_id AS sid, c.fx AS fx, c.width AS w, "
        "c.depth_scale AS ds, c.body_T_sensor AS bts, count(c) AS n"
    )
    assert records[0]["n"] == 1
    assert records[0]["fx"] == 5.0
    assert records[0]["w"] == 8
    assert records[0]["ds"] == 1e-3
    assert len(records[0]["bts"]) == 16
    assert records[0]["sid"] == store.sensors()[0].sensor_id

    assert _count(db, "MATCH ()-[r:HAS_CALIB]->(:CameraCalib) RETURN count(r) AS n") == 3

    # Idempotent: node and edge counts stable on rerun.
    attach_camera_calibs(db, store)
    assert _count(db, "MATCH (c:CameraCalib) RETURN count(c) AS n") == 1
    assert _count(db, "MATCH ()-[r:HAS_CALIB]->() RETURN count(r) AS n") == 3


def _subkf_symbols(store):
    from hydra_image_store import FrameKind
    from heracles.graph_interface import subkeyframe_symbol

    frames = store.frames(kind=FrameKind.SUBKEYFRAME)
    frames.sort(key=lambda f: f.anchor_node_id is None)  # anchored first
    return [subkeyframe_symbol(f.frame_id) for f in frames]


def test_spark_dsg_to_db_with_store_contract(db, tmp_path):
    import spark_dsg
    from heracles.graph_interface import spark_dsg_to_db
    from image_store_fixture import build_dsg, build_store

    G = build_dsg()
    G.set_labelspace(spark_dsg.Labelspace({0: "unknown"}), 2, 0)
    G.metadata.add({"LayerIdToHeraclesLayerStr": {2: "Object"}})
    store_dir = build_store(tmp_path)
    spark_dsg_to_db(G, db, image_store=str(store_dir))

    # Agents: all DSG agents; keyframe image paths + frame ids.
    recs, _, _ = db.execute(
        "MATCH (a:Agent) RETURN a.nodeSymbol AS ns, a.frame_id AS fid, "
        "a.rgb_path AS rgb, a.depth_path AS d, a.timestamp_ns AS ts ORDER BY ns"
    )
    assert [r["ns"] for r in recs] == ["a0", "a1"]
    assert recs[0]["rgb"].startswith(str(store_dir)) and recs[0]["d"] is not None
    assert recs[1]["d"] is None and recs[1]["ts"] == 200
    # Object -> Observations (merged child's track included).
    assert _count(db, "MATCH (:Object)-[:HAS_OBSERVATION]->(o:Observation) RETURN count(o) AS n") == 2
    assert _count(db, "MATCH (n) WHERE n.image_folder IS NOT NULL RETURN count(n) AS n") == 0
    # SubKeyframes + ANCHORED_TO + calibs + metadata.
    assert _count(db, "MATCH (s:SubKeyframe) RETURN count(s) AS n") == 2
    assert _count(db, "MATCH (:SubKeyframe)-[:ANCHORED_TO]->(:Agent) RETURN count(*) AS n") == 1
    assert _count(db, "MATCH (:Agent)-[:HAS_CALIB]->(:CameraCalib) RETURN count(*) AS n") == 2
    recs, _, _ = db.execute(
        "MATCH (m:_GraphMetadata {key:'image_store'}) RETURN m.image_store_paths AS p"
    )
    assert recs[0]["p"] == [str(store_dir)]
    # OBSERVED_IN: observation at ts 100 joins the keyframe agent a0.
    assert _count(db, "MATCH (:Observation)-[:OBSERVED_IN]->(:Agent {nodeSymbol:'a0'}) RETURN count(*) AS n") >= 1
