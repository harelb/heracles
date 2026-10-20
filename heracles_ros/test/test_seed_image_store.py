"""seed_db loads the run's image store, so SAM3 / the console find the keyframes.

No ROS imports. The DB test needs an explicit throwaway Neo4j (HERACLES_TEST_NEO4J_URI);
it never falls back to the default port because seeding wipes the database.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "heracles", "tests"))
pytest.importorskip("hydra_image_store")

from image_store_fixture import build_dsg, build_store  # noqa: E402

from heracles_ros.map_state import find_run_image_store, seed_db  # noqa: E402


def _run_dir(tmp_path):
    run = tmp_path / "run"
    build_store(run)
    dsg = run / "hydra" / "backend" / "dsg.json"
    dsg.parent.mkdir(parents=True)
    build_dsg().save(str(dsg), include_mesh=False)
    return run, dsg


def test_find_run_image_store(tmp_path):
    run, dsg = _run_dir(tmp_path)
    assert find_run_image_store(str(dsg)) == str(run / "image_store")
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "dsg.json").write_text("{}")
    assert find_run_image_store(str(flat / "dsg.json")) is None
    (flat / "image_store").mkdir()  # a directory without store.sqlite is not a store
    assert find_run_image_store(str(flat / "dsg.json")) is None


URI = os.environ.get("HERACLES_TEST_NEO4J_URI")
AUTH = (
    os.environ.get("HERACLES_TEST_NEO4J_USER", "neo4j"),
    os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"),
)


def _db():
    from heracles.query_interface import Neo4jWrapper

    if not URI:
        pytest.skip("set HERACLES_TEST_NEO4J_URI to a throwaway Neo4j")
    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"No test Neo4j at {URI}: {exc}")
    return db


def test_seed_db_records_image_store(tmp_path):
    run, dsg = _run_dir(tmp_path)
    _db().close()
    version = seed_db(URI, AUTH, str(dsg))
    assert version == 1
    db = _db()
    try:
        rows = db.query("MATCH (m:_GraphMetadata {key:'image_store'}) RETURN m.image_store_paths AS p")
        assert [os.path.realpath(p) for p in rows[0]["p"]] == [os.path.realpath(run / "image_store")]
        a0 = db.query("MATCH (a:Agent) WHERE a.nodeSymbol = 'a0' RETURN a.rgb_path AS rgb, a.depth_path AS depth")
        assert a0 and os.path.exists(a0[0]["rgb"]) and os.path.exists(a0[0]["depth"])
        n_obs = db.query("MATCH (:Object)-[:HAS_OBSERVATION]->(o:Observation) RETURN count(o) AS n")[0]["n"]
        assert n_obs >= 1
    finally:
        db.close()
    # explicit opt-out
    seed_db(URI, AUTH, str(dsg), image_store=None)
    db = _db()
    try:
        assert not db.query("MATCH (m:_GraphMetadata {key:'image_store'}) RETURN m")
    finally:
        db.close()
