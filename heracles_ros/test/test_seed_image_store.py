"""seed_db loads the run's image store and mesh, so the console / SAM3 find the images.

Uses heracles' own tiny scene graph and image store (heracles/tests). The DB
test wipes the dedicated test Neo4j (HERACLES_TEST_NEO4J_URI) and skips without it.
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("hydra_python.image_store")
pytest.importorskip("imageio.v3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "heracles" / "tests"))

from dsg_fixture import build_dsg  # noqa: E402
from heracles.graph_interface import read_mesh_path  # noqa: E402
from heracles.query_interface import Neo4jWrapper  # noqa: E402
from test_image_store_ingest import build_store  # noqa: E402

from heracles_ros.map_state import (  # noqa: E402
    find_run_image_store,
    find_run_mesh,
    seed_db,
)

URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


def _run_dir(tmp_path):
    run = tmp_path / "run"
    build_store(run)
    dsg = run / "hydra" / "backend" / "dsg.json"
    dsg.parent.mkdir(parents=True)
    build_dsg().save(str(dsg), include_mesh=False)
    (dsg.parent / "mesh.ply").write_text("ply")
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


def test_find_run_mesh(tmp_path):
    backend = tmp_path / "hydra" / "backend"
    backend.mkdir(parents=True)
    dsg = backend / "dsg.json"
    dsg.write_text("{}")
    assert find_run_mesh(str(dsg)) is None
    (backend / "mesh.ply").write_text("ply")
    assert find_run_mesh(str(dsg)) == str(backend / "mesh.ply")


@pytest.fixture
def db():
    db = Neo4jWrapper(URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:
        pytest.skip(f"No test Neo4j at {URI}: {exc}")
    db.query("MATCH (n) DETACH DELETE n")
    yield db
    db.close()


def _store_paths(db):
    rows = db.query(
        "MATCH (m:_GraphMetadata {key: 'image_store'}) RETURN m.image_store_paths AS p"
    )
    return rows[0]["p"] if rows else None


def test_seed_db_loads_the_runs_image_store_and_mesh(db, tmp_path):
    run, dsg = _run_dir(tmp_path)
    seed_db(URI, AUTH, str(dsg))

    assert [os.path.realpath(p) for p in _store_paths(db)] == [
        os.path.realpath(run / "image_store")
    ]
    obs = db.query(
        "MATCH (:Object)-[:HAS_OBSERVATION]->(o:Observation) RETURN o.rgb_path AS p"
    )
    assert obs and all(os.path.isfile(r["p"]) for r in obs)
    assert read_mesh_path(db)[0] == str(dsg.parent / "mesh.ply")

    # explicit opt-out: the reload wipes the previous map's images
    seed_db(URI, AUTH, str(dsg), image_store=None)
    assert _store_paths(db) is None
    assert not db.query("MATCH (o:Observation) RETURN o LIMIT 1")
