"""A rebuild from the change log keeps the objects' images (an image store map).

Uses heracles' own tiny scene graph and image store (heracles/tests) and the
dedicated test Neo4j (HERACLES_TEST_NEO4J_URI), which it wipes; skips without
either.
"""

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("hydra_python.image_store")
pytest.importorskip("imageio.v3")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "heracles" / "tests"))

from dsg_fixture import build_dsg  # noqa: E402
from heracles.query_interface import Neo4jWrapper  # noqa: E402
from heracles.utils import load_dsg_to_db  # noqa: E402
from test_image_store_ingest import build_store  # noqa: E402

from heracles_ros import change_log  # noqa: E402

URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


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


def _observation_links(db):
    return db.query(
        "MATCH (:Object)-[:HAS_OBSERVATION]->(o:Observation) "
        "RETURN o.nodeSymbol AS o ORDER BY o"
    )


def test_a_rebuild_relinks_the_objects_images(db, tmp_path):
    store_dir = build_store(tmp_path)
    load_dsg_to_db(URI, AUTH, build_dsg(), image_store=str(store_dir))
    before = _observation_links(db)
    assert len(before) == 2
    assert change_log._image_store(db) == [str(store_dir)]

    logged = []
    change_log.rebuild(db, build_dsg(), log=logged.append)

    assert _observation_links(db) == before
    assert not any("lost" in line for line in logged), logged
