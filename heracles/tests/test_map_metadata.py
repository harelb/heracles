"""_GraphMetadata written for viewers: the mesh path.

Requires a disposable Neo4j instance (its contents are cleared). Configure it
with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import pytest

from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    wrapper.execute("MATCH (n) DETACH DELETE n")
    yield wrapper
    wrapper.close()


def test_mesh_path_absent(db):
    assert gi.read_mesh_path(db) == (None, None)


def test_mesh_path_roundtrip(db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gi.store_mesh_path(db, "mesh.ply")
    assert gi.read_mesh_path(db) == (str(tmp_path / "mesh.ply"), "ply")

    # Storing again replaces the path; an explicit format wins.
    gi.store_mesh_path(db, tmp_path / "other.mesh", mesh_format="obj")
    assert gi.read_mesh_path(db) == (str(tmp_path / "other.mesh"), "obj")
    records, _, _ = db.execute(
        "MATCH (m:_GraphMetadata {key: 'mesh'}) RETURN count(m) AS n"
    )
    assert records[0]["n"] == 1
