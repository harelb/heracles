"""Tests for the epistemic-layer schema: create_epistemic_indexes.

Same DB guards/fixture pattern as test_provenance.py: testcontainers-neo4j when
available, else the NEO4J_URI / NEO4J_AUTH environment variables.
"""

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

import neo4j.exceptions

from heracles.graph_interface import create_epistemic_indexes, initialize_db
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


def test_create_epistemic_indexes_idempotent(db):
    create_epistemic_indexes(db)
    create_epistemic_indexes(db)  # IF NOT EXISTS: second call must not raise

    records, _, _ = db.execute("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
    names = records[0]["names"]
    assert "search_episode_id_unique" in names
    assert "assertion_id_unique" in names


def test_episode_id_uniqueness_enforced(db):
    create_epistemic_indexes(db)
    db.execute("CREATE (:SearchEpisode {episode_id: 'ep_1'})")
    with pytest.raises(neo4j.exceptions.ConstraintError):
        db.execute("CREATE (:SearchEpisode {episode_id: 'ep_1'})")


def test_assertion_id_uniqueness_enforced(db):
    create_epistemic_indexes(db)
    db.execute("CREATE (:Assertion {assertion_id: 'assert_1'})")
    with pytest.raises(neo4j.exceptions.ConstraintError):
        db.execute("CREATE (:Assertion {assertion_id: 'assert_1'})")


def test_constraints_survive_initialize_db_wipe(db):
    create_epistemic_indexes(db)
    db.execute("CREATE (:SearchEpisode {episode_id: 'ep_1'})")
    initialize_db(db)  # DETACH DELETE + index rebuild — constraints must remain
    records, _, _ = db.execute("MATCH (n:SearchEpisode) RETURN count(n) AS n")
    assert records[0]["n"] == 0
    db.execute("CREATE (:SearchEpisode {episode_id: 'ep_2'})")
    with pytest.raises(neo4j.exceptions.ConstraintError):
        db.execute("CREATE (:SearchEpisode {episode_id: 'ep_2'})")
