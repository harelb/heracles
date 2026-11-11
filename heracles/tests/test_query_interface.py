"""Neo4jWrapper against a live database.

Requires a disposable Neo4j instance. Configure it with
HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import pytest

from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    yield wrapper
    wrapper.close()


def test_execute_passes_a_parameters_dict(db):
    records, _, _ = db.execute("RETURN $x AS x", parameters={"x": 7})
    assert records[0]["x"] == 7


def test_execute_passes_keyword_parameters(db):
    records, _, _ = db.execute("RETURN $x AS x", x=7)
    assert records[0]["x"] == 7
