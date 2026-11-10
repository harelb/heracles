"""Agent (keyframe) nodes: spark_dsg agent layers -> Neo4j ``Agent`` rows.

The database tests require a disposable Neo4j instance (its contents are
cleared). Configure it with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import pytest
from dsg_fixture import build_dsg

from heracles import constants
from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    gi.initialize_db(wrapper)
    yield wrapper
    wrapper.close()


def test_agent_to_dict():
    G = build_dsg()
    rows = {r["nodeSymbol"]: r for r in map(gi.agent_to_dict, gi._agent_nodes(G))}
    assert set(rows) == {"a0", "a1"}  # objects in partition 0 are not agents
    assert (rows["a1"]["pos_x"], rows["a1"]["pos_y"], rows["a1"]["pos_z"]) == (4, 5, 6)
    assert rows["a1"]["rot_w"] == 1.0
    assert rows["a1"]["timestamp_ns"] == 2000
    assert "timestamp_ns" not in rows["a0"]  # unset stamp


def test_spark_dsg_to_db_writes_agents(db):
    G = build_dsg()
    gi.spark_dsg_to_db(G, db)

    records, _, _ = db.execute(
        f"MATCH (a:{constants.AGENTS}) "
        "RETURN a.nodeSymbol AS s, a.center AS c, a.timestamp_ns AS t ORDER BY s"
    )
    assert [r["s"] for r in records] == ["a0", "a1"]
    assert tuple(records[1]["c"]) == (4.0, 5.0, 6.0)
    assert records[0]["t"] is None and records[1]["t"] == 2000

    # Agents are not part of the reconstructed graph.
    G2 = gi.db_to_spark_dsg(db)
    assert G2.num_nodes() == G.num_nodes() - 2


def test_add_agents_from_dsg_is_idempotent(db):
    G = build_dsg()
    assert gi.add_agents_from_dsg(G, db) == 2
    assert gi.add_agents_from_dsg(G, db) == 2
    records, _, _ = db.execute(f"MATCH (a:{constants.AGENTS}) RETURN count(a) AS n")
    assert records[0]["n"] == 2
