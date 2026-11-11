"""Agent (keyframe) nodes: spark_dsg agent layers -> Neo4j ``Agent`` rows.

The database tests require a disposable Neo4j instance (its contents are
cleared). Configure it with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os

import numpy as np
import pytest
import spark_dsg
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


def test_agent_edges():
    trajectory, place_agent = gi.agent_edges(build_dsg())
    assert trajectory == [{"from": "a0", "to": "a1"}]
    assert place_agent == [{"from": "p0", "to": "a1", "label": constants.PLACES}]


def test_spark_dsg_to_db_writes_agent_edges(db):
    G = build_dsg()
    gi.spark_dsg_to_db(G, db)
    gi.add_agents_from_dsg(G, db)  # again: MERGE adds no duplicates
    records, _, _ = db.execute(
        f"MATCH (a:{constants.AGENTS})-[:AGENT_CONNECTED]->(b:{constants.AGENTS}) "
        "RETURN a.nodeSymbol AS f, b.nodeSymbol AS t"
    )
    assert [(r["f"], r["t"]) for r in records] == [("a0", "a1")]
    records, _, _ = db.execute(
        f"MATCH (p:{constants.PLACES})-[:CONTAINS]->(a:{constants.AGENTS}) "
        "RETURN p.nodeSymbol AS f, a.nodeSymbol AS t"
    )
    assert [(r["f"], r["t"]) for r in records] == [("p0", "a1")]


def _with_mesh_place(G):
    """Put agent a0 in a mesh place (layer 3, partition 1), as Hydra does."""
    mp = spark_dsg.TravNodeAttributes()
    mp.position = np.array([0.5, 0.5, 0.0])
    mp_id = spark_dsg.NodeSymbol("t", 0).value
    G.add_node(spark_dsg.DsgLayers.MESH_PLACES, mp_id, mp)
    G.insert_edge(mp_id, spark_dsg.NodeSymbol("a", 0).value)
    return G


def test_agent_edges_to_mesh_places():
    _, place_agent = gi.agent_edges(_with_mesh_place(build_dsg()))
    assert sorted(place_agent, key=lambda e: e["to"]) == [
        {"from": "t0", "to": "a0", "label": constants.MESH_PLACES},
        {"from": "p0", "to": "a1", "label": constants.PLACES},
    ]


def test_spark_dsg_to_db_writes_mesh_place_agent_edges(db):
    gi.spark_dsg_to_db(_with_mesh_place(build_dsg()), db)
    records, _, _ = db.execute(
        f"MATCH (p:{constants.MESH_PLACES})-[:CONTAINS]->(a:{constants.AGENTS}) "
        "RETURN p.nodeSymbol AS f, a.nodeSymbol AS t"
    )
    assert [(r["f"], r["t"]) for r in records] == [("t0", "a0")]
