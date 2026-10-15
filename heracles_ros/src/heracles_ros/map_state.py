"""Map-level state kept in Neo4j alongside the scene graph.

Neo4j is the single map at runtime, so anything a planner must know about the
world -- not just the scene graph's layers -- has to live there and travel with
the published graph. Two things do:

* ``map_version``: a counter every map writer bumps after a change. A planner
  that has to act on a change can wait until the graph it holds is at least
  that new, instead of replanning on the graph from before the change.
* ``holding``: which robot holds which object. The ``HOLDS`` edges never leave
  the database otherwise, since the published graph carries only the spatial
  layers.

Both are written into the graph's metadata under ``METADATA_KEY``. Robot pose
updates do not bump the version: they are not changes to the map.
"""

from __future__ import annotations

from typing import Dict, List

import spark_dsg
from heracles.query_interface import Neo4jWrapper
from heracles.utils import load_dsg_to_db

METADATA_KEY = "heracles"
_MAP_STATE = "_MapState"


def get_map_version(db) -> int:
    rows = db.query(f"MATCH (m:{_MAP_STATE}) RETURN m.version AS v")
    return int(rows[0]["v"]) if rows and rows[0]["v"] is not None else 0


def bump_map_version(db) -> int:
    rows = db.query(
        f"MERGE (m:{_MAP_STATE}) "
        "SET m.version = coalesce(m.version, 0) + 1 "
        "RETURN m.version AS v"
    )
    return int(rows[0]["v"])


def get_holding(db) -> Dict[str, List[str]]:
    rows = db.query(
        "MATCH (r:Robot)-[:HOLDS]->(o:Object) "
        "RETURN r.name AS robot, o.nodeSymbol AS object ORDER BY robot, object"
    )
    holding: Dict[str, List[str]] = {}
    for row in rows:
        holding.setdefault(row["robot"], []).append(row["object"])
    return holding


def map_metadata(db) -> dict:
    return {
        METADATA_KEY: {
            "map_version": get_map_version(db),
            "holding": get_holding(db),
        }
    }


def seed_db(uri: str, auth: tuple, dsg_path: str) -> int:
    """Replace the database contents with the scene graph at ``dsg_path``.

    Wipes everything first, including ``HOLDS`` edges, robot poses and the
    version counter, so it is only for the start of a run. The version is then
    bumped once, so a fresh map is never confused with "no map yet" (0).
    """
    load_dsg_to_db(uri, auth, spark_dsg.DynamicSceneGraph.load(dsg_path))
    with Neo4jWrapper(uri, auth, atomic_queries=True, print_profiles=False) as db:
        return bump_map_version(db)
