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

While the change log rebuilds the map (change_log.rebuild), the map nodes are
briefly gone or half written. ``rebuilding`` marks that time, so the publisher
holds the last map instead of publishing a partial one, and editors wait.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import spark_dsg
from heracles import graph_interface
from heracles.query_interface import Neo4jWrapper
from heracles.utils import load_dsg_to_db

METADATA_KEY = "heracles"
_MAP_STATE = graph_interface._MAP_STATE


# The counters live in heracles (no ROS dependency) so non-ROS writers (console,
# SAM3 query server) bump exactly the same nodes; prefer notify_map_change, which
# bumps both _MapState.version and the console's _State.version.
get_map_version = graph_interface.get_map_version
bump_map_version = graph_interface.bump_map_version
notify_map_change = graph_interface.notify_map_change


# A rebuild marked longer ago than this is taken as abandoned (its writer died
# mid-rebuild): the map is published again rather than held for ever.
REBUILD_STALE_S = 120.0


def set_rebuilding(db, rebuilding: bool) -> None:
    db.execute(
        f"MERGE (m:{_MAP_STATE}) SET m.rebuilding_since = $t",
        t=time.time() if rebuilding else None,
    )


def is_rebuilding(db) -> bool:
    rows = db.query(f"MATCH (m:{_MAP_STATE}) RETURN m.rebuilding_since AS t")
    since = rows[0]["t"] if rows else None
    return since is not None and time.time() - float(since) < REBUILD_STALE_S


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


def find_run_image_store(dsg_path: str) -> Optional[str]:
    """The image store of the run a scene graph belongs to, or ``None``.

    Hydra writes ``<run>/hydra/backend/dsg.json`` next to ``<run>/image_store``; a
    ``dsg.json`` directly in the run directory is accepted too.
    """
    path = os.path.abspath(dsg_path)
    parent = os.path.dirname(path)
    candidates = [os.path.join(parent, "image_store")]
    if os.path.basename(parent) in ("backend", "frontend") and os.path.basename(
        os.path.dirname(parent)
    ) == "hydra":
        candidates.insert(0, os.path.join(os.path.dirname(os.path.dirname(parent)), "image_store"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "store.sqlite")):
            return c
    return None


def seed_db(uri: str, auth: tuple, dsg_path: str, image_store="auto") -> int:
    """Replace the database contents with the scene graph at ``dsg_path``.

    Wipes everything first, including ``HOLDS`` edges, robot poses and the
    version counter, so it is only for the start of a run. The version is then
    bumped once, so a fresh map is never confused with "no map yet" (0).

    ``image_store``: ``"auto"`` (default) loads the run's ``<run>/image_store``
    when it exists (see :func:`find_run_image_store`), so Agent/Observation image
    paths resolve and ``_GraphMetadata {key:'image_store'}.image_store_paths`` tells
    SAM3 / the console where the keyframes are; a path (or list) uses that store;
    ``None`` loads the graph without images.
    """
    if image_store == "auto":
        image_store = find_run_image_store(dsg_path)
    load_dsg_to_db(
        uri,
        auth,
        spark_dsg.DynamicSceneGraph.load(dsg_path),
        image_store=image_store,
        source_file_path=dsg_path,
    )
    with Neo4jWrapper(uri, auth, atomic_queries=True, print_profiles=False) as db:
        # the graph is stored without its mesh: record the run's mesh file so viewers
        # (e.g. the web console's Neo4j source) can draw it
        mesh = find_run_mesh(dsg_path)
        if mesh is not None:
            graph_interface.store_mesh_path(db, mesh)
        # The original the change log rebuilds from: only a map that was seeded
        # from a file can be rebuilt from it.
        db.execute(
            f"MERGE (m:{_MAP_STATE}) SET m.prior_path = $p",
            p=os.path.abspath(dsg_path),
        )
        return bump_map_version(db)


def find_run_mesh(dsg_path: str) -> Optional[str]:
    """The mesh hydra saved next to ``dsg_path`` (``mesh.ply``), or ``None``."""
    path = os.path.join(os.path.dirname(os.path.abspath(dsg_path)), "mesh.ply")
    return path if os.path.isfile(path) else None


def get_prior_path(db) -> str:
    """The scene graph file the map was seeded from ("" if it was not).

    seed_db records it in _MapState. A map loaded from a file some other way
    (load_dsg_to_db with source_file_path, a console reload) records only the
    _GraphMetadata source, so fall back to that.
    """
    rows = db.query(f"MATCH (m:{_MAP_STATE}) RETURN m.prior_path AS p")
    path = (rows[0]["p"] or "") if rows else ""
    if path:
        return path
    rows = db.query("MATCH (m:_GraphMetadata {key: 'source'}) RETURN m.file_path AS p")
    return (rows[0]["p"] or "") if rows else ""
