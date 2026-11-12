"""Map-level state kept in Neo4j alongside the scene graph.

Neo4j is the single map at runtime, so anything a planner must know about the
world -- not just the scene graph's layers -- has to live there and travel with
the published graph. Two things do:

* ``map_version``: a counter every map writer bumps after a change. A planner
  that has to act on a change can wait until the graph it holds is at least
  that new, instead of replanning on the graph from before the change.
* ``holding``: which robot holds which object. The ``HOLDS`` edges never leave
  the database otherwise, since the published graph carries only the spatial
  layers. Holding lasts a session: a reload keeps it, the seed that starts a
  new session (seed_db) drops it.

Both are written into the graph's metadata under ``METADATA_KEY``. Robot pose
updates do not bump the version: they are not changes to the map.

A third, ``map_epoch`` (also in the metadata), is a new id written on every full load (seed_db, or
heracles.utils.load_dsg_to_db, which wipes the database). The version keeps
counting up through a load, but a planner should compare (epoch, version):
the same version on a different epoch is a different map.

While the change log rebuilds the map (change_log.rebuild) or a load replaces
it, the map nodes are briefly gone or half written. ``rebuilding`` marks that
time, so the publisher holds the last map instead of publishing a partial one,
and editors wait. Any other loader should wrap its load in ``reloading_map``
(heracles.map_reload), which also keeps robots, holding and the version.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import spark_dsg
from heracles.graph_interface import store_mesh_path
from heracles.map_reload import (  # noqa: F401  re-exported for map writers
    REBUILD_STALE_S,
    bump_map_version,
    get_map_epoch,
    get_map_version,
    is_rebuilding,
    mark_rebuild_failed,
    read_map_change,
    rebuild_state,
    reloading_map,
    set_rebuilding,
)
from heracles.query_interface import Neo4jWrapper
from heracles.utils import load_dsg_to_db

METADATA_KEY = "heracles"
_MAP_STATE = "_MapState"
# The console's own change counter, where it runs (see notify_map_change).
_CONSOLE_STATE = "_State"


def notify_map_change(db, changed_layers=None, changed_ids=None) -> int:
    """Bump the map version, and the console's counter if there is one.

    For every edit of the map: the publisher and planners watch the version,
    the console's viewers poll its ``_State`` node. That node is only matched,
    never created, so without a console this is bump_map_version.
    ``changed_layers`` / ``changed_ids``: see bump_map_version.
    """
    version = bump_map_version(db, changed_layers, changed_ids)
    db.execute(
        f"MATCH (s:{_CONSOLE_STATE}) "
        "SET s.version = coalesce(s.version, 0) + 1, s.updated = $t",
        t=time.time(),
    )
    return version


class _Transaction:
    """A transaction with Neo4jWrapper's execute and query, for map writers."""

    def __init__(self, tx):
        self.tx = tx

    def execute(self, query, parameters=None, **kwargs):
        return list(self.tx.run(query, parameters, **kwargs)), None, None

    def query(self, query):
        return [r.data() for r in self.tx.run(query)]


@contextmanager
def map_edit(db):
    """One transaction for an edit, its change log entry and its version bump.

    Yields a db to make them with, or None when the map is being rebuilt or
    reloaded (the edit must not run). The transaction first write-locks
    _MapState, which every rebuild writes when it starts (set_rebuilding):
    an edit and the start of a rebuild therefore never overlap, and two edits
    run one after the other. Nothing is written if the body raises.
    A db without a driver (a test fake) is used as it is.
    """
    driver = getattr(db, "driver", None)
    if driver is None:
        yield None if is_rebuilding(db) else db
        return
    # The change log's schema cannot be made in a transaction that writes
    # (Neo4j refuses it, and the transaction fails): make it first, on its own.
    from heracles_ros.change_log import ensure_schema

    ensure_schema(db)
    with driver.session(database=db.db_name) as session:
        with session.begin_transaction() as tx:
            t = _Transaction(tx)
            t._scene_change_schema = True
            t.execute(f"MERGE (m:{_MAP_STATE}) SET m.edits = coalesce(m.edits, 0) + 1")
            if is_rebuilding(t):
                yield None
                return
            yield t
            tx.commit()


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
            "map_epoch": get_map_epoch(db),
            "holding": get_holding(db),
        }
    }


def find_run_image_store(dsg_path: str) -> Optional[str]:
    """The image store of the run a scene graph belongs to, or ``None``.

    Hydra writes ``<run>/hydra/backend/dsg.json`` next to ``<run>/image_store``; an
    ``image_store`` directly next to the scene graph is accepted too. A directory
    without ``store.sqlite`` is not a store.
    """
    path = os.path.abspath(dsg_path)
    parent = os.path.dirname(path)
    candidates = [os.path.join(parent, "image_store")]
    if os.path.basename(parent) in ("backend", "frontend") and (
        os.path.basename(os.path.dirname(parent)) == "hydra"
    ):
        run = os.path.dirname(os.path.dirname(parent))
        candidates.insert(0, os.path.join(run, "image_store"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "store.sqlite")):
            return c
    return None


def find_run_mesh(dsg_path: str) -> Optional[str]:
    """The mesh Hydra saved next to ``dsg_path`` (``mesh.ply``), or ``None``."""
    path = os.path.join(os.path.dirname(os.path.abspath(dsg_path)), "mesh.ply")
    return path if os.path.isfile(path) else None


def seed_db(
    uri: str,
    auth: tuple,
    dsg_path: str,
    image_store="auto",
    new_session: bool = True,
    log=print,
) -> int:
    """Replace the map in the database with the scene graph at ``dsg_path``.

    load_dsg_to_db wipes the map and the change log, but keeps what is not the
    map (heracles.map_reload): the robots, and the version, which goes on to
    max(old)+1, so a fresh map is never confused with "no map yet" (0) nor with
    the map before it. The map gets a new epoch and records dsg_path as the
    original rebuilds start from. Returns the new version.

    ``new_session`` (default): this seed starts a run (the publisher's
    seed_on_start, i.e. ADT4_NEO4J_SEED=true), so what robots held in an earlier
    run is dropped -- each one logged through ``log`` -- and every robot starts
    with empty hands. Pass False to reseed within a session and keep holding
    (if the new map still has the object), as a console map load does.

    ``image_store``: ``"auto"`` (default) loads the run's ``<run>/image_store``
    when there is one (:func:`find_run_image_store`), so objects get their
    Observations and agents their keyframe images, and
    ``_GraphMetadata {key: 'image_store'}`` tells the console / SAM3 where the
    store is; a path (or list) loads that store; ``None`` loads no images.
    The run's ``mesh.ply`` next to ``dsg_path`` is recorded for viewers.
    """
    if image_store == "auto":
        image_store = find_run_image_store(dsg_path)
    load_dsg_to_db(
        uri,
        auth,
        spark_dsg.DynamicSceneGraph.load(dsg_path),
        source_file_path=dsg_path,
        image_store=image_store,
        keep_holding=not new_session,
        log=log,
    )
    with Neo4jWrapper(uri, auth, atomic_queries=True, print_profiles=False) as db:
        mesh = find_run_mesh(dsg_path)
        if mesh is not None:
            store_mesh_path(db, mesh)
        return get_map_version(db)


def get_prior_path(db) -> str:
    """The scene graph file the map was loaded from ("" if it was not).

    load_dsg_to_db (and so seed_db) records it in _MapState. A map loaded from
    a file some other way, e.g. a loader that wiped _MapState too, may record
    only the _GraphMetadata source, so fall back to that.
    """
    rows = db.query(f"MATCH (m:{_MAP_STATE}) RETURN m.prior_path AS p")
    path = (rows[0]["p"] or "") if rows else ""
    if path:
        return path
    rows = db.query("MATCH (m:_GraphMetadata {key: 'source'}) RETURN m.file_path AS p")
    return (rows[0]["p"] or "") if rows else ""
