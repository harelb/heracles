"""Reloading the map in Neo4j without losing the state that is not map.

Neo4j holds more than the scene graph's layers: a ``_MapState`` node (the map
version planners wait on, the file the map came from, a "rebuilding" flag) and
the ``Robot`` nodes with what each robot ``HOLDS``. A reload that wipes the
database used to take all of that with it: the version went back to 1, so an
old plan looked current; nobody held anything any more; and the publisher sent
an empty, then half-loaded, map while the load ran.

Wrap any full load in ``reloading_map``::

    with reloading_map(db, "console reset") as reload:
        <wipe the map>           # e.g. initialize_db(db, keep_labels=KEEP_LABELS)
        reload.wiped()           # right after the wipe
        <load the new map>

While it runs the map is marked rebuilding, so the publisher keeps the last
map. The mark is refreshed from a thread, since a large map takes longer to
load than ``REBUILD_STALE_S`` (the limit after which a mark is taken as left by
a dead writer). On success the robots and what they hold come back, the
version becomes max(old)+1 and, for a full load, the map gets a new epoch: an
id that changes on every load, so a planner can tell a reloaded map from the
one it planned on even if the versions match. If the load fails after the
wipe, the map is marked as failed and stays held -- a half-built map is not
published -- until a later load or rebuild succeeds.

Holding is kept only within a session. A reload in the middle of one (the
console's map load, a snapshot restore, a rebuild from the change log) keeps
what each robot holds: the robots still have it in hand. The seed at the start
of a new session (``keep_holding=False``; heracles_ros ``seed_db``, which the
publisher runs when ADT4_NEO4J_SEED=true) must not: a pick from an earlier run
would otherwise make the planner believe the robot still holds that object, and
plan to put it down first. That seed drops every HOLDS and logs each one it
dropped. Robots and their poses are kept either way (the state updater rewrites
the poses every second); so are the version count and the "rebuilding" mark.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from contextlib import contextmanager

MAP_STATE = "_MapState"
# What a full load keeps through its wipe (initialize_db's keep_labels).
KEEP_LABELS = (MAP_STATE, "Robot")

# A rebuild marked longer ago than this is taken as abandoned (its writer died
# mid-rebuild): the map is published again rather than held for ever.
REBUILD_STALE_S = 120.0
# How often a running reload renews its mark, well inside REBUILD_STALE_S.
REBUILD_REFRESH_S = 10.0


def set_rebuilding(db, rebuilding: bool) -> None:
    """Mark the map as being rebuilt, or done (which also clears a failure)."""
    if rebuilding:
        db.execute(f"MERGE (m:{MAP_STATE}) SET m.rebuilding_since = $t", t=time.time())
    else:
        db.execute(
            f"MERGE (m:{MAP_STATE}) SET m.rebuilding_since = null "
            "REMOVE m.rebuild_failed"
        )


def mark_rebuild_failed(db, why: str) -> None:
    """Hold the map: a load failed half way, so what Neo4j has is not a map."""
    db.execute(
        f"MERGE (m:{MAP_STATE}) SET m.rebuild_failed = $why, m.rebuilding_since = null",
        why=why or "unknown error",
    )


def rebuild_state(db):
    """(a rebuild is running now, why the last one failed or "")."""
    rows = db.query(
        f"MATCH (m:{MAP_STATE}) RETURN m.rebuilding_since AS t, m.rebuild_failed AS f"
    )
    since = rows[0]["t"] if rows else None
    failed = (rows[0]["f"] or "") if rows else ""
    running = since is not None and time.time() - float(since) < REBUILD_STALE_S
    return running, failed


def is_rebuilding(db) -> bool:
    """Whether the map must not be published or edited now."""
    running, failed = rebuild_state(db)
    return running or bool(failed)


def get_map_version(db) -> int:
    rows = db.query(f"MATCH (m:{MAP_STATE}) RETURN m.version AS v")
    return int(rows[0]["v"]) if rows and rows[0]["v"] is not None else 0


BUMP_MAP_VERSION = (
    f"MERGE (m:{MAP_STATE}) "
    "SET m.version = coalesce(m.version, 0) + 1 "
    "SET m.changed_version = m.version, m.changed_layers = $layers, "
    "m.changed_ids = $ids "
    "RETURN m.version AS v"
)
# How many changed node ids a bump records; past that a reader reloads everything.
MAX_CHANGED_IDS = 500


def changed_params(changed_layers=None, changed_ids=None) -> dict:
    """BUMP_MAP_VERSION's parameters (None: what changed is not known)."""
    layers = None if changed_layers is None else [str(x) for x in changed_layers]
    ids = None if changed_ids is None else [str(x) for x in changed_ids]
    if ids is not None and len(ids) > MAX_CHANGED_IDS:
        ids = None
    return {"layers": layers, "ids": ids}


def bump_map_version(db, changed_layers=None, changed_ids=None) -> int:
    """Bump the map version after a change to the map; returns the new version.

    A writer that knows what it changed passes the layers (``"layer:partition"``
    strings) and node symbols, so a reader that holds the map at the previous
    version can re-read only those (``read_map_change``). Without them the
    change is recorded as unknown, i.e. everything.
    """
    records, _, _ = db.execute(
        BUMP_MAP_VERSION, **changed_params(changed_layers, changed_ids)
    )
    return int(records[0]["v"])


def read_map_change(db):
    """``(version, changed_layers, changed_ids)`` of the last version bump.

    The two lists are None when what made this version is not known: the
    writer did not say, or the version was not made by ``bump_map_version``
    (a reload sets it directly). A reader that holds the map at exactly
    ``version - 1`` may re-read only the changed nodes; otherwise it reloads.
    """
    rows = db.query(
        f"MATCH (m:{MAP_STATE}) RETURN m.version AS v, m.changed_version AS cv, "
        "m.changed_layers AS layers, m.changed_ids AS ids"
    )
    if not rows or rows[0]["v"] is None:
        return 0, None, None
    row = rows[0]
    version = int(row["v"])
    if row["cv"] != version:
        return version, None, None
    layers = None if row["layers"] is None else list(row["layers"])
    ids = None if row["ids"] is None else list(row["ids"])
    return version, layers, ids


def get_map_epoch(db) -> str:
    rows = db.query(f"MATCH (m:{MAP_STATE}) RETURN m.epoch AS e")
    return (rows[0]["e"] or "") if rows else ""


def _robots(db):
    """{robot: properties} and {robot: [held object symbols]} before a wipe."""
    rows = db.query("MATCH (r:Robot) RETURN r.name AS n, properties(r) AS p")
    robots = {r["n"]: r["p"] for r in rows if r["n"]}
    rows = db.query(
        "MATCH (r:Robot)-[:HOLDS]->(o:Object) RETURN r.name AS r, o.nodeSymbol AS o"
    )
    holding = {}
    for r in rows:
        holding.setdefault(r["r"], []).append(r["o"])
    return robots, holding


class MapReload:
    """What reloading_map saved before the wipe; call wiped() right after it."""

    def __init__(self, db, what, log, full, keep_holding=True):
        self.db = db
        self.what = what
        self.log = log
        self.full = full
        self.keep_holding = keep_holding
        self.version = get_map_version(db)
        self.robots, self.holding = _robots(db)
        self.is_wiped = False
        # "robot:object" for every HOLDS a new session's seed dropped.
        self.dropped = []

    def wiped(self) -> None:
        """Mark the map again at once: a full wipe deleted the mark with the map."""
        self.is_wiped = True
        set_rebuilding(self.db, True)

    def finish(self) -> list:
        """Put robots back, holding too unless keep_holding is False, version
        max(old)+1, new epoch if full.

        Returns the held objects the new map no longer has (none are kept when
        keep_holding is False: those are in ``dropped``).
        """
        db = self.db
        for name, props in self.robots.items():
            # Only robots the wipe took: a pose written meanwhile is newer.
            db.execute(
                "MERGE (r:Robot {name: $n}) ON CREATE SET r += $p", n=name, p=props
            )
        lost = []
        if not self.keep_holding:
            # A wipe takes HOLDS with the objects; a load that kept them (or a
            # writer meanwhile) must not leave one either.
            rows = db.execute(
                "MATCH (r:Robot)-[h:HOLDS]->(o) "
                "WITH r.name AS r, o.nodeSymbol AS o, h DELETE h RETURN r, o"
            )[0]
            found = {(robot, o) for robot, held in self.holding.items() for o in held}
            found |= {(row["r"], row["o"]) for row in rows}
            self.dropped = [f"{r}:{o}" for r, o in sorted(found)]
        for robot, held in (self.holding if self.keep_holding else {}).items():
            rows = db.execute(
                "MATCH (r:Robot {name: $r}), (o:Object) WHERE o.nodeSymbol IN $o "
                "MERGE (r)-[:HOLDS]->(o) RETURN o.nodeSymbol AS o",
                r=robot,
                o=held,
            )[0]
            kept = {row["o"] for row in rows}
            lost += [f"{robot}:{o}" for o in held if o not in kept]
        db.execute(
            f"MERGE (m:{MAP_STATE}) SET m.version = "
            "CASE WHEN coalesce(m.version, 0) > $old THEN m.version ELSE $old END + 1",
            old=self.version,
        )
        if self.full:
            db.execute(f"MERGE (m:{MAP_STATE}) SET m.epoch = $e", e=uuid.uuid4().hex)
        return lost


@contextmanager
def reloading_map(db, what="map reload", log=print, full=True, keep_holding=True):
    """Hold the published map while the body replaces it; see the module doc.

    full: the whole database is reloaded (a new map epoch). change_log's
    rebuild of the same map passes False.
    keep_holding: what each robot holds survives the reload (a reload within a
    session). The seed of a new session passes False: every HOLDS is dropped,
    and each one is logged.
    """
    reload = MapReload(db, what, log, full, keep_holding)
    set_rebuilding(db, True)
    stop = threading.Event()

    def refresh():
        while not stop.wait(REBUILD_REFRESH_S):
            try:
                set_rebuilding(db, True)
            except Exception as ex:  # the load itself will say what is wrong
                log(f"{what}: could not renew the rebuilding mark: {ex}")

    # The driver's execute_query is thread safe; a session is not.
    refresher = None
    if getattr(db, "atomic_queries", True):
        refresher = threading.Thread(target=refresh, daemon=True)
        refresher.start()
    try:
        yield reload
        stop.set()
        if refresher is not None:
            refresher.join()
        lost = reload.finish()
        if lost:
            log(
                f"{what}: held object(s) not in the new map, no longer held: "
                + ", ".join(lost)
            )
        for held in reload.dropped:
            log(
                f"{what}: new session: dropped stale holding {held} from a previous "
                "run (a fresh seed starts with every robot's hands empty)"
            )
        set_rebuilding(db, False)
    except BaseException as ex:
        stop.set()
        if refresher is not None:
            refresher.join()
        if reload.is_wiped:
            if not reload.keep_holding:
                # The wipe took them; a retry starts with no HOLDS to report.
                for robot, held in sorted(reload.holding.items()):
                    for o in held:
                        log(
                            f"{what}: new session: dropped stale holding "
                            f"{robot}:{o} from a previous run"
                        )
            why = f"{what} failed after the map was wiped: {ex!r}"
            log(
                f"ERROR: {why}. The map is held (not published) until a reload or "
                "reset succeeds.\n" + traceback.format_exc()
            )
            mark_rebuild_failed(db, why)
        else:
            set_rebuilding(db, False)
        raise
