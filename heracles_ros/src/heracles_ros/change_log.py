"""The map as the prior scene graph plus a log of changes, so changes can be undone.

Neo4j holds the map every planner and agent reads. Changes used to be written
into it directly, and a deleted node was gone until the next run. Now every
change is also recorded, in order, as a ``_SceneChange`` node:

* ``seq``     order of the change
* ``source``  who made it: "awcd/<robot>", "sget/chat", "executor/<robot>"
* ``op``      add_object, move_object, remove_object, add_place, remove_place,
              add_connection, remove_connection
* ``data``    JSON of the op's arguments, with the ids it was given (an added
              object's symbol), so a replay recreates the same nodes
* ``key``     the reporter's own id for the change (change detection's), used
              to tell which reported changes it later withdrew
* ``active``  false once undone, cleared or withdrawn

The prior scene graph file is the original. ``rebuild`` makes Neo4j that
original plus the active changes again: it replaces the map nodes with the
prior's, replays the active changes in order, and puts back what robots hold.
Robot nodes, the version counter and the log itself are kept. Undoing a change
is then deactivating it and rebuilding, and a reset deactivates them all.

While rebuilding, the map is flagged (map_state.set_rebuilding) so the
publisher keeps the last map instead of publishing a half-built one.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, List, Optional

from heracles_ros.map_state import (
    bump_map_version,
    get_holding,
    set_rebuilding,
)
from heracles_ros.scene_change import (
    ADDED,
    MOVED,
    REMOVED,
    add_connection,
    add_object,
    add_place,
    move_object,
    remove_connection,
    remove_object,
    remove_place,
)

_LOG = "_SceneChange"
_MAP_LABELS = ("Object", "MeshPlace", "Place", "Room", "Building")

OBJECT_OPS = {ADDED: "add_object", REMOVED: "remove_object", MOVED: "move_object"}


def record(db, source: str, op: str, data: dict, key: str = "") -> int:
    """Append a change to the log; returns its seq."""
    rows = db.execute(
        f"""
        OPTIONAL MATCH (c:{_LOG})
        WITH coalesce(max(c.seq), 0) + 1 AS seq
        CREATE (n:{_LOG} {{seq: seq, source: $source, op: $op, data: $data,
                          key: $key, active: true, time: $t}})
        RETURN seq
        """,
        source=source,
        op=op,
        data=json.dumps(data),
        key=key,
        t=time.time(),
    )[0]
    return int(rows[0]["seq"])


def record_applied(
    db, source: str, applied: Iterable[dict], key: str = ""
) -> List[int]:
    """Log object changes as scene_change.apply_changes returns them."""
    seqs = []
    for a in applied:
        data = {"symbol": a["symbol"]}
        if a["kind"] == ADDED:
            data.update(label=a.get("label") or "unknown", xyz=list(a["new"]))
        elif a["kind"] == MOVED:
            data.update(xyz=list(a["new"]))
        seqs.append(record(db, source, OBJECT_OPS[a["kind"]], data, key))
    return seqs


def entries(db, active_only: bool = True, source: Optional[str] = None) -> List[dict]:
    where = []
    if active_only:
        where.append("c.active")
    if source is not None:
        where.append("c.source = $source")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.execute(
        f"MATCH (c:{_LOG}) {clause} RETURN c ORDER BY c.seq", source=source
    )[0]
    out = []
    for r in rows:
        c = dict(r["c"])
        c["data"] = json.loads(c.get("data") or "{}")
        out.append(c)
    return out


def deactivate(db, seqs: Iterable[int]) -> int:
    seqs = [int(s) for s in seqs]
    if not seqs:
        return 0
    rows = db.execute(
        f"MATCH (c:{_LOG}) WHERE c.seq IN $seqs AND c.active "
        "SET c.active = false RETURN count(c) AS n",
        seqs=seqs,
    )[0]
    return int(rows[0]["n"])


def update_data(db, seq: int, data: dict) -> None:
    db.execute(
        f"MATCH (c:{_LOG} {{seq: $seq}}) SET c.data = $data",
        seq=int(seq),
        data=json.dumps(data),
    )


def apply_op(db, op: str, data: dict):
    """Apply one logged change; returns what scene_change returned (None: nothing done)."""
    if op == "add_object":
        return add_object(db, data["symbol"], data.get("label", ""), data["xyz"])
    if op == "move_object":
        return move_object(db, data["symbol"], data["xyz"])
    if op == "remove_object":
        return remove_object(db, data["symbol"])
    if op == "add_place":
        return add_place(db, data["symbol"], data["x"], data["y"], data["neighbours"])
    if op == "remove_place":
        return remove_place(db, data["symbol"])
    if op == "add_connection":
        return add_connection(db, data["a"], data["b"])
    if op == "remove_connection":
        return remove_connection(db, data["a"], data["b"])
    raise ValueError(f"unknown scene change op '{op}'")


def objects(db) -> dict:
    """{symbol: (class, (x, y, z))} of every object in the map."""
    rows = db.execute(
        "MATCH (o:Object) RETURN o.nodeSymbol AS s, o.class AS c, o.center AS p"
    )[0]
    return {
        r["s"]: (
            r["c"] or "",
            None if r["p"] is None else (r["p"].x, r["p"].y, r["p"].z),
        )
        for r in rows
    }


def object_diff(before: dict, after: dict, min_move_m: float = 0.05) -> List[dict]:
    """What happened to the objects between two objects() snapshots, as applied records."""
    out = []
    for sym, (label, old) in before.items():
        if sym not in after:
            out.append(
                {
                    "kind": REMOVED,
                    "symbol": sym,
                    "label": label,
                    "old": old,
                    "new": None,
                }
            )
            continue
        new = after[sym][1]
        if old and new and sum((a - b) ** 2 for a, b in zip(old, new)) > min_move_m**2:
            out.append(
                {"kind": MOVED, "symbol": sym, "label": label, "old": old, "new": new}
            )
    for sym, (label, new) in after.items():
        if sym not in before:
            out.append(
                {"kind": ADDED, "symbol": sym, "label": label, "old": None, "new": new}
            )
    return out


def _load_prior(prior):
    """The prior map as a scene graph ready for spark_dsg_to_db (a fresh copy)."""
    import spark_dsg

    graph = (
        spark_dsg.DynamicSceneGraph.load(prior)
        if isinstance(prior, str)
        else prior.clone()
    )
    # What heracles.utils.load_dsg_to_db sets before loading.
    graph.metadata.add(
        {
            "LayerIdToHeraclesLayerStr": {
                2: "Object",
                5: "Building",
                20: "MeshPlace",
                "3[1]": "MeshPlace",
                3: "Place",
                4: "Room",
            }
        }
    )
    return graph


def rebuild(db, prior, log=print) -> List[dict]:
    """Make the map the prior scene graph plus the active changes.

    prior is the prior scene graph's path, or a loaded spark_dsg graph (copied,
    not changed). Returns the object changes the rebuild made, as applied
    records, for announcing; the map version is bumped once at the end.
    """
    from heracles.graph_interface import spark_dsg_to_db

    graph = _load_prior(prior)
    before = objects(db)
    holding = get_holding(db)
    set_rebuilding(db, True)
    try:
        db.execute(
            "MATCH (n) WHERE "
            + " OR ".join(f"n:{label}" for label in _MAP_LABELS)
            + " DETACH DELETE n"
        )
        spark_dsg_to_db(graph, db)
        replayed = skipped = 0
        for c in entries(db):
            if apply_op(db, c["op"], c["data"]) is None:
                skipped += 1
            else:
                replayed += 1
        for robot, held in holding.items():
            db.execute(
                "MATCH (r:Robot {name: $r}), (o:Object) WHERE o.nodeSymbol IN $o "
                "MERGE (r)-[:HOLDS]->(o)",
                r=robot,
                o=held,
            )
        log(
            f"rebuilt the map from the prior scene graph: {replayed} change(s) replayed"
            + (f", {skipped} no longer applicable" if skipped else "")
        )
    finally:
        set_rebuilding(db, False)
    bump_map_version(db)
    return object_diff(before, objects(db))
