"""The map as the prior scene graph plus a log of changes, so changes can be undone.

Neo4j holds the map every planner and agent reads. Changes used to be written
into it directly, and a deleted node was gone until the next run. Now every
change is also recorded, in order, as a ``_SceneChange`` node:

* ``seq``     order of the change
* ``source``  who made it: "awcd/<robot>", "sget/chat", "executor/<robot>",
              "kept/awcd/<robot>" (change detection's, kept: keep_detections)
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

While rebuilding, the map is flagged (heracles.map_reload.reloading_map) so
the publisher keeps the last map instead of publishing a half-built one.
"""

from __future__ import annotations

import json
import time
from typing import Iterable, List, Optional

from heracles_ros.map_state import _Transaction, reloading_map
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
_MAP_LABELS = ("Object", "MeshPlace", "Place", "Room", "Building", "Retired")

OBJECT_OPS = {ADDED: "add_object", REMOVED: "remove_object", MOVED: "move_object"}


def ensure_schema(db) -> None:
    """Make seq unique, so two writers logging at once cannot both take max+1.

    The constraint outlives wipes of the data (it is schema). Creating it fails
    if the log already has repeated seqs; record then still works, unguarded.
    """
    if getattr(db, "_scene_change_schema", False) or isinstance(db, _Transaction):
        # (in a transaction that writes, Neo4j refuses schema changes and
        # fails the transaction: map_edit makes the schema before it)
        return
    try:
        db.execute(
            f"CREATE CONSTRAINT scene_change_seq IF NOT EXISTS "
            f"FOR (c:{_LOG}) REQUIRE c.seq IS UNIQUE"
        )
    except Exception as ex:
        print(f"change_log: seq uniqueness constraint not created: {ex}")
    try:
        db._scene_change_schema = True
    except AttributeError:
        pass


def _is_seq_clash(ex) -> bool:
    return type(ex).__name__ == "ConstraintError" or "already exists" in str(ex)


def record(db, source: str, op: str, data: dict, key: str = "") -> int:
    """Append a change to the log; returns its seq.

    seq is max+1, read and written in one statement; a writer that raced
    another to the same seq hits the uniqueness constraint and tries again.
    """
    ensure_schema(db)
    for attempt in range(5):
        try:
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
        except Exception as ex:
            if attempt == 4 or not _is_seq_clash(ex):
                raise
            time.sleep(0.01 * (attempt + 1))


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


def relabel(db, source: str, new_source: str) -> int:
    """Move the active entries of ``source`` to ``new_source``; how many moved."""
    rows = db.execute(
        f"MATCH (c:{_LOG}) WHERE c.active AND c.source = $source "
        "SET c.source = $new RETURN count(c) AS n",
        source=source,
        new=new_source,
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


def _map_symbols(db) -> set:
    """Every map node's symbol (lower case: place edits are logged that way)."""
    rows = db.execute(
        "MATCH (n) WHERE "
        + " OR ".join(f"n:{label}" for label in _MAP_LABELS)
        + " RETURN n.nodeSymbol AS s"
    )[0]
    return {str(r["s"]).lower() for r in rows if r["s"] is not None}


def _outside_links(db) -> int:
    """Edges from map nodes to nodes the rebuild keeps (Observation, ...)."""
    is_map = " OR ".join(f"n:{label}" for label in _MAP_LABELS)
    rows = db.execute(
        f"MATCH (n)-[e]-(m) WHERE ({is_map}) AND NOT ({is_map.replace('n:', 'm:')}) "
        "AND NOT m:Robot RETURN count(e) AS k"
    )[0]
    return int(rows[0]["k"]) if rows else 0


def _image_store(db):
    """The image store the map was loaded with, if the loader recorded one.

    heracles.graph_interface.spark_dsg_to_db records its store directories as
    ``image_store_paths`` (a list, one per robot); ``path`` / ``file_path`` is
    what older loaders wrote. Returns the list or path, or "" for none.
    """
    rows = db.execute(
        "MATCH (m:_GraphMetadata {key: 'image_store'}) "
        "RETURN coalesce(m.image_store_paths, m.path, m.file_path) AS p"
    )[0]
    return (rows[0]["p"] or "") if rows else ""


def _logged_symbols(log_entries) -> set:
    out = set()
    for c in log_entries:
        for k in ("symbol", "a", "b"):
            if c["data"].get(k):
                out.add(str(c["data"][k]).lower())
    return out


def rebuild(db, prior, log=print) -> List[dict]:
    """Make the map the prior scene graph plus the active changes.

    prior is the prior scene graph's path, or a loaded spark_dsg graph (copied,
    not changed). Returns the object changes the rebuild made, as applied
    records, for announcing; the map version is bumped once at the end.

    Only what is in the prior or the change log comes back. Map nodes written
    some other way (not through record) are dropped, and edges from map nodes
    to kept nodes (an object's Observations) are lost: both are counted and
    logged. A change that fails to replay is skipped and logged. If the prior
    cannot be loaded after the map was deleted, the map is marked failed and
    held (heracles.map_reload) rather than published half built.
    """
    import inspect

    from heracles.graph_interface import spark_dsg_to_db

    graph = _load_prior(prior)
    before = objects(db)
    before_symbols = _map_symbols(db)
    links_before = _outside_links(db)
    load_kwargs = {}
    image_store = _image_store(db)
    if image_store:
        if "image_store" in inspect.signature(spark_dsg_to_db).parameters:
            load_kwargs["image_store"] = image_store
        else:
            log(
                f"WARNING: the map was loaded with image store {image_store}, but "
                "this heracles cannot load one: object images are not relinked"
            )
    with reloading_map(db, "rebuild from the prior scene graph", log, full=False) as r:
        db.execute(
            "MATCH (n) WHERE "
            + " OR ".join(f"n:{label}" for label in _MAP_LABELS)
            + " DETACH DELETE n"
        )
        r.wiped()
        spark_dsg_to_db(graph, db, **load_kwargs)
        base_symbols = _map_symbols(db)
        replayed = skipped = 0
        failed = []
        for c in entries(db):
            try:
                done = apply_op(db, c["op"], c["data"])
            except Exception as ex:
                failed.append(f"#{c['seq']} {c['op']} {c['data']}: {ex}")
                continue
            if done is None:
                skipped += 1
            else:
                replayed += 1
        dropped = sorted(
            before_symbols
            - _map_symbols(db)
            - base_symbols
            - _logged_symbols(entries(db, active_only=False))
        )
        links_lost = max(0, links_before - _outside_links(db))
        log(
            f"rebuilt the map from the prior scene graph: {replayed} change(s) replayed"
            + (f", {skipped} no longer applicable" if skipped else "")
        )
        if failed:
            log(f"WARNING: {len(failed)} change(s) failed to replay and were skipped:")
            for f in failed:
                log(f"  {f}")
        if dropped:
            log(
                f"WARNING: the rebuild dropped {len(dropped)} map node(s) that were "
                "neither in the prior nor in the change log (written around it): "
                + ", ".join(dropped)
            )
        if links_lost:
            log(
                f"WARNING: the rebuild lost {links_lost} link(s) from map nodes to "
                "other nodes (e.g. Object->Observation, so object images)"
            )
    return object_diff(before, objects(db))
