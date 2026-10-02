"""Apply observed scene changes to the Neo4j scene graph.

Change detection reports objects that moved, appeared or disappeared; these
functions write them into Neo4j, the one map every planner reads. Each change
comes back as an "applied" record -- what the map now says, including where
the object was before -- or None when there was nothing to apply.

An object a robot is holding is not moved by an observation: the robot is
carrying it, and the holding rule keeps it on the robot.

Positions are written everywhere the map keeps them: center and bbox_center
(what the published scene graph reads) and pos_x / pos_y / pos_z (what Cypher
queries read). The CONTAINS edge from the nearest mesh place is redone, so
queries for "objects in this place" stay right.
"""

from __future__ import annotations

import time
from typing import List, Optional

ADDED, REMOVED, MOVED = 0, 1, 2
KIND_NAMES = {ADDED: "ADDED", REMOVED: "REMOVED", MOVED: "MOVED"}
NEW_OBJECT_SIZE = (0.5, 0.5, 0.5)


def object_center(db, symbol) -> Optional[tuple]:
    rows = db.execute(
        "MATCH (o:Object {nodeSymbol: $s}) RETURN o.center AS c", **{"s": symbol}
    )[0]
    if not rows:
        return None
    c = rows[0]["c"]
    return (c.x, c.y, c.z)


def holder_of(db, symbol) -> Optional[str]:
    rows = db.execute(
        "MATCH (r:Robot)-[:HOLDS]->(o:Object {nodeSymbol: $s}) RETURN r.name AS r",
        **{"s": symbol},
    )[0]
    return rows[0]["r"] if rows else None


def _relink_to_nearest_place(db, symbol):
    db.execute(
        """
        MATCH (o:Object {nodeSymbol: $s})
        OPTIONAL MATCH (:MeshPlace)-[old:CONTAINS]->(o)
        DELETE old
        WITH o
        MATCH (p:MeshPlace)
        WITH o, p, point.distance(
            point({x: p.center.x, y: p.center.y}), point({x: o.center.x, y: o.center.y})
        ) AS d
        ORDER BY d LIMIT 1
        MERGE (p)-[:CONTAINS]->(o)
        """,
        **{"s": symbol},
    )


def _set_position(db, symbol, xyz):
    x, y, z = (float(v) for v in xyz)
    db.execute(
        """
        MATCH (o:Object {nodeSymbol: $s})
        SET o.center = point({x: $x, y: $y, z: $z}),
            o.bbox_center = point({x: $x, y: $y, z: $z}),
            o.pos_x = $x, o.pos_y = $y, o.pos_z = $z
        """,
        **{"s": symbol, "x": x, "y": y, "z": z},
    )
    _relink_to_nearest_place(db, symbol)


def _next_object_symbol(db) -> str:
    rows = db.execute("MATCH (o:Object) RETURN o.nodeSymbol AS s")[0]
    taken = [int(r["s"][1:]) for r in rows if r["s"] and r["s"][1:].isdigit()]
    return f"O{max(taken, default=-1) + 1}"


def move_object(db, symbol, xyz) -> Optional[dict]:
    old = object_center(db, symbol)
    if old is None:
        return None
    if holder_of(db, symbol) is not None:
        return None
    _set_position(db, symbol, xyz)
    return {"kind": MOVED, "symbol": symbol, "old": old, "new": tuple(xyz)}


def remove_object(db, symbol) -> Optional[dict]:
    old = object_center(db, symbol)
    if old is None:
        return None
    db.execute(
        "MATCH (o:Object {nodeSymbol: $s}) DETACH DELETE o", **{"s": symbol}
    )
    return {"kind": REMOVED, "symbol": symbol, "old": old, "new": None}


def add_object(db, symbol, label, xyz) -> Optional[dict]:
    if symbol and object_center(db, symbol) is not None:
        # Already known: an observation of where it is now.
        return move_object(db, symbol, xyz)
    symbol = symbol or _next_object_symbol(db)
    x, y, z = (float(v) for v in xyz)
    now = time.time_ns()
    sx, sy, sz = NEW_OBJECT_SIZE
    db.execute(
        """
        CREATE (o:Object {
            nodeSymbol: $s, class: $label, attr_type: 'ObjectNodeAttributes',
            center: point({x: $x, y: $y, z: $z}),
            bbox_center: point({x: $x, y: $y, z: $z}),
            bbox_dim: point({x: $sx, y: $sy, z: $sz}),
            pos_x: $x, pos_y: $y, pos_z: $z,
            color_r: 0, color_g: 0, color_b: 0,
            is_active: true, is_predicted: false, registered: false,
            first_observed_ns: [$now], last_observed_ns: [$now]
        })
        """,
        **{"s": symbol, "label": label or "unknown", "x": x, "y": y, "z": z,
                     "sx": sx, "sy": sy, "sz": sz, "now": now},
    )
    _relink_to_nearest_place(db, symbol)
    return {"kind": ADDED, "symbol": symbol, "old": None, "new": (x, y, z)}


def apply_changes(db, changes) -> List[dict]:
    """Apply [(kind, symbol, label, (x, y, z))]; returns the applied ones."""
    applied = []
    for kind, symbol, label, xyz in changes:
        if kind == MOVED:
            done = move_object(db, symbol, xyz)
        elif kind == REMOVED:
            done = remove_object(db, symbol)
        elif kind == ADDED:
            done = add_object(db, symbol, label, xyz)
        else:
            done = None
        if done is not None:
            applied.append(done)
    return applied
