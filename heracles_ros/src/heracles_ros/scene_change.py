"""Apply observed scene changes to the Neo4j scene graph.

Change detection reports objects that moved, appeared or disappeared; these
functions write them into Neo4j, the one map every planner reads. Each change
comes back as an "applied" record -- what the map now says, including where
the object was before and its class -- or None when there was nothing to apply.
The class is read before a removal, since afterwards the map no longer has it.

An object a robot is holding is not moved or removed by an observation: the
robot is carrying it, and the holding rule keeps it on the robot. Change
detection does report it removed, since it is gone from where it lay.

Positions are written everywhere the map keeps them: center and bbox_center
(what the published scene graph reads) and pos_x / pos_y / pos_z (what Cypher
queries read). The CONTAINS edge from the nearest mesh place is redone, so
queries for "objects in this place" stay right.

Places (MeshPlace nodes) and the connections between them are edited here
too, for map edits by language (SGET): the change log (change_log.py) replays
every edit through these same functions when it rebuilds the map.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

from heracles import constants

ADDED, REMOVED, MOVED = 0, 1, 2
KIND_NAMES = {ADDED: "ADDED", REMOVED: "REMOVED", MOVED: "MOVED"}
NEW_OBJECT_SIZE = (0.5, 0.5, 0.5)
# admission_policy_version of objects the scene change writer adds
SCENE_CHANGE_POLICY_VERSION = "scene_change_v1"


def object_center(db, symbol) -> Optional[tuple]:
    rows = db.execute(
        "MATCH (o:Object {nodeSymbol: $s}) RETURN o.center AS c", **{"s": symbol}
    )[0]
    if not rows:
        return None
    c = rows[0]["c"]
    return (c.x, c.y, c.z)


def object_class(db, symbol) -> str:
    rows = db.execute(
        "MATCH (o:Object {nodeSymbol: $s}) RETURN o.class AS c", **{"s": symbol}
    )[0]
    return (rows[0]["c"] or "") if rows else ""


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
    """The next object symbol never used in this run.

    Symbols of objects added and later undone count as used: an old goal
    naming one must not silently mean a new object.
    """
    rows = db.execute("MATCH (o:Object) RETURN o.nodeSymbol AS s")[0]
    used = [r["s"] for r in rows]
    logged = db.execute("MATCH (c:_SceneChange {op: 'add_object'}) RETURN c.data AS d")[
        0
    ]
    used += [json.loads(r["d"]).get("symbol", "") for r in logged]
    taken = [int(s[1:]) for s in used if s and s[1:].isdigit()]
    return f"O{max(taken, default=-1) + 1}"


def move_object(db, symbol, xyz) -> Optional[dict]:
    old = object_center(db, symbol)
    if old is None:
        return None
    if holder_of(db, symbol) is not None:
        return None
    _set_position(db, symbol, xyz)
    return {
        "kind": MOVED,
        "symbol": symbol,
        "label": object_class(db, symbol),
        "old": old,
        "new": tuple(xyz),
    }


def remove_object(db, symbol) -> Optional[dict]:
    old = object_center(db, symbol)
    if old is None:
        return None
    if holder_of(db, symbol) is not None:
        return None
    label = object_class(db, symbol)
    db.execute("MATCH (o:Object {nodeSymbol: $s}) DETACH DELETE o", **{"s": symbol})
    return {"kind": REMOVED, "symbol": symbol, "label": label, "old": old, "new": None}


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
            first_observed_ns: [$now], last_observed_ns: [$now],
            admission_status: $admitted, admission_policy_version: $policy,
            admission_reason: 'observed scene change'
        })
        """,
        **{
            "s": symbol,
            "label": label or "unknown",
            "x": x,
            "y": y,
            "z": z,
            "sx": sx,
            "sy": sy,
            "sz": sz,
            "now": now,
            # Readers (scene queries, the console) only see admitted objects; an
            # observed change the writer accepted is admitted by that decision.
            "admitted": constants.ADMITTED,
            "policy": SCENE_CHANGE_POLICY_VERSION,
        },
    )
    _relink_to_nearest_place(db, symbol)
    return {
        "kind": ADDED,
        "symbol": symbol,
        "label": label or "unknown",
        "old": None,
        "new": (x, y, z),
    }


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


def place_exists(db, symbol) -> bool:
    rows = db.execute(
        "MATCH (p:MeshPlace {nodeSymbol: $s}) RETURN p.nodeSymbol AS s", **{"s": symbol}
    )[0]
    return bool(rows)


def next_place_symbol(db) -> str:
    rows = db.execute(
        "MATCH (p:MeshPlace) RETURN max(toInteger(substring(p.nodeSymbol, 1))) AS n"
    )[0]
    top = rows[0]["n"] if rows else None
    return f"t{(top or 0) + 1}"


def add_place(db, symbol, x, y, neighbours) -> Optional[dict]:
    """A traversable place at (x, y), connected both ways to existing places.

    It copies its nearest place's attributes (type, clearance, ...) except for
    where it is, so the scene graph can be built from it. None if the symbol
    is taken or a neighbour does not exist.
    """
    neighbours = [n.strip().lower() for n in neighbours if n.strip()]
    symbol = (symbol or "").strip().lower() or next_place_symbol(db)
    if place_exists(db, symbol) or not neighbours:
        return None
    known = db.execute(
        "MATCH (p:MeshPlace) WHERE p.nodeSymbol IN $n RETURN p.nodeSymbol AS s",
        **{"n": neighbours},
    )[0]
    if len({r["s"] for r in known}) != len(set(neighbours)):
        return None
    x, y = float(x), float(y)
    db.execute(
        """
        MATCH (src:MeshPlace)
        WITH src, (src.pos_x - $x)^2 + (src.pos_y - $y)^2 AS d ORDER BY d LIMIT 1
        CREATE (p:MeshPlace) SET p = properties(src),
            p.nodeSymbol = $s, p.center = point({x: $x, y: $y, z: 0.0}),
            p.pos_x = $x, p.pos_y = $y, p.pos_z = 0.0
        """,
        **{"s": symbol, "x": x, "y": y},
    )
    db.execute(
        """
        MATCH (p:MeshPlace {nodeSymbol: $s}), (q:MeshPlace) WHERE q.nodeSymbol IN $n
        CREATE (p)-[:MESH_PLACE_CONNECTED]->(q), (q)-[:MESH_PLACE_CONNECTED]->(p)
        """,
        **{"s": symbol, "n": neighbours},
    )
    return {"symbol": symbol, "x": x, "y": y, "neighbours": neighbours}


def remove_place(db, symbol) -> Optional[dict]:
    """Delete a place with all its edges. None if there is no such place.

    Objects it contained move to their nearest remaining place, so they stay
    in the map.
    """
    symbol = (symbol or "").strip().lower()
    if not place_exists(db, symbol):
        return None
    rows = db.execute(
        """
        MATCH (p:MeshPlace {nodeSymbol: $s})
        OPTIONAL MATCH (p)-[r]-()
        RETURN count(r) AS edges
        """,
        **{"s": symbol},
    )[0]
    contained = db.execute(
        "MATCH (:MeshPlace {nodeSymbol: $s})-[:CONTAINS]->(o:Object) RETURN o.nodeSymbol AS o",
        **{"s": symbol},
    )[0]
    db.execute("MATCH (p:MeshPlace {nodeSymbol: $s}) DETACH DELETE p", **{"s": symbol})
    for r in contained:
        _relink_to_nearest_place(db, r["o"])
    return {
        "symbol": symbol,
        "edges": int(rows[0]["edges"]),
        "objects": [r["o"] for r in contained],
    }


def _connections(db, a, b) -> int:
    rows = db.execute(
        """
        MATCH (p:MeshPlace {nodeSymbol: $a})-[r:MESH_PLACE_CONNECTED]-(q:MeshPlace {nodeSymbol: $b})
        RETURN count(r) AS n
        """,
        **{"a": a, "b": b},
    )[0]
    return int(rows[0]["n"]) if rows else 0


def remove_connection(db, a, b) -> Optional[dict]:
    """Delete the connection between two places, both directions."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not _connections(db, a, b):
        return None
    db.execute(
        """
        MATCH (p:MeshPlace {nodeSymbol: $a})-[r:MESH_PLACE_CONNECTED]-(q:MeshPlace {nodeSymbol: $b})
        DELETE r
        """,
        **{"a": a, "b": b},
    )
    return {"a": a, "b": b}


def add_connection(db, a, b) -> Optional[dict]:
    """Connect two existing places both ways (no-op if already connected)."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if a == b or not (place_exists(db, a) and place_exists(db, b)):
        return None
    if _connections(db, a, b):
        return None
    db.execute(
        """
        MATCH (p:MeshPlace {nodeSymbol: $a}), (q:MeshPlace {nodeSymbol: $b})
        CREATE (p)-[:MESH_PLACE_CONNECTED]->(q), (q)-[:MESH_PLACE_CONNECTED]->(p)
        """,
        **{"a": a, "b": b},
    )
    return {"a": a, "b": b}
