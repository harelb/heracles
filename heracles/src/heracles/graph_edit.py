"""Interactive scene-graph editing on Neo4j: single nodes, edges, groups, undo, snapshots.

Ported from SGET's ``backend/neo4j_crud.py`` (arkinj/scene_graph_editing_tool), which
filled heracles' gap: heracles bulk-loads whole graphs (UNWIND + MERGE) but had no way
to change one node.  What changed on the way in:

* **Every edit is a unit of work.**  :func:`apply_edit` runs a list of *ops* (the JSON
  schema below) inside one :class:`~heracles.transactions.GraphUnitOfWork`: one Neo4j
  transaction with the optimistic revision check, a ``GraphMutation`` audit node (who,
  from where, a human-readable summary) and both map-version counters bumped *in the same
  transaction* (``_State{key:'global'}.version`` for viewers, ``_MapState.version`` for the
  heracles DSG publisher), so a robot never sees half an edit.
* **Undo.**  Each op returns its inverse; the inverse list is stored on the
  ``GraphMutation`` (``inverse_json``) and :func:`undo_edit` replays it as a new unit of
  work (``undo_of`` / ``undone_by`` link the two).
* **Retire instead of delete.**  ``retire_node`` swaps the layer label for ``:Retired``
  (keeping properties and edges), so readers that match by layer label -- ``db_to_spark_dsg``,
  the planners, Cypher like ``MATCH (o:Object)`` -- stop seeing it, and undo just swaps it
  back.  ``delete_node`` (DETACH DELETE) is still there; its inverse re-creates the node
  with every property and edge.
* **Symbols are allocated server-side** (next free index of the layer's prefix over the
  whole DB: spark_dsg node ids are global, so a prefix must never collide across labels).
* **A new class is appended to the ``_Labelspace``** of its layer (object / room), so the
  node gets a ``semantic_label`` and survives ``db_to_spark_dsg``.
* **group** creates a parent one layer up (operator regions are ``Room`` nodes, layer 4)
  at the members' centroid, with an optional ``boundary`` polygon, and ``CONTAINS`` edges to
  the members -- given explicitly or selected by point-in-polygon on their centers.
* Property names that reach Cypher text are whitelisted (``EDITABLE_PROPERTIES``) or
  validated as identifiers; labels and relationship types come from fixed tables.

Op schema (plain JSON; the console's ``/api/edit/.../batch`` and its chat tool use it)::

    {"op": "create_node", "layer": "Object", "pos": [x, y, z?], "class": "crate",
     "name"?: str, "symbol"?: "O900", "size"?: 1.0, "bbox"?: {"center": [..], "dim": [..]},
     "boundary"?: [[x, y], ...], "props"?: {...}}
    {"op": "update_node", "symbol": "O41", "set": {"class": "crate", "center": [x, y, z]},
     "unset"?: [keys], "expect"?: {key: value}}
    {"op": "retire_node", "symbol": "O41", "reason"?: str}
    {"op": "restore_node", "symbol": "O41"}
    {"op": "delete_node", "symbol": "O41"}               # hard DETACH DELETE
    {"op": "create_edge", "from": "R3", "to": "O41", "type"?: "CONTAINS"}
    {"op": "delete_edge", "from": "R3", "to": "O41", "type"?: "CONTAINS"}
    {"op": "group", "parent_layer": "Room", "class": "region", "name": "courtyard",
     "members"?: [symbols] | "polygon"?: [[x, y], ...], "member_layers"?: [labels]}
    {"op": "add_class", "layer": "object" | "room", "name": "crate"}

Internal ops that only appear as inverses: ``recreate_node``, ``set_labelspace``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from . import constants
from .transactions import GraphUnitOfWork

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema tables
# ---------------------------------------------------------------------------

# Intralayer (sibling) relationship type per layer label (heracles add_edges_from_dsg).
INTRALAYER_EDGE_TYPES = {
    constants.OBJECTS: "OBJECT_CONNECTED",
    constants.PLACES: "PLACE_CONNECTED",
    constants.MESH_PLACES: "MESH_PLACE_CONNECTED",
    constants.ROOMS: "ROOM_CONNECTED",
    constants.BUILDINGS: "BUILDING_CONNECTED",
}
# Interlayer edges are parent -> child.
INTERLAYER_EDGE_TYPE = "CONTAINS"
EDGE_TYPES = set(INTRALAYER_EDGE_TYPES.values()) | {INTERLAYER_EDGE_TYPE}

RETIRED = "Retired"  # label a retired node carries instead of its layer label

# label -> (layer, partition, symbol prefix, default attr_type, labelspace key)
LAYER_SPECS: Mapping[str, tuple[int, int, str, str, str | None]] = {
    constants.OBJECTS: (2, 0, "O", "ObjectNodeAttributes", "object"),
    constants.PLACES: (3, 0, "p", "PlaceNodeAttributes", None),
    constants.MESH_PLACES: (3, 1, "P", "Place2dNodeAttributes", "object"),
    constants.ROOMS: (4, 0, "R", "RoomNodeAttributes", "room"),
    constants.BUILDINGS: (5, 0, "B", "NodeAttributes", None),
}
MAP_LABELS = tuple(LAYER_SPECS)

_LAYER_ALIASES = {
    "object": constants.OBJECTS, "objects": constants.OBJECTS,
    "place": constants.PLACES, "places": constants.PLACES,
    "meshplace": constants.MESH_PLACES, "meshplaces": constants.MESH_PLACES,
    "mesh_place": constants.MESH_PLACES, "mesh_places": constants.MESH_PLACES,
    "room": constants.ROOMS, "rooms": constants.ROOMS,
    "region": constants.ROOMS, "regions": constants.ROOMS,
    "building": constants.BUILDINGS, "buildings": constants.BUILDINGS,
}  # fmt: skip

# Valid children of a parent created by ``group`` (SGET's _PARENT_LAYERS, inverted;
# rooms also contain objects directly: operator regions are about what is in them).
GROUP_CHILD_LAYERS = {
    constants.ROOMS: (constants.PLACES, constants.MESH_PLACES, constants.OBJECTS),
    constants.BUILDINGS: (constants.ROOMS,),
    constants.PLACES: (constants.OBJECTS,),
    constants.MESH_PLACES: (constants.OBJECTS,),
}

# Stored as Neo4j Point3D (``point()`` in SET clauses); ``boundary`` is a Point3D list.
POINT3D_PROPERTIES = {"center", "bbox_center", "bbox_dim"}
POINT_LIST_PROPERTIES = {"boundary"}

# Properties update_node may change (SGET's ALLOWED_PROPERTIES minus nodeSymbol, plus
# the admission/provenance fields and operator annotations).  Names reach Cypher text.
EDITABLE_PROPERTIES = frozenset({
    "attr_type", "center", "bbox_center", "bbox_dim", "class", "name",
    "color_r", "color_g", "color_b", "registered", "is_active", "is_predicted",
    "distance", "first_observed_ns", "last_observed_ns", "min_radius", "max_radius",
    "radii", "states", "boundary", "class_prob_keys", "class_prob_values",
    "semantic_label", "pos_x", "pos_y", "pos_z",
    constants.ADMISSION_STATUS, constants.ADMISSION_POLICY_VERSION,
    constants.ADMISSION_REASON, constants.ADMISSION_BELIEF,
    "source", "blocked", "blocked_reason", "note",
})  # fmt: skip
# Kept for SGET callers.
ALLOWED_PROPERTIES = EDITABLE_PROPERTIES

OPERATOR_POLICY = "operator_edit_v0"
MAP_EDIT_KIND = "map_edit"

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SYMBOL = re.compile(r"([A-Za-z])\(?(\d+)\)?\Z")


class EditError(ValueError):
    """An op is invalid or cannot apply (unknown node, bad property, bad edge ...)."""


class NodeNotFound(EditError):
    pass


class EditConflict(EditError):
    """An op's ``expect`` no longer matches the graph (someone changed it since)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def layer_label(layer) -> str:
    """Normalise a layer name / alias / heracles label to the Neo4j label."""
    if isinstance(layer, str):
        if layer in LAYER_SPECS:
            return layer
        lab = _LAYER_ALIASES.get(layer.strip().lower().replace(" ", ""))
        if lab:
            return lab
    raise EditError(f"unknown layer {layer!r} (one of {', '.join(MAP_LABELS)})")


def layer_key(label: str) -> str:
    """``"layer:partition"`` string used for ``_State.dirty_layers``."""
    lay, part, *_ = LAYER_SPECS[label]
    return f"{lay}:{part}"


def determine_edge_type(from_label: str, to_label: str) -> str:
    """Relationship type between two layers: intralayer ``*_CONNECTED`` or ``CONTAINS``."""
    if from_label == to_label:
        t = INTRALAYER_EDGE_TYPES.get(from_label)
        if t is None:
            raise EditError(f"no intralayer edge type for {from_label!r}")
        return t
    if LAYER_SPECS[from_label][0] == LAYER_SPECS[to_label][0]:
        raise EditError(f"no edge type between {from_label} and {to_label} (same layer)")
    return INTERLAYER_EDGE_TYPE


def _is_point(v) -> bool:
    return hasattr(v, "srid") and hasattr(v, "x") and hasattr(v, "y")


def _xyz(v) -> list[float]:
    vals = [float(c) for c in v]
    if len(vals) == 2:
        vals.append(0.0)
    if len(vals) != 3 or not all(math.isfinite(c) for c in vals):
        raise EditError(f"expected [x, y(, z)], got {v!r}")
    return vals


def plain(v):
    """JSON-safe copy of a Neo4j value (points -> [x, y, z], temporals -> iso)."""
    if _is_point(v):
        return [float(c) for c in v]
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if hasattr(v, "iso_format"):
        return v.iso_format()
    return v


def _same(a, b, tol: float = 1e-6) -> bool:
    a, b = plain(a), plain(b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b)) <= tol
    return a == b


def parse_symbol(sym: str) -> tuple[str, int]:
    m = _SYMBOL.match(str(sym).strip())
    if not m:
        raise EditError(f"bad node symbol {sym!r}")
    return m.group(1), int(m.group(2))


def symbol_variants(sym: str) -> list[str]:
    """``"O12"`` and ``"O(12)"``: both spellings exist in heracles databases."""
    c, i = parse_symbol(sym)
    return [f"{c}{i}", f"{c}({i})"]


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Even-odd ray casting; ``poly`` is a list of [x, y] (open or closed ring)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-300) + xi:
            inside = not inside
        j = i
    return inside


def polygon_centroid(poly: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Area centroid (falls back to the vertex mean for a degenerate polygon)."""
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i][0], poly[i][1]
        x1, y1 = poly[(i + 1) % n][0], poly[(i + 1) % n][1]
        cr = x0 * y1 - x1 * y0
        a += cr
        cx += (x0 + x1) * cr
        cy += (y0 + y1) * cr
    if abs(a) < 1e-9:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    return (cx / (3 * a), cy / (3 * a))


def _polygon(poly) -> list[list[float]]:
    try:
        pts = [[float(p[0]), float(p[1])] for p in poly]
    except (TypeError, ValueError, IndexError):
        raise EditError("polygon must be a list of [x, y]") from None
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise EditError("polygon needs at least 3 vertices")
    if not all(math.isfinite(c) for p in pts for c in p):
        raise EditError("polygon has non-finite vertices")
    return pts


# ---------------------------------------------------------------------------
# The op applier (runs inside one transaction)
# ---------------------------------------------------------------------------


# index-backed lookup of a map node by symbol (a label-less MATCH scans every node)
_BY_SYMBOL = (
    "CALL { "
    + " UNION ".join(f"MATCH (n:{lab}) WHERE n.nodeSymbol IN $syms RETURN n" for lab in (*MAP_LABELS, RETIRED))
    + " } "
)


def _first_label(labels) -> str | None:
    """The label to MATCH a node by (map labels first; any indexable identifier)."""
    labels = [str(x) for x in labels or []]
    for lab in (*MAP_LABELS, RETIRED):
        if lab in labels:
            return lab
    return next((lab for lab in labels if _IDENT.match(lab)), None)


class _Applier:
    """Applies ops through ``tx.run`` and collects changes, inverses, dirty layers."""

    def __init__(self, tx, revision: int):
        self.tx = tx
        self.revision = revision  # the revision this edit commits as
        self.changes: list[dict] = []
        self.inverse: list[dict] = []  # in application order; reversed when stored
        self.affected: list[str] = []
        self.dirty: set[str] = set()
        self._attr_type_cache: dict[str, str] = {}

    # -- tx helpers ---------------------------------------------------------
    def rows(self, query: str, **params) -> list[dict]:
        return [r.data() for r in self.tx.run(query, **params)]

    def node(self, sym: str, *, retired: bool | None = False) -> dict:
        """{label, props, retired} of the node named ``sym``; NodeNotFound if absent.

        ``retired``: False = only active map nodes, True = only retired, None = either.
        """
        rows = self.rows(_BY_SYMBOL + "RETURN labels(n) AS labels, properties(n) AS props",
                         syms=symbol_variants(sym))
        for r in rows:
            labs = set(r["labels"])
            is_retired = RETIRED in labs
            label = next((lab for lab in MAP_LABELS if lab in labs), None)
            if is_retired:
                label = r["props"].get("retired_label") or label
            if label is None:
                continue
            if retired is not None and is_retired != retired:
                continue
            return {"label": label, "props": r["props"], "retired": is_retired,
                    "symbol": r["props"]["nodeSymbol"]}  # fmt: skip
        state = {False: "active ", True: "retired ", None: ""}[retired]
        raise NodeNotFound(f"no {state}map node {sym}")

    def touch(self, label: str, sym: str) -> None:
        self.dirty.add(layer_key(label))
        if sym not in self.affected:
            self.affected.append(sym)

    def default_attr_type(self, label: str) -> str:
        if label not in self._attr_type_cache:
            rows = self.rows(
                f"MATCH (n:{label}) WHERE n.attr_type IS NOT NULL "
                "RETURN n.attr_type AS t, count(*) AS c ORDER BY c DESC LIMIT 1"
            )
            self._attr_type_cache[label] = rows[0]["t"] if rows else LAYER_SPECS[label][3]
        return self._attr_type_cache[label]

    def next_symbol(self, label: str) -> str:
        prefix = LAYER_SPECS[label][2]
        rows = self.rows(
            "MATCH (n) WHERE n.nodeSymbol STARTS WITH $p AND n.nodeSymbol =~ $re "
            "RETURN max(toInteger(replace(replace(substring(n.nodeSymbol, 1), '(', ''), ')', ''))) AS m",
            p=prefix,
            re=re.escape(prefix) + r"\(?[0-9]+\)?",
        )
        m = rows[0]["m"] if rows and rows[0]["m"] is not None else -1
        return f"{prefix}{int(m) + 1}"

    # -- labelspace ---------------------------------------------------------
    def labelspace(self, key: str):
        rows = self.rows(
            "MATCH (l:_Labelspace {layer: $k}) RETURN l.ids AS ids, l.names AS names", k=key
        )
        if not rows:
            return None
        return list(rows[0]["ids"] or []), list(rows[0]["names"] or [])

    def ensure_class(self, label: str, name: str) -> int | None:
        """The class's semantic id, appending it to the layer's labelspace if new.

        An object labelspace is only extended, never created (creating one would
        orphan every existing object's semantic label); a missing room labelspace is
        created, with ids above any room's existing semantic label.
        """
        key = LAYER_SPECS[label][4]
        if key is None or not name:
            return None
        ls = self.labelspace(key)
        if ls is not None and name in ls[1]:
            return int(ls[0][ls[1].index(name)])
        if ls is None:
            if key != "room":
                return None
            # no room labelspace yet: 0 = unknown, new ids above any room's label
            top = self.rows(f"MATCH (r:{constants.ROOMS}) RETURN max(r.semantic_label) AS m")
            m = top[0]["m"] if top else None
            ids, names = ([0], ["unknown"]) if m is None else ([], [])
            if name not in names:
                ids, names = ids + [max(int(m or 0), 0) + 1], names + [name]
            self.tx.run(
                "MERGE (l:_Labelspace {layer: $k}) SET l.ids = $ids, l.names = $names",
                k=key, ids=ids, names=names,
            )  # fmt: skip
            self.inverse.append({"op": "set_labelspace", "layer": key, "delete": True})
            self.changes.append({"kind": "class", "layer": key, "name": name})
            return ids[names.index(name)]
        ids, names = ls
        sid = max(ids, default=-1) + 1
        self.tx.run(
            "MATCH (l:_Labelspace {layer: $k}) SET l.ids = $ids, l.names = $names",
            k=key, ids=ids + [sid], names=names + [name],
        )  # fmt: skip
        self.inverse.append({"op": "set_labelspace", "layer": key, "ids": ids, "names": names})
        self.changes.append({"kind": "class", "layer": key, "name": name})
        return sid

    def op_add_class(self, op):
        key = str(op.get("layer", "object")).lower()
        label = {"object": constants.OBJECTS, "room": constants.ROOMS}.get(key)
        if label is None:
            raise EditError("add_class layer is 'object' or 'room'")
        name = str(op.get("name") or "").strip().lower()
        if not name:
            raise EditError("add_class needs a name")
        self.ensure_class(label, name)

    def op_set_labelspace(self, op):
        key = str(op["layer"])
        prev = self.labelspace(key)
        if op.get("delete"):
            self.tx.run("MATCH (l:_Labelspace {layer: $k}) DELETE l", k=key)
        else:
            self.tx.run(
                "MERGE (l:_Labelspace {layer: $k}) SET l.ids = $ids, l.names = $names",
                k=key, ids=list(op["ids"]), names=list(op["names"]),
            )  # fmt: skip
        if prev is None:
            self.inverse.append({"op": "set_labelspace", "layer": key, "delete": True})
        else:
            self.inverse.append({"op": "set_labelspace", "layer": key, "ids": prev[0], "names": prev[1]})
        self.changes.append({"kind": "labelspace", "layer": key})

    # -- nodes ------------------------------------------------------------
    def op_create_node(self, op) -> str:
        label = layer_label(op.get("layer", constants.OBJECTS))
        lay, part, _prefix, _at, ls_key = LAYER_SPECS[label]
        sym = op.get("symbol")
        if sym:
            parse_symbol(sym)
            if self.rows(_BY_SYMBOL + "RETURN count(n) AS c", syms=symbol_variants(sym))[0]["c"]:
                raise EditError(f"node {sym} already exists")
        else:
            sym = self.next_symbol(label)
        if op.get("pos") is None and not op.get("boundary"):
            raise EditError("create_node needs pos (or a boundary)")
        boundary = _polygon(op["boundary"]) if op.get("boundary") else None
        if op.get("pos") is not None:
            pos = _xyz(op["pos"])
        else:
            cx, cy = polygon_centroid(boundary)
            pos = [cx, cy, 0.0]
        cls = str(op.get("class") or "").strip().lower() or None
        if label == constants.OBJECTS and not cls:
            cls = "unknown"
        if label == constants.ROOMS and not cls:
            cls = "region"
        name = op.get("name")
        name = str(name).strip() if name else None
        attr_type = op.get("attr_type") or self.default_attr_type(label)
        props: dict[str, Any] = {
            "nodeSymbol": sym, "layer": lay, "partition": part, "attr_type": attr_type,
            "pos_x": pos[0], "pos_y": pos[1], "pos_z": pos[2],
            "is_active": False, "is_predicted": False,
            "source": str(op.get("source") or "operator"),
            "created_at_revision": self.revision,
        }  # fmt: skip
        if cls:
            props["class"] = cls
        if name or (label == constants.OBJECTS and cls):
            props["name"] = name or cls
        if label == constants.OBJECTS:
            now = time.time_ns()
            props.update({
                "registered": False, "color_r": 255, "color_g": 160, "color_b": 0,
                constants.ADMISSION_STATUS: constants.ADMITTED,
                constants.ADMISSION_POLICY_VERSION: OPERATOR_POLICY,
                constants.ADMISSION_REASON: str(op.get("reason") or "added by the operator"),
            })  # fmt: skip
            if attr_type == "KhronosObjectAttributes":
                props["first_observed_ns"] = [now]
                props["last_observed_ns"] = [now]
        for k, v in dict(op.get("props") or {}).items():
            if k not in EDITABLE_PROPERTIES or k in POINT3D_PROPERTIES | POINT_LIST_PROPERTIES:
                raise EditError(f"property {k!r} cannot be set on create")
            props[k] = v
        if cls and ls_key:
            sid = self.ensure_class(label, cls)
            if sid is not None:
                props["semantic_label"] = sid
        # bounding box: explicit, else the boundary's extent, else a cube of `size`
        bbox_c = bbox_d = None
        if op.get("bbox"):
            bbox_c, bbox_d = _xyz(op["bbox"]["center"]), _xyz(op["bbox"]["dim"])
        elif boundary:
            xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
            bbox_c = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, pos[2]]
            bbox_d = [max(xs) - min(xs), max(ys) - min(ys), float(op.get("height", 1.0))]
        elif label in (constants.OBJECTS, constants.ROOMS):
            s = float(op.get("size", 1.0))
            bbox_c, bbox_d = list(pos), [s, s, s]
        q = f"CREATE (n:{label}) SET n = $props, n.center = point({{x: $x, y: $y, z: $z}})"
        params: dict[str, Any] = {"props": props, "x": pos[0], "y": pos[1], "z": pos[2]}
        if bbox_c is not None:
            q += (", n.bbox_center = point({x: $bx, y: $by, z: $bz}), "
                  "n.bbox_dim = point({x: $bl, y: $bw, z: $bh})")  # fmt: skip
            params.update(bx=bbox_c[0], by=bbox_c[1], bz=bbox_c[2],
                          bl=bbox_d[0], bw=bbox_d[1], bh=bbox_d[2])  # fmt: skip
        if boundary:
            q += ", n.boundary = [p IN $boundary | point({x: p[0], y: p[1], z: $z})]"
            params["boundary"] = boundary
        self.tx.run(q, **params)
        self.inverse.append({"op": "delete_node", "symbol": sym})
        self.changes.append({"kind": "create", "symbol": sym, "layer": label, "class": cls,
                             "name": props.get("name"), "pos": pos})  # fmt: skip
        self.touch(label, sym)
        return sym

    def op_update_node(self, op):
        n = self.node(op.get("symbol", ""))
        label, props, sym = n["label"], n["props"], n["symbol"]
        sets = dict(op.get("set") or {})
        if "pos" in sets:
            sets["center"] = sets.pop("pos")
        unset = [str(k) for k in op.get("unset") or []]
        for k in list(sets) + unset:
            if k not in EDITABLE_PROPERTIES:
                raise EditError(f"property {k!r} is not editable")
        for k, v in dict(op.get("expect") or {}).items():
            if not _same(props.get(k), v):
                raise EditConflict(
                    f"{sym}.{k} changed since this edit (now {plain(props.get(k))!r}, "
                    f"expected {plain(v)!r})"
                )
        if "class" in sets and sets["class"] is not None:
            sets["class"] = str(sets["class"]).strip().lower()
            if not sets["class"]:
                raise EditError("class cannot be empty")
            if LAYER_SPECS[label][4] and "semantic_label" not in sets:
                sid = self.ensure_class(label, sets["class"])
                if sid is not None:
                    sets["semantic_label"] = sid
        if "center" in sets:
            c = _xyz(sets["center"])
            old = plain(props.get("center"))
            if len(sets["center"]) == 2 and old:  # keep z when only x, y are given
                c[2] = old[2]
            sets["center"] = c
            sets.update(pos_x=c[0], pos_y=c[1], pos_z=c[2])
            # an object's box moves with it unless the edit sets the box too
            if old and props.get("bbox_center") is not None and "bbox_center" not in sets:
                bc = plain(props["bbox_center"])
                sets["bbox_center"] = [bc[i] + c[i] - old[i] for i in range(3)]
        for k in ("bbox_center", "bbox_dim"):
            if k in sets:
                sets[k] = _xyz(sets[k])
        if "boundary" in sets and sets["boundary"] is not None:
            z = float(plain(props.get("center") or [0, 0, 0])[2])
            sets["boundary"] = [[p[0], p[1], p[2] if len(p) > 2 else z] for p in _polygon(sets["boundary"])]
        changed = {k: v for k, v in sets.items() if not _same(props.get(k), v)}
        gone = [k for k in unset if k in props]
        if not changed and not gone:
            return
        clauses, params = [], {"syms": symbol_variants(sym)}
        for i, (k, v) in enumerate(changed.items()):
            if not _IDENT.match(k):
                raise EditError(f"bad property name {k!r}")
            if v is None:
                clauses.append(f"n.{k} = null")
            elif k in POINT3D_PROPERTIES:
                clauses.append(f"n.{k} = point({{x: $v{i}[0], y: $v{i}[1], z: $v{i}[2]}})")
                params[f"v{i}"] = v
            elif k in POINT_LIST_PROPERTIES:
                clauses.append(f"n.{k} = [p IN $v{i} | point({{x: p[0], y: p[1], z: p[2]}})]")
                params[f"v{i}"] = v
            else:
                clauses.append(f"n.{k} = $v{i}")
                params[f"v{i}"] = v
        for k in gone:
            clauses.append(f"n.{k} = null")
        self.tx.run(
            f"MATCH (n:{label}) WHERE n.nodeSymbol IN $syms SET " + ", ".join(clauses), **params
        )
        before = {k: plain(props[k]) for k in list(changed) + gone if k in props}
        inv_set = dict(before)
        inv_unset = [k for k in changed if k not in props]
        inv: dict[str, Any] = {"op": "update_node", "symbol": sym, "set": inv_set}
        if inv_unset:
            inv["unset"] = inv_unset
        inv["expect"] = {k: plain(v) for k, v in changed.items() if v is not None}
        self.inverse.append(inv)
        for k in list(changed) + gone:
            if k in ("pos_x", "pos_y", "pos_z", "semantic_label") or (
                k == "bbox_center" and "center" in changed
            ):
                continue
            self.changes.append({"kind": "update", "symbol": sym, "layer": label, "field": k,
                                 "old": plain(props.get(k)), "new": plain(changed.get(k))})  # fmt: skip
        self.touch(label, sym)

    def op_retire_node(self, op):
        n = self.node(op.get("symbol", ""))
        label, sym = n["label"], n["symbol"]
        reason = str(op.get("reason") or "retired by the operator")
        self.tx.run(
            f"MATCH (n:{label}) WHERE n.nodeSymbol = $s "
            f"REMOVE n:{label} SET n:{RETIRED}, n.retired_label = $label, "
            "n.retired_at_revision = $rev, n.retirement_reason = $reason",
            s=sym, label=label, rev=self.revision, reason=reason,
        )  # fmt: skip
        self.inverse.append({"op": "restore_node", "symbol": sym})
        self.changes.append({"kind": "retire", "symbol": sym, "layer": label,
                             "class": n["props"].get("class"), "name": n["props"].get("name")})  # fmt: skip
        self.touch(label, sym)

    def op_restore_node(self, op):
        n = self.node(op.get("symbol", ""), retired=True)
        label, sym = n["label"], n["symbol"]
        if label not in LAYER_SPECS:
            raise EditError(f"{sym}: unknown retired label {label!r}")
        reason = n["props"].get("retirement_reason")
        self.tx.run(
            f"MATCH (n:{RETIRED}) WHERE n.nodeSymbol = $s "
            f"REMOVE n:{RETIRED} SET n:{label} "
            "REMOVE n.retired_label, n.retired_at_revision, n.retirement_reason",
            s=sym,
        )
        self.inverse.append({"op": "retire_node", "symbol": sym, "reason": reason})
        self.changes.append({"kind": "restore", "symbol": sym, "layer": label,
                             "class": n["props"].get("class"), "name": n["props"].get("name")})  # fmt: skip
        self.touch(label, sym)

    def op_delete_node(self, op):
        n = self.node(op.get("symbol", ""), retired=None)
        label, sym = n["label"], n["symbol"]
        own = RETIRED if n["retired"] else label
        labels = self.rows(f"MATCH (n:{own}) WHERE n.nodeSymbol = $s RETURN labels(n) AS l", s=sym)[0]["l"]
        edges = self.rows(
            f"MATCH (n:{own}) WHERE n.nodeSymbol = $s MATCH (n)-[r]-(m) "
            "RETURN type(r) AS type, startNode(r) = n AS out, m.nodeSymbol AS other, "
            "labels(m) AS other_labels, elementId(m) AS other_eid, properties(r) AS props",
            s=sym,
        )
        props = n["props"]
        enc_props, points, point_lists = {}, {}, {}
        for k, v in props.items():
            if _is_point(v):
                points[k] = plain(v)
            elif isinstance(v, list) and v and _is_point(v[0]):
                point_lists[k] = plain(v)
            else:
                enc_props[k] = plain(v)
        self.tx.run(f"MATCH (n:{own}) WHERE n.nodeSymbol = $s DETACH DELETE n", s=sym)
        self.inverse.append({
            "op": "recreate_node", "labels": list(labels), "props": enc_props,
            "points": points, "point_lists": point_lists,
            "edges": [{**{k: plain(e[k]) for k in ("type", "out", "other", "other_eid", "props")},
                       "other_label": _first_label(e["other_labels"])} for e in edges],
        })  # fmt: skip
        self.changes.append({"kind": "delete", "symbol": sym, "layer": label,
                             "class": props.get("class"), "name": props.get("name"),
                             "edges": len(edges)})  # fmt: skip
        self.touch(label, sym)

    def op_recreate_node(self, op):
        labels = [str(x) for x in op.get("labels") or []]
        for lab in labels:
            if not _IDENT.match(lab):
                raise EditError(f"bad label {lab!r}")
        props = dict(op.get("props") or {})
        sym = props.get("nodeSymbol")
        if not sym or not labels:
            raise EditError("recreate_node needs labels and props.nodeSymbol")
        if self.rows(_BY_SYMBOL + "RETURN count(n) AS c", syms=[sym])[0]["c"]:
            raise EditError(f"node {sym} already exists")
        own = _first_label(labels)
        clauses = []
        params: dict[str, Any] = {"props": props}
        for i, (k, v) in enumerate(dict(op.get("points") or {}).items()):
            if not _IDENT.match(k):
                raise EditError(f"bad property name {k!r}")
            clauses.append(f"n.{k} = point({{x: $p{i}[0], y: $p{i}[1], z: $p{i}[2]}})")
            params[f"p{i}"] = _xyz(v)
        for i, (k, v) in enumerate(dict(op.get("point_lists") or {}).items()):
            if not _IDENT.match(k):
                raise EditError(f"bad property name {k!r}")
            clauses.append(f"n.{k} = [p IN $l{i} | point({{x: p[0], y: p[1], z: p[2]}})]")
            params[f"l{i}"] = [_xyz(p) for p in v]
        self.tx.run(
            f"CREATE (n:{':'.join(labels)}) SET n = $props"
            + ("".join(", " + c for c in clauses)),
            **params,
        )
        for e in op.get("edges") or []:
            t = str(e["type"])
            if not _IDENT.match(t):
                raise EditError(f"bad relationship type {t!r}")
            ol = e.get("other_label")
            if e.get("other") and ol and _IDENT.match(str(ol)):
                match_m = f"MATCH (m:{ol}) WHERE m.nodeSymbol = $other"
            elif e.get("other"):
                match_m = "MATCH (m) WHERE m.nodeSymbol = $other"
            else:
                match_m = "MATCH (m) WHERE elementId(m) = $eid"
            arrow = f"(n)-[r:{t}]->(m)" if e.get("out") else f"(m)-[r:{t}]->(n)"
            self.tx.run(
                f"MATCH (n:{own}) WHERE n.nodeSymbol = $s {match_m} CREATE {arrow} SET r = $rp",
                s=sym, other=e.get("other"), eid=e.get("other_eid"), rp=dict(e.get("props") or {}),
            )  # fmt: skip
        label = next((lab for lab in MAP_LABELS if lab in labels), None)
        if label is None and RETIRED in labels:
            label = props.get("retired_label")
        self.inverse.append({"op": "delete_node", "symbol": sym})
        self.changes.append({"kind": "recreate", "symbol": sym, "layer": label,
                             "class": props.get("class"), "name": props.get("name")})  # fmt: skip
        if label in LAYER_SPECS:
            self.touch(label, sym)

    # -- edges ------------------------------------------------------------
    def _edge_ends(self, op):
        a, b = self.node(op.get("from", "")), self.node(op.get("to", ""))
        t = op.get("type") or determine_edge_type(a["label"], b["label"])
        if t not in EDGE_TYPES:
            raise EditError(f"unknown edge type {t!r}")
        if t == INTERLAYER_EDGE_TYPE:
            la, lb = LAYER_SPECS[a["label"]][0], LAYER_SPECS[b["label"]][0]
            if la == lb:
                raise EditError(f"CONTAINS needs two layers ({a['symbol']}, {b['symbol']})")
            if la < lb:  # CONTAINS always points parent -> child
                a, b = b, a
        elif a["label"] != b["label"]:
            raise EditError(f"{t} joins two {a['label']} nodes, not {a['label']}-{b['label']}")
        return a, b, t

    def op_create_edge(self, op):
        a, b, t = self._edge_ends(op)
        if a["symbol"] == b["symbol"]:
            raise EditError("an edge needs two different nodes")
        pattern = "(a)-[r:%s]->(b)" if t == INTERLAYER_EDGE_TYPE else "(a)-[r:%s]-(b)"
        exists = self.rows(
            f"MATCH (a:{a['label']} {{nodeSymbol: $a}}), (b:{b['label']} {{nodeSymbol: $b}}) "
            f"RETURN exists {{ MATCH {pattern % t} }} AS e",
            a=a["symbol"], b=b["symbol"],
        )[0]["e"]  # fmt: skip
        if exists:
            return
        self.tx.run(
            f"MATCH (a:{a['label']} {{nodeSymbol: $a}}), (b:{b['label']} {{nodeSymbol: $b}}) "
            f"CREATE (a)-[:{t}]->(b)",
            a=a["symbol"], b=b["symbol"],
        )  # fmt: skip
        self.inverse.append({"op": "delete_edge", "from": a["symbol"], "to": b["symbol"], "type": t})
        self.changes.append({"kind": "edge+", "from": a["symbol"], "to": b["symbol"], "type": t})
        self.touch(a["label"], a["symbol"])
        self.touch(b["label"], b["symbol"])

    def op_delete_edge(self, op):
        a, b, t = self._edge_ends(op)
        pattern = "(a)-[r:%s]->(b)" if t == INTERLAYER_EDGE_TYPE else "(a)-[r:%s]-(b)"
        rows = self.rows(
            f"MATCH (a:{a['label']} {{nodeSymbol: $a}}), (b:{b['label']} {{nodeSymbol: $b}}) "
            f"MATCH {pattern % t} WITH r, startNode(r).nodeSymbol AS s, endNode(r).nodeSymbol AS e "
            "DELETE r RETURN s, e",
            a=a["symbol"], b=b["symbol"],
        )  # fmt: skip
        if not rows:
            raise EditError(f"no {t} edge between {a['symbol']} and {b['symbol']}")
        for r in rows:
            self.inverse.append({"op": "create_edge", "from": r["s"], "to": r["e"], "type": t})
            self.changes.append({"kind": "edge-", "from": r["s"], "to": r["e"], "type": t})
        self.touch(a["label"], a["symbol"])
        self.touch(b["label"], b["symbol"])

    # -- group ------------------------------------------------------------
    def members_in_polygon(self, poly, layers) -> list[dict]:
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        out = []
        for lab in layers:
            rows = self.rows(
                f"MATCH (n:{lab}) WHERE n.center IS NOT NULL "
                "AND n.center.x >= $x0 AND n.center.x <= $x1 "
                "AND n.center.y >= $y0 AND n.center.y <= $y1 "
                "RETURN n.nodeSymbol AS s, n.center AS c",
                x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys),
            )  # fmt: skip
            for r in rows:
                c = plain(r["c"])
                if point_in_polygon(c[0], c[1], poly):
                    out.append({"symbol": r["s"], "label": lab, "center": c})
        return out

    def op_group(self, op) -> str:
        parent = layer_label(op.get("parent_layer", constants.ROOMS))
        allowed = GROUP_CHILD_LAYERS.get(parent)
        if not allowed:
            raise EditError(f"cannot group under {parent}")
        member_layers = [layer_label(x) for x in op.get("member_layers") or allowed]
        bad = [x for x in member_layers if x not in allowed]
        if bad:
            raise EditError(f"a {parent} cannot contain {', '.join(bad)}")
        poly = _polygon(op["polygon"]) if op.get("polygon") else None
        if op.get("members"):
            members = []
            for s in op["members"]:
                n = self.node(s)
                if n["label"] not in allowed:
                    raise EditError(f"a {parent} cannot contain {n['label']} {n['symbol']}")
                members.append({"symbol": n["symbol"], "label": n["label"],
                                "center": plain(n["props"].get("center") or [0, 0, 0])})  # fmt: skip
        elif poly is not None:
            members = self.members_in_polygon(poly, member_layers)
        else:
            raise EditError("group needs members or a polygon")
        if not members and not op.get("allow_empty", poly is not None):
            raise EditError("nothing to group")
        zs = [m["center"][2] for m in members]
        z = sum(zs) / len(zs) if zs else 0.0
        if poly is not None:
            cx, cy = polygon_centroid(poly)
        elif members:
            cx = sum(m["center"][0] for m in members) / len(members)
            cy = sum(m["center"][1] for m in members) / len(members)
        else:
            raise EditError("nothing to group")
        create = {"op": "create_node", "layer": parent, "pos": [cx, cy, z],
                  "class": op.get("class"), "name": op.get("name"), "symbol": op.get("symbol"),
                  "source": op.get("source")}  # fmt: skip
        if poly is not None:
            create["boundary"] = poly
        if zs:
            create["height"] = max(1.0, max(zs) - min(zs))
        sym = self.op_create_node(create)
        created = self.changes.pop()  # the parent's "create" becomes the "group" change
        for lab in sorted({m["label"] for m in members}):
            self.tx.run(
                f"MATCH (p:{parent} {{nodeSymbol: $p}}) UNWIND $members AS m "
                f"MATCH (c:{lab} {{nodeSymbol: m}}) MERGE (p)-[:CONTAINS]->(c)",
                p=sym, members=[m["symbol"] for m in members if m["label"] == lab],
            )  # fmt: skip
        counts = Counter(m["label"] for m in members)
        for lab in counts:
            self.dirty.add(layer_key(lab))
        self.changes.append({**created, "kind": "group", "members": dict(counts),
                             "boundary_vertices": len(poly) if poly else 0})  # fmt: skip
        return sym

    # -- dispatch ---------------------------------------------------------
    OPS = {
        "create_node", "update_node", "retire_node", "restore_node", "delete_node",
        "recreate_node", "create_edge", "delete_edge", "group", "add_class", "set_labelspace",
    }  # fmt: skip

    def apply(self, op: Mapping) -> Any:
        if not isinstance(op, Mapping):
            raise EditError(f"an op is an object, got {type(op).__name__}")
        name = op.get("op")
        if name not in self.OPS:
            raise EditError(f"unknown op {name!r}")
        return getattr(self, f"op_{name}")(op)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _bump_versions(tx, dirty_layers, affected) -> tuple[int, int]:
    """heracles.graph_interface.notify_map_change, inside the unit of work."""
    rec = tx.run(
        "MERGE (s:_State {key: 'global'}) "
        "ON CREATE SET s.version = 1 "
        "ON MATCH SET s.version = s.version + 1 "
        "SET s.dirty_layers = $dl, s.affected_ids = $ai "
        "WITH s MERGE (m:_MapState) "
        "SET m.version = coalesce(m.version, 0) + 1 "
        "RETURN s.version AS sv, m.version AS mv",
        dl=list(dirty_layers),
        ai=list(affected)[:500],
    ).single(strict=True)
    return int(rec["sv"]), int(rec["mv"])


def _driver(db_or_driver):
    """A neo4j driver from a driver or a heracles ``Neo4jWrapper``."""
    return getattr(db_or_driver, "driver", None) or db_or_driver


# The unit of work MERGEs one _State per key and creates a GraphMutation per edit;
# without these constraints two concurrent first edits create two _State nodes and
# every later edit fails (ResultNotSingleError).  Created once per driver/database.
_EDIT_CONSTRAINTS = (
    "CREATE CONSTRAINT open_set_state_key_unique IF NOT EXISTS "
    "FOR (n:_State) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT open_set_mutation_id_unique IF NOT EXISTS "
    "FOR (n:GraphMutation) REQUIRE n.mutation_id IS UNIQUE",
)
_constrained: set[tuple[int, str]] = set()


def _ensure_edit_constraints(driver, database: str) -> None:
    key = (id(driver), database)
    if key in _constrained:
        return
    for stmt in _EDIT_CONSTRAINTS:
        try:
            driver.execute_query(stmt, database_=database)
        except Exception as e:  # noqa: BLE001 — e.g. existing duplicates; edits still work
            logger.warning("graph_edit: could not create constraint (%s): %s", stmt.split()[2], e)
    _constrained.add(key)


def new_mutation_id(prefix: str = "edit") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def apply_edit(
    db,
    ops: Sequence[Mapping],
    *,
    provenance: Mapping[str, Any],
    expected_revision: int | None = None,
    database: str = "neo4j",
    mutation_id: str | None = None,
    summary: str | None = None,
    dry_run: bool = False,
    annotations: Mapping[str, Any] | None = None,
) -> dict:
    """Apply ``ops`` in one unit of work; returns what happened.

    ``expected_revision`` is the ``open_set_graph`` revision the caller planned against
    (``RevisionConflict`` if it moved); ``None`` takes the current one.  ``dry_run``
    applies everything and rolls back: the result (changes, summary, member counts) is
    exactly what a commit would do, nothing is written.

    Returns ``{mutation_id, revision, changes, summary, affected, dirty_layers,
    state_version, map_version, dry_run}``.
    """
    ops = list(ops)
    if not ops:
        raise EditError("no ops")
    driver = _driver(db)
    _ensure_edit_constraints(driver, database)
    if expected_revision is None:
        expected_revision = graph_revision(driver, database=database)
    mid = mutation_id or new_mutation_id()
    with GraphUnitOfWork(
        driver,
        database=database,
        expected_revision=expected_revision,
        mutation_id=mid,
        provenance=dict(provenance),
    ) as uow:
        ap = _Applier(uow, uow.next_revision)
        for i, op in enumerate(ops):
            try:
                ap.apply(op)
            except EditError as e:
                raise type(e)(f"op {i + 1} ({op.get('op') if isinstance(op, Mapping) else '?'}): {e}") from None
        lines = summarize(ap.changes)
        out = {
            "mutation_id": mid,
            "revision": uow.next_revision,
            "changes": ap.changes,
            "summary": lines,
            "affected": ap.affected,
            "dirty_layers": sorted(ap.dirty),
            "state_version": None,
            "map_version": None,
            "dry_run": dry_run,
        }
        if dry_run or not ap.changes:
            uow.rollback()
            out["revision"] = uow.current_revision
            out["mutation_id"] = None if not dry_run else mid
            return out
        sv, mv = _bump_versions(uow, sorted(ap.dirty), ap.affected)
        inverse = list(reversed(ap.inverse))
        uow.annotations.update({
            "kind": MAP_EDIT_KIND,
            "summary": summary or "; ".join(lines)[:2000],
            "ops_json": json.dumps(ops, default=str),
            "inverse_json": json.dumps(inverse, default=str),
            "affected": ap.affected[:500],
            "state_version": sv,
            "map_version": mv,
            **dict(annotations or {}),
        })  # fmt: skip
        out["revision"] = uow.commit()
        out["state_version"], out["map_version"] = sv, mv
        return out


def undo_edit(
    db,
    mutation_id: str | None = None,
    *,
    provenance: Mapping[str, Any],
    database: str = "neo4j",
    force: bool = False,
) -> dict:
    """Undo a map edit (default: the latest one not yet undone, never an undo itself).

    Replays the stored inverse ops as a new unit of work.  An inverse update carries
    ``expect`` (the values the edit wrote): if a later edit changed them, the undo
    raises :class:`EditConflict` unless ``force``.
    """
    driver = _driver(db)
    if mutation_id:
        recs, _, _ = driver.execute_query(
            "MATCH (m:GraphMutation {mutation_id: $id}) RETURN properties(m) AS p",
            id=mutation_id, database_=database,
        )  # fmt: skip
    else:
        recs, _, _ = driver.execute_query(
            "MATCH (m:GraphMutation {kind: $k}) WHERE m.undone_by IS NULL AND m.undo_of IS NULL "
            "RETURN properties(m) AS p ORDER BY m.revision DESC LIMIT 1",
            k=MAP_EDIT_KIND, database_=database,
        )  # fmt: skip
    if not recs:
        raise EditError("nothing to undo" if not mutation_id else f"unknown edit {mutation_id}")
    m = dict(recs[0]["p"])
    if m.get("kind") != MAP_EDIT_KIND or not m.get("inverse_json"):
        raise EditError(f"{m.get('mutation_id')} is not an undoable map edit")
    if m.get("undone_by"):
        raise EditError(f"{m['mutation_id']} was already undone by {m['undone_by']}")
    inverse = json.loads(m["inverse_json"])
    if force:
        for op in inverse:
            op.pop("expect", None)
    target = m["mutation_id"]
    new_id = new_mutation_id("undo")
    driver_rev = graph_revision(driver, database=database)
    with GraphUnitOfWork(
        driver, database=database, expected_revision=driver_rev,
        mutation_id=new_id, provenance=dict(provenance),
    ) as uow:  # fmt: skip
        ap = _Applier(uow, uow.next_revision)
        for i, op in enumerate(inverse):
            try:
                ap.apply(op)
            except EditError as e:
                raise type(e)(f"cannot undo {target} (step {i + 1}): {e}") from None
        uow.run(
            "MATCH (m:GraphMutation {mutation_id: $id}) SET m.undone_by = $by", id=target, by=new_id
        )
        sv, mv = _bump_versions(uow, sorted(ap.dirty), ap.affected)
        lines = summarize(ap.changes)
        uow.annotations.update({
            "kind": MAP_EDIT_KIND,
            "summary": f"undo: {m.get('summary') or target}"[:2000],
            "ops_json": json.dumps(inverse, default=str),
            "inverse_json": json.dumps(list(reversed(ap.inverse)), default=str),
            "undo_of": target,
            "affected": ap.affected[:500],
            "state_version": sv,
            "map_version": mv,
        })  # fmt: skip
        rev = uow.commit()
    return {"mutation_id": new_id, "undo_of": target, "revision": rev, "changes": ap.changes,
            "summary": lines, "affected": ap.affected, "dirty_layers": sorted(ap.dirty),
            "state_version": sv, "map_version": mv, "dry_run": False}  # fmt: skip


def edit_history(db, *, limit: int = 50, database: str = "neo4j") -> list[dict]:
    """Recent graph mutations, newest first (map edits carry summary / undo links)."""
    recs, _, _ = _driver(db).execute_query(
        "MATCH (m:GraphMutation) RETURN properties(m) AS p "
        "ORDER BY m.revision DESC LIMIT $n",
        n=int(limit), database_=database,
    )  # fmt: skip
    out = []
    for r in recs:
        p = dict(r["p"])
        try:
            prov = json.loads(p.get("provenance_json") or "{}")
        except ValueError:
            prov = {}
        ops = p.get("ops_json")
        out.append({
            "mutation_id": p.get("mutation_id"),
            "revision": p.get("revision"),
            "kind": p.get("kind") or "mutation",
            "committed_at": plain(p.get("committed_at")),
            "summary": p.get("summary") or "",
            "provenance": prov,
            "undo_of": p.get("undo_of"),
            "undone_by": p.get("undone_by"),
            "undoable": bool(p.get("inverse_json")) and not p.get("undone_by"),
            "n_ops": len(json.loads(ops)) if ops else 0,
            "affected": list(p.get("affected") or []),
            "map_version": p.get("map_version"),
        })  # fmt: skip
    return out


def graph_revision(db, *, database: str = "neo4j") -> int:
    """The ``open_set_graph`` revision (0 before the first unit of work); read-only."""
    recs, _, _ = _driver(db).execute_query(
        "MATCH (s:_State {key: $k}) RETURN s.revision AS r",
        k=GraphUnitOfWork.STATE_KEY, database_=database,
    )  # fmt: skip
    return int(recs[0]["r"]) if recs and recs[0]["r"] is not None else 0


def edits_since(db, revision: int, *, database: str = "neo4j") -> int:
    """How many map edits committed after ``revision`` (for "map changed since planning")."""
    recs, _, _ = _driver(db).execute_query(
        "MATCH (m:GraphMutation {kind: $k}) WHERE m.revision > $r RETURN count(m) AS c",
        k=MAP_EDIT_KIND, r=int(revision), database_=database,
    )  # fmt: skip
    return int(recs[0]["c"]) if recs else 0


def regions(db, *, database: str = "neo4j") -> list[dict]:
    """Room nodes with name, class, boundary polygon and member counts. ``polygon_source``:
    "drawn" (the node's boundary) or "members" (derived from the ground its places cover)."""
    recs, _, _ = _driver(db).execute_query(
        f"MATCH (r:{constants.ROOMS}) OPTIONAL MATCH (r)-[:CONTAINS]->(c) "
        "WITH r, [l IN labels(c) WHERE l IN $labels][0] AS lab, count(c) AS n "
        "RETURN r.nodeSymbol AS symbol, r.name AS name, r.class AS class, r.center AS center, "
        "r.boundary AS boundary, r.source AS source, collect([lab, n]) AS members",
        labels=list(MAP_LABELS), database_=database,
    )  # fmt: skip
    out = []
    derived = None
    for r in recs:
        members = {lab: n for lab, n in r["members"] if lab}
        polygon = [p[:2] for p in plain(r["boundary"])] if r["boundary"] else None
        polygon_source = "drawn" if polygon else None
        if polygon is None:  # mapped regions carry no boundary: the ground their places cover
            if derived is None:
                derived = member_footprints(db, database=database)
            polygon = derived.get(r["symbol"])
            polygon_source = "members" if polygon else None
        out.append({
            "symbol": r["symbol"], "name": r["name"], "class": r["class"],
            "center": plain(r["center"]), "polygon": polygon, "polygon_source": polygon_source,
            "source": r["source"], "members": members,
        })  # fmt: skip
    return out


_footprint_cache: dict = {}


def footprint_polygon(places, *, close_m: float = 1.0, simplify_m: float = 2.0) -> list[list[float]] | None:
    """Polygon ([x, y] ring) of the ground ``places`` cover: the union of their footprints
    (``(center, radii)``: ray radii about the center, else a 0.5 m disc), gaps up to
    ``close_m`` closed. Of several pieces the largest is kept; holes (a road around a
    building) are dropped. None without shapely or places."""
    try:
        import shapely
        from shapely.geometry import Point, Polygon
    except ImportError:
        return None
    shapes = []
    for c, radii in places:
        radii = list(radii or [])
        if len(radii) >= 3:
            n = len(radii)
            shape = Polygon([(c[0] + d * math.cos(2 * math.pi * i / n), c[1] + d * math.sin(2 * math.pi * i / n))
                             for i, d in enumerate(radii)])  # fmt: skip
            shapes.append(shape if shape.is_valid else shape.buffer(0))
        else:
            shapes.append(Point(c[0], c[1]).buffer(0.5))
    if not shapes:
        return None
    area = shapely.union_all(shapes).buffer(close_m / 2).buffer(-close_m / 2)
    largest = max(getattr(area, "geoms", [area]), key=lambda g: g.area)
    if largest.is_empty:
        return None
    ring = largest.exterior.simplify(simplify_m).coords[:-1]
    return [[round(x, 2), round(y, 2)] for x, y in ring] if len(ring) >= 3 else None


def member_footprints(db, *, database: str = "neo4j") -> dict:
    """Room symbol -> ``footprint_polygon`` of its member places, for rooms without a
    drawn boundary. Cached per graph revision and map version."""
    from .graph_interface import _MAP_STATE

    ver, _, _ = _driver(db).execute_query(f"MATCH (m:{_MAP_STATE}) RETURN max(m.version) AS v", database_=database)
    key = (id(_driver(db)), database, graph_revision(db, database=database), ver[0]["v"] if ver else None)
    if key in _footprint_cache:
        return _footprint_cache[key]
    recs, _, _ = _driver(db).execute_query(
        f"MATCH (r:{constants.ROOMS})-[:CONTAINS]->(p) WHERE p.center IS NOT NULL "
        "AND (p:MeshPlace OR p:Place) AND r.boundary IS NULL "
        "RETURN r.nodeSymbol AS r, p.center AS c, p.radii AS radii",
        database_=database,
    )  # fmt: skip
    places: dict[str, list] = {}
    for rec in recs:
        places.setdefault(rec["r"], []).append((plain(rec["c"]), plain(rec["radii"])))
    out = {}
    for sym, members in places.items():
        poly = footprint_polygon(members)
        if poly:
            out[sym] = poly
    if len(_footprint_cache) > 16:
        _footprint_cache.clear()
    _footprint_cache[key] = out
    return out


# ---------------------------------------------------------------------------
# Human-readable diffs
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
        return "(" + ", ".join(f"{x:.1f}" for x in v) + ")"
    if v is None:
        return "-"
    return str(v)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


_LAYER_WORD = {
    constants.OBJECTS: "object", constants.PLACES: "place",
    constants.MESH_PLACES: "place", constants.ROOMS: "region", constants.BUILDINGS: "building",
}  # fmt: skip


def summarize(changes: Sequence[Mapping]) -> list[str]:
    """One line per change, e.g. ``O41 class vehicle -> crate`` or
    ``new region R0 'courtyard' containing 37 places, 5 objects``."""
    lines: list[str] = []
    edges_add: Counter = Counter()
    edges_del: Counter = Counter()
    for c in changes:
        k = c.get("kind")
        what = c.get("name") or c.get("class") or ""
        word = _LAYER_WORD.get(c.get("layer"), "node")
        if k == "update":
            lines.append(f"{c['symbol']} {c['field']} {_fmt(c.get('old'))} -> {_fmt(c.get('new'))}")
        elif k == "create":
            lines.append(f"new {word} {c['symbol']} '{what}' at {_fmt(c.get('pos'))}")
        elif k == "group":
            parts = Counter()
            for lab, n in (c.get("members") or {}).items():
                parts[_LAYER_WORD.get(lab, lab.lower())] += n
            inner = ", ".join(_plural(n, w) for w, n in parts.items()) or "nothing"
            lines.append(f"new {word} {c['symbol']} '{what}' containing {inner}")
        elif k == "retire":
            lines.append(f"remove {c['symbol']} ({what}) (retired, can be undone)")
        elif k == "restore":
            lines.append(f"restore {c['symbol']} ({what})")
        elif k == "delete":
            lines.append(f"delete {c['symbol']} ({what}) permanently, with {_plural(c.get('edges', 0), 'edge')}")
        elif k == "recreate":
            lines.append(f"re-create {c['symbol']} ({what})")
        elif k == "edge+":
            edges_add[c["type"]] += 1
            if edges_add[c["type"]] <= 3:
                lines.append(f"connect {c['from']} -> {c['to']} ({c['type']})")
        elif k == "edge-":
            edges_del[c["type"]] += 1
            if edges_del[c["type"]] <= 3:
                lines.append(f"disconnect {c['from']} -> {c['to']} ({c['type']})")
        elif k == "class":
            lines.append(f"new {c['layer']} class '{c['name']}'")
        elif k == "labelspace":
            lines.append(f"{c['layer']} labelspace restored")
    for t, n in edges_add.items():
        if n > 3:
            lines.append(f"... and {n - 3} more {t} edges")
    for t, n in edges_del.items():
        if n > 3:
            lines.append(f"... and {n - 3} more {t} edges removed")
    return lines


# ---------------------------------------------------------------------------
# Snapshots: db_to_spark_dsg + what a DSG cannot carry
# ---------------------------------------------------------------------------

# Node properties a DSG round trip drops or resets (spark_dsg_to_db stamps admission
# defaults; RoomNodeAttributes has no boundary), re-applied on restore.
SNAPSHOT_EXTRA_KEYS = (
    "boundary", "name", "class", "source", "blocked", "blocked_reason", "note",
    "created_at_revision",
    constants.ADMISSION_STATUS, constants.ADMISSION_POLICY_VERSION,
    constants.ADMISSION_REASON, constants.ADMISSION_BELIEF,
)  # fmt: skip
# Edges db_to_spark_dsg rebuilds; every other edge between map nodes (e.g. Room ->
# Object CONTAINS, Object -> Observation) is kept in the extras.
_DSG_EDGES = {
    ("OBJECT_CONNECTED", constants.OBJECTS, constants.OBJECTS),
    ("MESH_PLACE_CONNECTED", constants.MESH_PLACES, constants.MESH_PLACES),
    ("PLACE_CONNECTED", constants.PLACES, constants.PLACES),
    ("ROOM_CONNECTED", constants.ROOMS, constants.ROOMS),
    ("BUILDING_CONNECTED", constants.BUILDINGS, constants.BUILDINGS),
    ("CONTAINS", constants.MESH_PLACES, constants.OBJECTS),
    ("CONTAINS", constants.PLACES, constants.OBJECTS),
    ("CONTAINS", constants.ROOMS, constants.PLACES),
    ("CONTAINS", constants.ROOMS, constants.MESH_PLACES),
    ("CONTAINS", constants.BUILDINGS, constants.ROOMS),
}


def _exec(db, query, **params):
    records, _, _ = db.execute(query, **params)
    return records


def map_counts(db) -> dict:
    """Node count per map label + total relationships between map nodes."""
    out = {}
    for lab in MAP_LABELS:
        out[lab] = int(_exec(db, f"MATCH (n:{lab}) RETURN count(n) AS c")[0]["c"])
    out["edges"] = int(_exec(
        db,
        "MATCH (a)-[r]->(b) WHERE any(l IN labels(a) WHERE l IN $labs) "
        "AND any(l IN labels(b) WHERE l IN $labs) RETURN count(r) AS c",
        labs=list(MAP_LABELS),
    )[0]["c"])  # fmt: skip
    return out


def snapshot_extras(db) -> dict:
    """What :func:`export_snapshot` stores next to the DSG JSON."""
    nodes = {}
    for lab in MAP_LABELS:
        for r in _exec(db, f"MATCH (n:{lab}) RETURN n.nodeSymbol AS s, properties(n) AS p"):
            p = r["p"]
            keep = {k: plain(p[k]) for k in SNAPSHOT_EXTRA_KEYS if k in p}
            if lab != constants.ROOMS:
                keep.pop("boundary", None)  # Place2d boundaries round-trip through the DSG
            if keep:
                nodes[r["s"]] = {**keep, "__label": lab}
    edges = []
    for r in _exec(
        db,
        "MATCH (a)-[r]->(b) WHERE a.nodeSymbol IS NOT NULL AND b.nodeSymbol IS NOT NULL "
        "AND (any(l IN labels(a) WHERE l IN $labs) OR any(l IN labels(b) WHERE l IN $labs)) "
        "RETURN type(r) AS t, a.nodeSymbol AS a, b.nodeSymbol AS b, "
        "[l IN labels(a) WHERE l IN $labs][0] AS la, [l IN labels(b) WHERE l IN $labs][0] AS lb, "
        "labels(b)[0] AS lbx",
        labs=list(MAP_LABELS),
    ):  # fmt: skip
        if (r["t"], r["la"], r["lb"]) not in _DSG_EDGES:
            edges.append({"type": r["t"], "from": r["a"], "to": r["b"], "la": r["la"],
                          "lb": r["lb"] or r["lbx"]})
    meta = [dict(r["p"]) for r in _exec(db, "MATCH (m:_GraphMetadata) RETURN properties(m) AS p")]
    return {"nodes": nodes, "edges": edges, "graph_metadata": meta}


def export_snapshot(db, directory, *, include_mesh: bool = False) -> dict:
    """Write ``dsg.json`` (``db_to_spark_dsg``), ``extras.json`` and ``meta.json`` into
    ``directory``; returns the meta (counts, versions, revision).

    ``include_mesh`` copies the mesh of the source DSG (``_GraphMetadata{key:'source'}``)
    into the snapshot, as SGET's Save-As did.
    """
    import pathlib

    import spark_dsg

    from .graph_interface import db_to_spark_dsg, get_map_version, read_version

    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    G = db_to_spark_dsg(db)
    mesh = False
    if include_mesh:
        src = _exec(db, "MATCH (m:_GraphMetadata {key: 'source'}) RETURN m.file_path AS p")
        p = pathlib.Path(src[0]["p"]) if src and src[0]["p"] else None
        if p is not None and p.is_file():
            source = spark_dsg.DynamicSceneGraph.load(str(p))
            if source.has_mesh():
                G.mesh = source.mesh
                mesh = True
    G.save(str(d / "dsg.json"), include_mesh=mesh)
    extras = snapshot_extras(db)
    (d / "extras.json").write_text(json.dumps(extras, default=str))
    rev = _exec(db, "MATCH (s:_State {key: $k}) RETURN s.revision AS r", k=GraphUnitOfWork.STATE_KEY)
    meta = {
        "counts": map_counts(db),
        "state_version": read_version(db),
        "map_version": get_map_version(db),
        "revision": int(rev[0]["r"]) if rev and rev[0]["r"] is not None else 0,
        "mesh": mesh,
        "created_at": time.time(),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def restore_snapshot(db, directory, *, log=None) -> dict:
    """Replace the map in Neo4j with a snapshot.  DESTRUCTIVE: callers confirm first.

    Deletes every map-layer node (and ``:Retired`` ones) and the labelspaces, reloads
    ``dsg.json`` with ``spark_dsg_to_db``, re-applies the extras (region boundaries,
    admission/operator properties, non-DSG edges whose ends exist, graph metadata) and
    moves both version counters past where they were, so every reader reloads.  Agents,
    observations and the mutation history stay.  Returns the restored counts.
    """
    import pathlib

    import spark_dsg

    from .graph_interface import get_map_version, read_version, spark_dsg_to_db

    say = log or (lambda msg: logger.info(msg))
    d = pathlib.Path(directory)
    G = spark_dsg.DynamicSceneGraph.load(str(d / "dsg.json"))
    extras = json.loads((d / "extras.json").read_text()) if (d / "extras.json").is_file() else {}
    sv, mv = read_version(db), get_map_version(db)
    say(f"deleting the current map (versions {sv}/{mv}) ...")
    for lab in MAP_LABELS + (RETIRED,):
        while True:
            n = _exec(
                db,
                f"MATCH (n:{lab}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS c",
            )[0]["c"]  # fmt: skip
            if not n:
                break
    _exec(db, "MATCH (l:_Labelspace) DELETE l")
    say(f"loading {G.num_nodes()} nodes ...")
    spark_dsg_to_db(G, db)
    def lab_of(x):
        return x if x and _IDENT.match(str(x)) else None

    groups: dict[str | None, list] = {}
    for sym, p in (extras.get("nodes") or {}).items():
        lab = lab_of(p.get("__label"))
        groups.setdefault(lab, []).append(
            {"s": sym, "p": {k: v for k, v in p.items() if k not in ("boundary", "__label")}, "b": p.get("boundary")}
        )
    for lab, rows in groups.items():
        node = f"(n:{lab} {{nodeSymbol: row.s}})" if lab else "(n {nodeSymbol: row.s})"
        for i in range(0, len(rows), 2000):
            _exec(
                db,
                f"UNWIND $rows AS row MATCH {node} SET n += row.p "
                "FOREACH (_ IN CASE WHEN row.b IS NULL THEN [] ELSE [1] END | "
                "SET n.boundary = [p IN row.b | point({x: p[0], y: p[1], z: p[2]})])",
                rows=rows[i : i + 2000],
            )
    by_key: dict[tuple, list] = {}
    for e in extras.get("edges") or []:
        if _IDENT.match(str(e["type"])):
            key = (e["type"], lab_of(e.get("la")), lab_of(e.get("lb")))
            by_key.setdefault(key, []).append({"a": e["from"], "b": e["to"]})
    for (t, la, lb), es in by_key.items():
        ma = f"(a:{la} {{nodeSymbol: e.a}})" if la else "(a {nodeSymbol: e.a})"
        mb = f"(b:{lb} {{nodeSymbol: e.b}})" if lb else "(b {nodeSymbol: e.b})"
        for i in range(0, len(es), 5000):
            _exec(db, f"UNWIND $es AS e MATCH {ma} MATCH {mb} MERGE (a)-[:{t}]->(b)", es=es[i : i + 5000])
    for m in extras.get("graph_metadata") or []:
        if m.get("key"):
            _exec(db, "MERGE (g:_GraphMetadata {key: $k}) SET g += $p", k=m["key"], p=m)
    _exec(
        db,
        "MERGE (s:_State {key: 'global'}) SET s.version = $sv, s.dirty_layers = [], "
        "s.affected_ids = [] WITH s MERGE (m:_MapState) SET m.version = $mv",
        sv=int(sv) + 1, mv=int(mv) + 1,
    )  # fmt: skip
    counts = map_counts(db)
    say(f"restored: {counts}")
    return counts
