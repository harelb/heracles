"""
Bidirectional conversion between spark_dsg scene graphs and Neo4j.

Design
------
Node conversion uses **property-presence** rather than hardcoded per-type
functions.  ``node_to_dict()`` inspects which attributes a node actually has
(via ``hasattr``) and stores whatever it finds, along with an ``attr_type``
string (the Python class name) so reconstruction can create the right C++
attribute class.

This makes the pipeline robust to new attribute types (e.g.,
TravNodeAttributes in MESH_PLACES, KhronosObjectAttributes in OBJECTS)
without code changes.

Lossy properties
----------------
Some attribute data is NOT round-tripped through Neo4j:
- TravNodeAttributes boundary (radii, states) — deferred, store only position
- Place2dNodeAttributes boundary (polygon points) — deferred
- AgentNodeAttributes — stored as ``Agent`` rows (pose, stamp) that are not
  read back by ``db_to_spark_dsg``; dbow vectors are not stored

The source JSON file should be considered the authoritative copy.  Neo4j
stores the subset needed for visualization and editing.

Edge handling remains layer-aware since edge types are determined by the
layer relationship (intralayer vs interlayer), not by attribute types.
"""

import datetime
import logging
import os

import neo4j
import numpy as np
import parse
import spark_dsg

from . import constants

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribute type registry for reconstruction (Neo4j → spark_dsg)
# ---------------------------------------------------------------------------

# Maps attr_type strings (stored on Neo4j nodes) to spark_dsg attribute
# constructors.  Used by db_record_to_spark_attrs() to create the right
# C++ class when reconstructing a DSG from the database.
#
# Lives here (not in constants.py) because it maps to C++ classes and is
# only used during reconstruction — it's conversion logic, not a shared
# constant.
ATTR_TYPE_REGISTRY = {
    "ObjectNodeAttributes": spark_dsg.ObjectNodeAttributes,
    "PlaceNodeAttributes": spark_dsg.PlaceNodeAttributes,
    "Place2dNodeAttributes": spark_dsg.Place2dNodeAttributes,
    "RoomNodeAttributes": spark_dsg.RoomNodeAttributes,
    # e.g. hydra's open-vocabulary regions: name, label, color, semantic feature
    "SemanticNodeAttributes": spark_dsg.SemanticNodeAttributes,
    "NodeAttributes": spark_dsg.NodeAttributes,
}

# Conditionally register types that may not exist in older spark_dsg versions.
if hasattr(spark_dsg, "KhronosObjectAttributes"):
    ATTR_TYPE_REGISTRY["KhronosObjectAttributes"] = spark_dsg.KhronosObjectAttributes
if hasattr(spark_dsg, "TraversabilityNodeAttributes"):
    ATTR_TYPE_REGISTRY["TraversabilityNodeAttributes"] = (
        spark_dsg.TraversabilityNodeAttributes
    )
if hasattr(spark_dsg, "TravNodeAttributes"):
    ATTR_TYPE_REGISTRY["TravNodeAttributes"] = spark_dsg.TravNodeAttributes


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------


def initialize_db(db, keep_labels=()):
    """Delete every node (except those with one of keep_labels) and redo the indexes."""

    def try_drop_index(db, index_name):
        try:
            db.execute(f"DROP INDEX {index_name}")
        except neo4j.exceptions.DatabaseError:
            print(f"No index `{index_name}`")

    keep = " OR ".join(f"n:{label}" for label in keep_labels)
    db.execute(
        f"MATCH (n) WHERE NOT ({keep}) DETACH DELETE n"
        if keep
        else "MATCH (n) DETACH DELETE n"
    )

    try_drop_index(db, "object_node_symbol")
    try_drop_index(db, "place_node_symbol")
    try_drop_index(db, "mesh_place_node_symbol")
    try_drop_index(db, "room_node_symbol")
    try_drop_index(db, "building_node_symbol")
    try_drop_index(db, "agent_node_symbol")
    try_drop_index(db, "observation_node_symbol")

    db.execute(
        f"CREATE INDEX object_node_symbol FOR (n:{constants.OBJECTS}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX place_node_symbol FOR (n:{constants.PLACES}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX mesh_place_node_symbol FOR (n:{constants.MESH_PLACES}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX room_node_symbol FOR (n:{constants.ROOMS}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX building_node_symbol FOR (n:{constants.BUILDINGS}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX agent_node_symbol FOR (n:{constants.AGENTS}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX observation_node_symbol FOR (n:{constants.OBSERVATIONS}) ON (n.nodeSymbol)"
    )


# ---------------------------------------------------------------------------
# Generic node → dict conversion (spark_dsg → flat dict for Neo4j)
# ---------------------------------------------------------------------------


def node_to_dict(node, object_labelspace=None, room_labelspace=None):
    """Convert any spark_dsg node to a flat dict for Neo4j storage.

    Uses property presence — works with any attribute type without
    hardcoding.  Stores ``attr_type`` so reconstruction can create the
    right C++ class.  Warns if a semantic_label is not found in the
    labelspace.

    Note: some attribute data (boundary details, agent poses, mesh
    connections) is intentionally NOT stored.  See module docstring.
    """
    attrs = node.attributes
    d = {
        "nodeSymbol": node.id.str(True),
        "attr_type": type(attrs).__name__,
        "pos_x": float(attrs.position[0]),
        "pos_y": float(attrs.position[1]),
        "pos_z": float(attrs.position[2]),
    }

    # SemanticNodeAttributes fields (optional).
    if hasattr(attrs, "semantic_label"):
        # Preserve the ID even when this graph has no labelspace for the layer.
        d["semantic_label"] = int(attrs.semantic_label)
        ls = (
            room_labelspace
            if isinstance(attrs, spark_dsg.RoomNodeAttributes)
            else object_labelspace
        )
        label_key = str(attrs.semantic_label)
        if type(attrs) is spark_dsg.SemanticNodeAttributes:
            # hydra's open-vocabulary regions: semantic_label indexes the region
            # prompts, which hydra writes as the room labelspace (Hydra #210);
            # older graphs carried the matched prompt as the node's name instead
            if room_labelspace and label_key in room_labelspace:
                d["class"] = room_labelspace[label_key]
        elif ls and label_key in ls:
            d["class"] = ls[label_key]
        else:
            logger.warning(
                "Node %s: semantic_label %s not in labelspace",
                d["nodeSymbol"],
                label_key,
            )

    if hasattr(attrs, "name") and attrs.name:
        d["name"] = attrs.name

    # Color (SemanticNodeAttributes — 3 uint8 RGB values).
    if hasattr(attrs, "color"):
        try:
            c = attrs.color
            d["color_r"] = int(c[0])
            d["color_g"] = int(c[1])
            d["color_b"] = int(c[2])
        except Exception:
            pass

    # Bounding box (ObjectNodeAttributes, RoomNodeAttributes).
    if hasattr(attrs, "bounding_box"):
        try:
            if attrs.bounding_box.is_valid():
                bb = attrs.bounding_box
                d["bbox_x"] = float(bb.world_P_center[0])
                d["bbox_y"] = float(bb.world_P_center[1])
                d["bbox_z"] = float(bb.world_P_center[2])
                d["bbox_l"] = float(bb.dimensions[0])
                d["bbox_w"] = float(bb.dimensions[1])
                d["bbox_h"] = float(bb.dimensions[2])
        except Exception as e:
            logger.warning(
                "Node %s: failed to read bounding_box: %s",
                d["nodeSymbol"],
                e,
            )

    # Registered flag (ObjectNodeAttributes).
    if hasattr(attrs, "registered"):
        d["registered"] = bool(attrs.registered)

    # Base attributes (all nodes have these).
    d["is_active"] = bool(attrs.is_active)
    d["is_predicted"] = bool(attrs.is_predicted)

    # Distance to nearest obstacle (PlaceNodeAttributes).
    if hasattr(attrs, "distance"):
        d["distance"] = float(attrs.distance)

    # Observation timestamps (Traversability, TravNode have single int;
    # KhronosObjectAttributes has lists of ints — store as-is either way).
    if hasattr(attrs, "first_observed_ns"):
        val = attrs.first_observed_ns
        if isinstance(val, (list, tuple)):
            d["first_observed_ns"] = [int(v) for v in val]
        else:
            d["first_observed_ns"] = int(val)
    if hasattr(attrs, "last_observed_ns"):
        val = attrs.last_observed_ns
        if isinstance(val, (list, tuple)):
            d["last_observed_ns"] = [int(v) for v in val]
        else:
            d["last_observed_ns"] = int(val)

    # Room class probabilities — stored as parallel key/value lists
    # since Neo4j doesn't support map properties.
    if hasattr(attrs, "semantic_class_probabilities"):
        probs = attrs.semantic_class_probabilities
        if probs:
            d["class_prob_keys"] = list(probs.keys())
            d["class_prob_values"] = [float(v) for v in probs.values()]

    # TravNodeAttributes boundary (radii + states as flat lists).
    # States are TraversabilityState enums stored as ints.
    if hasattr(attrs, "radii"):
        d["radii"] = [float(r) for r in attrs.radii]
        d["states"] = [int(s) for s in attrs.states]
        d["min_radius"] = float(attrs.min_radius)
        d["max_radius"] = float(attrs.max_radius)

    # Place2dNodeAttributes boundary (polygon of 3D points).
    # Stored as temporary flat lists (boundary_x/y/z), then converted
    # to a native Neo4j Point3D list by insert_nodes_to_db.
    if hasattr(attrs, "boundary") and isinstance(attrs.boundary, list):
        boundary = attrs.boundary
        if boundary:
            d["boundary_x"] = [float(pt[0]) for pt in boundary]
            d["boundary_y"] = [float(pt[1]) for pt in boundary]
            d["boundary_z"] = [float(pt[2]) for pt in boundary]

    return d


# ---------------------------------------------------------------------------
# Generic bulk insert (flat dicts → Neo4j nodes)
# ---------------------------------------------------------------------------


def insert_nodes_to_db(db, layer_label, node_dicts):
    """Bulk insert nodes into Neo4j for a given layer.

    Uses MERGE on nodeSymbol (idempotent).  Position is stored as a Neo4j
    Point3D.  All other dict keys are set as scalar properties via ``n += node``.

    If any nodes have boundary_x/y/z (Place2d polygon points), a follow-up
    query converts them to a native Neo4j Point3D list for spatial queries.
    """
    if not node_dicts:
        logger.info("insert_nodes_to_db: no nodes to insert for layer %s", layer_label)
        return
    db.execute(
        f"""
        WITH $nodes AS nodes
        UNWIND nodes AS node
        WITH point({{x: node.pos_x, y: node.pos_y, z: node.pos_z}}) AS p3d, node
        MERGE (n:{layer_label} {{nodeSymbol: node.nodeSymbol}})
        SET n.center = p3d, n += node
        """,
        nodes=node_dicts,
    )

    # Convert flat bbox_x/y/z/l/w/h to Point3D bbox_center and bbox_dim.
    # db_record_to_spark_attrs() expects these as Point3D for reconstruction.
    has_bbox = any("bbox_x" in d for d in node_dicts)
    if has_bbox:
        db.execute(
            f"""
            MATCH (n:{layer_label})
            WHERE n.bbox_x IS NOT NULL
            SET n.bbox_center = point({{x: n.bbox_x, y: n.bbox_y, z: n.bbox_z}}),
                n.bbox_dim = point({{x: n.bbox_l, y: n.bbox_w, z: n.bbox_h}})
            REMOVE n.bbox_x, n.bbox_y, n.bbox_z, n.bbox_l, n.bbox_w, n.bbox_h
            """
        )

    # Convert flat boundary_x/y/z lists to native Point3D list.
    # This enables Neo4j spatial functions (point.distance, point.withinBBox)
    # on boundary points in future queries.
    has_boundary = any("boundary_x" in d for d in node_dicts)
    if has_boundary:
        db.execute(
            f"""
            MATCH (n:{layer_label})
            WHERE n.boundary_x IS NOT NULL
            WITH n, range(0, size(n.boundary_x)-1) AS indices
            SET n.boundary = [i IN indices |
                point({{x: n.boundary_x[i], y: n.boundary_y[i], z: n.boundary_z[i]}})]
            REMOVE n.boundary_x, n.boundary_y, n.boundary_z
            """
        )


# ---------------------------------------------------------------------------
# Per-layer wrappers (know which spark_dsg layer to iterate)
# ---------------------------------------------------------------------------


def add_objects_from_dsg(G, db, store=None, object_labelspace=None):
    """Insert Object nodes and, with an image ``store`` (see
    :func:`open_image_store`), ``(:Object)-[:HAS_OBSERVATION]->(:Observation)``
    for every store observation of each object (merged nodes included)."""
    if object_labelspace is None:
        object_labelspace = {}
    objects = list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)
    nodes = [node_to_dict(o, object_labelspace=object_labelspace) for o in objects]
    insert_nodes_to_db(db, constants.OBJECTS, nodes)

    if store is None:
        return
    observations = []
    edges = []
    for o in objects:
        sym = o.id.str(True)
        for d in _observation_dicts(sym, store.observations(int(o.id.value))):
            observations.append(d)
            edges.append({"from": sym, "to": d["nodeSymbol"]})
    if observations:
        insert_observations_to_db(db, observations)
        insert_edges(
            db,
            constants.HAS_OBSERVATION,
            constants.OBJECTS,
            constants.OBSERVATIONS,
            edges,
        )


def add_places_from_dsg(G, db):
    nodes = [node_to_dict(p) for p in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes]
    insert_nodes_to_db(db, constants.PLACES, nodes)


def add_mesh_places_from_dsg(G, db, object_labelspace=None):
    try:
        mesh_place_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except IndexError:
        mesh_place_layer = G.get_layer(20)

    if object_labelspace is None:
        object_labelspace = {}
    nodes = [
        node_to_dict(p, object_labelspace=object_labelspace)
        for p in mesh_place_layer.nodes
    ]
    insert_nodes_to_db(db, constants.MESH_PLACES, nodes)


def name_unnamed_regions(nodes):
    """Name hydra's open-vocabulary regions that have a class but no name.

    Hydra (#210) no longer writes a region's name; consoles, Cypher and the chat
    refer to regions by name.  Each unnamed region takes its class ("lawn");
    regions sharing a class are numbered in symbol order ("lawn", "lawn 2", ...).
    Named regions and other room types are left as they are.
    """
    taken = {n["name"] for n in nodes if n.get("name")}
    count = {}
    for n in sorted(nodes, key=lambda n: (len(n["nodeSymbol"]), n["nodeSymbol"])):
        if n.get("name") or not n.get("class") or n.get("attr_type") != "SemanticNodeAttributes":
            continue
        base = n["class"]
        k = count.get(base, 0)
        while True:
            k += 1
            name = base if k == 1 else f"{base} {k}"
            if name not in taken:
                break
        count[base] = k
        taken.add(name)
        n["name"] = name
    return nodes


def add_rooms_from_dsg(G, db, room_labelspace=None):
    if room_labelspace is None:
        room_labelspace = {}
    nodes = [
        node_to_dict(r, room_labelspace=room_labelspace)
        for r in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    ]
    insert_nodes_to_db(db, constants.ROOMS, name_unnamed_regions(nodes))


def add_buildings_from_dsg(G, db):
    nodes = [node_to_dict(b) for b in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes]
    insert_nodes_to_db(db, constants.BUILDINGS, nodes)


# ---------------------------------------------------------------------------
# Agents (robot keyframe poses)
# ---------------------------------------------------------------------------


def _agent_stamp_ns(attrs):
    """Agent node stamp in ns, None when unset (the binding defaults it to 0).

    The Python binding returns the stamp as a ``datetime.timedelta``, so the
    value has microsecond resolution (the last three digits are 0). Looking up
    the node's keyframe with ``hydra_python.image_store.ImageStore.agent_frame``
    goes by the agent node id and the nearest stamp within 50 ms, so this does
    not change which frame is found.
    """
    ts = getattr(attrs, "timestamp", None)
    if ts is None:
        return None
    ns = (ts // datetime.timedelta(microseconds=1)) * 1000
    return ns if ns > 0 else None


def _agent_nodes(G):
    """Every agent node of ``G``.

    Agents share layer id 2 with OBJECTS but live in non-zero partitions (one
    per robot prefix), so ``G.get_layer(DsgLayers.AGENTS)`` (partition 0) would
    return the objects. Enumerate the partitions instead.
    """
    agents_layer = spark_dsg.DsgLayers.name_to_layer_id("AGENTS").layer
    for key in G.layer_keys:
        if key.layer != agents_layer or key.partition == 0:
            continue
        for a in G.get_layer(key.layer, key.partition).nodes:
            if isinstance(a.attributes, spark_dsg.AgentNodeAttributes):
                yield a


def agent_to_dict(agent):
    """Flat ``Agent`` row: position, orientation (w, x, y, z) and stamp."""
    attrs = agent.attributes
    rot = attrs.world_R_body
    d = {
        "nodeSymbol": agent.id.str(True),
        "pos_x": float(attrs.position[0]),
        "pos_y": float(attrs.position[1]),
        "pos_z": float(attrs.position[2]),
        "rot_w": float(rot.w),
        "rot_x": float(rot.x),
        "rot_y": float(rot.y),
        "rot_z": float(rot.z),
    }
    ts_ns = _agent_stamp_ns(attrs)
    if ts_ns is not None:
        d["timestamp_ns"] = ts_ns
    return d


def insert_agents_to_db(db, agents):
    """MERGE ``Agent`` nodes (pose and stamp). The keyframe image properties
    (``frame_id``, ``rgb_path``, ``depth_path``) of an agent with an image store
    keyframe are only overwritten when the row has them."""
    return db.execute(
        f"""
    WITH $agents AS agents
    UNWIND agents AS agent
    WITH point({{x: agent.pos_x, y: agent.pos_y, z: agent.pos_z}}) AS p3d, agent
    MERGE (n:{constants.AGENTS} {{nodeSymbol: agent.nodeSymbol}})
    SET n.center = p3d,
        n.rot_w = agent.rot_w,
        n.rot_x = agent.rot_x,
        n.rot_y = agent.rot_y,
        n.rot_z = agent.rot_z,
        n.frame_id = coalesce(agent.frame_id, n.frame_id),
        n.rgb_path = coalesce(agent.rgb_path, n.rgb_path),
        n.depth_path = coalesce(agent.depth_path, n.depth_path),
        n.timestamp_ns = coalesce(agent.timestamp_ns, n.timestamp_ns)
    """,
        agents=agents,
    )


def add_agents_from_dsg(G, db, store=None, keyframes=None):
    """Insert one ``Agent`` node per agent (keyframe) node of ``G``.

    With an image ``store``, each agent with its own keyframe (see
    :func:`split_keyframes`) also gets ``frame_id`` and absolute ``rgb_path`` /
    ``depth_path``, and the exact frame stamp as ``timestamp_ns``.
    ``keyframes`` is ``split_keyframes(G, store)`` if already computed. Returns
    the number of agents written.
    """
    if store is None:
        keyframes = {}
    else:
        keyframes = (keyframes or split_keyframes(G, store))[0]
    agents = []
    for a in _agent_nodes(G):
        d = agent_to_dict(a)
        f = keyframes.get(int(a.id.value))
        if f is not None:
            d["frame_id"] = int(f.frame_id)
            d["rgb_path"] = _path_or_none(f.rgb_path)
            d["depth_path"] = _path_or_none(f.depth_path)
            # exact; the DSG attribute only has microsecond resolution
            d["timestamp_ns"] = int(f.timestamp_ns)
        agents.append(d)
    if agents:
        insert_agents_to_db(db, agents)
        insert_agent_edges_to_db(db, *agent_edges(G))
    return len(agents)


def _layer_key(name):
    key = spark_dsg.DsgLayers.name_to_layer_id(name)
    return (key.layer, key.partition)


# (layer, partition) of the place layers an agent can be in; 20 is the layer id
# mesh places had before they became partition 1 of the places layer.
_AGENT_PLACE_LAYERS = {
    _layer_key("PLACES"): constants.PLACES,
    _layer_key("MESH_PLACES"): constants.MESH_PLACES,
    (20, 0): constants.MESH_PLACES,
}


def _place_label(G, node_id):
    """Heracles label of a place or mesh place node, else None."""
    layer = G.get_node(node_id).layer
    return _AGENT_PLACE_LAYERS.get((layer.layer, layer.partition))


def agent_edges(G):
    """The agent edges of ``G``: ``(trajectory, place_agent)``.

    ``trajectory`` holds one ``{"from", "to"}`` row per edge between two agent
    nodes (the robot path), ``place_agent`` one row per edge between a place or
    mesh place and an agent, from the place.
    """
    agent_ids = {a.id.value for a in _agent_nodes(G)}
    trajectory, place_agent = [], []
    for a in _agent_nodes(G):
        sym = a.id.str(True)
        for sid in a.siblings():
            if sid in agent_ids and a.id.value < sid:
                to_ns = spark_dsg.NodeSymbol(sid).str(True)
                trajectory.append({"from": sym, "to": to_ns})
        for nid in list(a.parents()) + list(a.children()):
            label = _place_label(G, nid)
            if label is not None:
                from_ns = spark_dsg.NodeSymbol(nid).str(True)
                place_agent.append({"from": from_ns, "to": sym, "label": label})
    return trajectory, place_agent


def insert_agent_edges_to_db(db, trajectory, place_agent):
    """MERGE ``(:Agent)-[:AGENT_CONNECTED]->(:Agent)`` and
    ``(:Place|:MeshPlace)-[:CONTAINS]->(:Agent)``, so re-ingesting adds no
    duplicates. Edges to nodes that are not in the database are skipped."""
    if trajectory:
        db.execute(
            f"""
        WITH $edges AS edges
        UNWIND edges AS e
        MATCH (a:{constants.AGENTS} {{nodeSymbol: e.from}})
        MATCH (b:{constants.AGENTS} {{nodeSymbol: e.to}})
        MERGE (a)-[:AGENT_CONNECTED]->(b)
        """,
            edges=trajectory,
        )
    for label in (constants.PLACES, constants.MESH_PLACES):
        rows = [e for e in place_agent if e["label"] == label]
        if not rows:
            continue
        db.execute(
            f"""
        WITH $edges AS edges
        UNWIND edges AS e
        MATCH (p:{label} {{nodeSymbol: e.from}})
        MATCH (a:{constants.AGENTS} {{nodeSymbol: e.to}})
        MERGE (p)-[:CONTAINS]->(a)
        """,
            edges=rows,
        )


# ---------------------------------------------------------------------------
# Image store (hydra_python.image_store): keyframes, observations, cameras
# ---------------------------------------------------------------------------


class ImageStoreError(ValueError):
    """Raised when an ``image_store`` argument cannot be opened as an image store."""


def open_image_store(image_store, index=True):
    """Return a read-only ``hydra_python.image_store.ImageStore``, or None.

    ``image_store`` is a store directory, a run directory containing
    ``image_store/``, a list of those (e.g. one per robot) or an open
    ``ImageStore`` (returned as is). ``index`` is passed to ``ImageStore.open``:
    it builds the store of a run that has only raw image logs. Returns None for
    None or an empty list.
    """
    if image_store is None or (
        isinstance(image_store, (list, tuple)) and len(image_store) == 0
    ):
        return None
    from hydra_python.image_store import (
        FrameKindError,
        ImageStore,
        SchemaVersionError,
    )

    if isinstance(image_store, ImageStore):
        return image_store
    if isinstance(image_store, (str, os.PathLike)):
        image_store = [image_store]
    paths = [os.path.abspath(os.path.expanduser(str(p))) for p in image_store]
    try:
        store = ImageStore.open(*paths, index=index)
    except (FileNotFoundError, SchemaVersionError) as e:
        raise ImageStoreError(f"cannot open image store {paths}: {e}") from e
    try:
        store.info()  # every frame kind must be one this reader knows
    except FrameKindError as e:
        store.close()
        raise ImageStoreError(f"cannot read image store {paths}: {e}") from e
    return store


def _store_paths(store):
    """The store directories of an open ``ImageStore``, as strings."""
    return [str(r) for r in store.roots]


def _node_symbol_str(node_id):
    return spark_dsg.NodeSymbol(int(node_id)).str(True)


def _path_or_none(p):
    return None if p is None else str(p)


# An agent node's own keyframe: the keyframe anchored to it within this time of
# the node's stamp (``ImageStore.agent_frame``'s default).
AGENT_FRAME_MAX_DT_NS = 50_000_000
# Without a node stamp: the anchored keyframe taken at the node's own pose.
_IDENTITY_ATOL = 1e-3


def _agent_stamps(G):
    """``agent node id -> stamp ns`` for every agent node of ``G`` with a stamp."""
    out = {}
    if G is None:
        return out
    for a in _agent_nodes(G):
        ts = _agent_stamp_ns(a.attributes)
        if ts is not None:
            out[int(a.id.value)] = ts
    return out


def _anchor_offset(frame):
    """Largest entry of ``anchor_T_body - I`` (0 when the frame has none)."""
    if frame.anchor_T_body is None:
        return 0.0
    return float(np.max(np.abs(np.asarray(frame.anchor_T_body) - np.eye(4))))


def split_keyframes(G, store):
    """Split the store's keyframes into the agents' own keyframes and the others.

    Every keyframe is anchored to the agent node nearest in time. An agent's own
    keyframe is the one anchored to it within ``AGENT_FRAME_MAX_DT_NS`` of the
    node's stamp in ``G``; for an agent without a stamp, the anchored keyframe
    whose ``anchor_T_body`` is the identity. Returns ``(own, others)``:
    ``{agent node id: Frame}`` and the other keyframes in time order (these
    become ``SubKeyframe`` nodes). Frames whose anchors are absent from ``G``
    remain SubKeyframes and use the store's odometry pose.
    """
    from hydra_python.image_store import FrameKind

    frames = store.frames(kind=FrameKind.KEYFRAME)
    stamps = _agent_stamps(G)
    by_anchor = {}
    for f in frames:
        if f.anchor_node_id is not None:
            by_anchor.setdefault(int(f.anchor_node_id), []).append(f)
    own = {}
    for a, fs in by_anchor.items():
        # A missing anchor cannot receive an own keyframe: leaving its frames
        # in others preserves them with the odometry fallback at ingest.
        if G is not None and not G.has_node(a):
            continue
        ts = stamps.get(a)
        if ts is not None:
            best = min(
                fs, key=lambda f: (abs(f.timestamp_ns - ts), f.timestamp_ns, f.frame_id)
            )
            if abs(best.timestamp_ns - ts) <= AGENT_FRAME_MAX_DT_NS:
                own[a] = best
            continue
        candidates = [f for f in fs if _anchor_offset(f) < _IDENTITY_ATOL]
        if candidates:
            own[a] = min(
                candidates,
                key=lambda f: (_anchor_offset(f), f.timestamp_ns, f.frame_id),
            )
    own_ids = {f.frame_id for f in own.values()}
    return own, [f for f in frames if f.frame_id not in own_ids]


def subkeyframe_symbol(frame_id):
    """``nodeSymbol`` of a SubKeyframe node: ``s`` + the decimal store ``frame_id``.

    Store frame ids embed the robot id, so the symbol is unique across robots.
    """
    return f"s{int(frame_id)}"


def _rot_to_quat(R):
    """3x3 rotation matrix -> unit quaternion (w, x, y, z) with w >= 0."""
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        ]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        ]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        ]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        ]
    q = np.array(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def subkeyframe_to_dict(frame, pose):
    """One SubKeyframe row from a store keyframe that is not an agent's own.

    ``pose`` is the 4x4 world_T_body: the anchor agent's pose composed with the
    frame's anchor_T_body, or odometry when the anchor is unknown.
    """
    q = _rot_to_quat(pose[:3, :3])
    d = {
        "nodeSymbol": subkeyframe_symbol(frame.frame_id),
        "frame_id": int(frame.frame_id),
        "pos_x": float(pose[0, 3]),
        "pos_y": float(pose[1, 3]),
        "pos_z": float(pose[2, 3]),
        "rot_w": float(q[0]),
        "rot_x": float(q[1]),
        "rot_y": float(q[2]),
        "rot_z": float(q[3]),
        "timestamp_ns": int(frame.timestamp_ns),
        "rgb_path": _path_or_none(frame.rgb_path),
        "depth_path": _path_or_none(frame.depth_path),
        "sensor_id": int(frame.sensor_id),
    }
    if frame.anchor_node_id is not None:
        d["anchor_symbol"] = _node_symbol_str(frame.anchor_node_id)
    return d


def insert_subkeyframes_to_db(db, subframes):
    return db.execute(
        f"""
    WITH $subframes AS subframes
    UNWIND subframes AS s
    MERGE (n:{constants.SUBKEYFRAMES} {{nodeSymbol: s.nodeSymbol}})
    SET n.center = point({{x: s.pos_x, y: s.pos_y, z: s.pos_z}}),
        n.rot_w = s.rot_w, n.rot_x = s.rot_x, n.rot_y = s.rot_y, n.rot_z = s.rot_z,
        n.frame_id = s.frame_id,
        n.rgb_path = s.rgb_path,
        n.depth_path = s.depth_path,
        n.timestamp_ns = s.timestamp_ns
    """,
        subframes=subframes,
    )


def add_subkeyframes_from_dsg(G, db, store, keyframes=None):
    """Insert a ``SubKeyframe`` node for every store keyframe that is not an
    agent's own keyframe (:func:`split_keyframes`), with
    ``(:SubKeyframe)-[:ANCHORED_TO]->(:Agent)`` when it is anchored.

    The pose is the anchor agent's pose in ``G`` composed with the frame's
    anchor_T_body (odometry when the anchor is not in ``G``). ``keyframes`` is
    ``split_keyframes(G, store)`` if already computed. Returns the number of
    SubKeyframe nodes written.
    """
    if store is None:
        return 0
    others = (keyframes or split_keyframes(G, store))[1]
    subframes = [subkeyframe_to_dict(f, store.body_pose(f, G)[0]) for f in others]
    if not subframes:
        return 0
    insert_subkeyframes_to_db(db, subframes)
    edges = [
        {"from": s["nodeSymbol"], "to": s["anchor_symbol"]}
        for s in subframes
        if "anchor_symbol" in s
    ]
    insert_edges(
        db, constants.ANCHORED_TO, constants.SUBKEYFRAMES, constants.AGENTS, edges
    )
    return len(subframes)


def _observation_dicts(obj_symbol, observations):
    """Observation rows for one object node, one per store observation.

    ``nodeSymbol`` is ``<obj>_<timestamp_ns>``; when two observations of the
    object share a timestamp the later one gets ``_<obs_id>`` appended.
    """
    out, seen = [], set()
    for ob in observations:
        sym = f"{obj_symbol}_{ob.timestamp_ns}"
        if sym in seen:
            sym = f"{sym}_{ob.obs_id}"
        seen.add(sym)
        d = {
            "nodeSymbol": sym,
            "timestamp_ns": int(ob.timestamp_ns),
            "obs_id": int(ob.obs_id),
            "frame_id": int(ob.frame_id),
            "rgb_path": _path_or_none(ob.rgb_path),
            "depth_path": _path_or_none(ob.depth_path),
            "mask_path": _path_or_none(ob.mask_path),
            "mask_file": os.path.basename(str(ob.mask_path)) if ob.mask_path else None,
            # every observation has an RLE mask; only kept ones have a PNG
            "mask_rle": ob.mask_rle or None,
            "score": float(ob.score),
            "kept": bool(ob.kept),
        }
        if ob.bbox2d is not None:
            x0, y0, x1, y1 = (int(v) for v in ob.bbox2d)
            d["bbox_2d"] = [x0, y0, x1, y1]
        out.append(d)
    return out


def insert_observations_to_db(db, observations):
    # MERGE on nodeSymbol only, so re-ingesting updates in place.
    return db.execute(
        f"""
    WITH $observations AS observations
    UNWIND observations AS obs
    MERGE (o:{constants.OBSERVATIONS} {{nodeSymbol: obs.nodeSymbol}})
    SET o.timestamp_ns = obs.timestamp_ns,
        o.obs_id = obs.obs_id,
        o.frame_id = obs.frame_id,
        o.rgb_path = obs.rgb_path,
        o.depth_path = obs.depth_path,
        o.mask_path = obs.mask_path,
        o.mask_file = obs.mask_file,
        o.mask_rle = obs.mask_rle,
        o.bbox_2d = obs.bbox_2d,
        o.score = obs.score,
        o.kept = obs.kept
    """,
        observations=observations,
    )


# Frame-bearing labels an Observation is linked to by store frame_id.
_FRAME_LABELS = (constants.AGENTS, constants.SUBKEYFRAMES)


def ensure_provenance_indexes(db):
    """Idempotent indexes backing :func:`link_observations_to_frames` and
    :func:`attach_camera_calibs`."""
    for label in _FRAME_LABELS + (constants.OBSERVATIONS,):
        db.execute(
            f"CREATE INDEX {label.lower()}_timestamp_ns IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.timestamp_ns)"
        )
        db.execute(
            f"CREATE INDEX {label.lower()}_frame_id IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.frame_id)"
        )
    db.execute(
        f"CREATE INDEX {constants.CAMERA_CALIBS.lower()}_calib_id IF NOT EXISTS "
        f"FOR (n:{constants.CAMERA_CALIBS}) ON (n.calib_id)"
    )


def link_observations_to_frames(db):
    """MERGE ``(:Observation)-[:OBSERVED_IN]->(frame)`` for every Agent or
    SubKeyframe holding the store frame the observation was made in (same
    ``frame_id``; store frame ids embed the robot id, so robots whose frames
    share a stamp are not mixed up).

    Observations of frames that are not keyframes get no edge. Returns the
    number of matched (observation, frame) pairs.
    """
    ensure_provenance_indexes(db)
    total = 0
    for label in _FRAME_LABELS:
        records, _, _ = db.execute(
            f"""
        MATCH (o:{constants.OBSERVATIONS}) WHERE o.frame_id IS NOT NULL
        MATCH (f:{label} {{frame_id: o.frame_id}})
        MERGE (o)-[:{constants.OBSERVED_IN}]->(f)
        RETURN count(*) AS n
        """
        )
        total += records[0]["n"] if records else 0
    return total


def _calib_to_dict(calib):
    """Flat CameraCalib row from a store ``Calib`` (sensor)."""
    return {
        "calib_id": str(int(calib.sensor_id)),
        "sensor_id": int(calib.sensor_id),
        "fx": float(calib.fx),
        "fy": float(calib.fy),
        "cx": float(calib.cx),
        "cy": float(calib.cy),
        "width": int(calib.width),
        "height": int(calib.height),
        "depth_scale": float(calib.depth_scale),
        "body_T_sensor": [
            float(v) for v in np.asarray(calib.body_T_sensor).reshape(-1)
        ],
    }


def attach_camera_calibs(db, store, G=None, keyframes=None):
    """Insert a ``CameraCalib`` node per store sensor (``calib_id`` is the
    decimal ``sensor_id``) and ``HAS_CALIB`` edges to it from every Agent and
    anchored SubKeyframe. ``G`` supplies the agent stamps that pick each agent's
    own keyframe (:func:`split_keyframes`); ``keyframes`` is
    ``split_keyframes(G, store)`` if already computed. Returns the number of
    edges written.
    """
    ensure_provenance_indexes(db)
    calibs = [_calib_to_dict(c) for c in store.sensors() if c is not None]
    if not calibs:
        return 0
    db.execute(
        f"""
    UNWIND $calibs AS c
    MERGE (n:{constants.CAMERA_CALIBS} {{calib_id: c.calib_id}})
    SET n.sensor_id = c.sensor_id,
        n.fx = c.fx, n.fy = c.fy, n.cx = c.cx, n.cy = c.cy,
        n.width = c.width, n.height = c.height,
        n.depth_scale = c.depth_scale, n.body_T_sensor = c.body_T_sensor
    """,
        calibs=calibs,
    )
    known = {c["sensor_id"] for c in calibs}
    own, others = keyframes or split_keyframes(G, store)
    agent_pairs = [
        {"ns": _node_symbol_str(a), "calib_id": str(int(f.sensor_id))}
        for a, f in own.items()
        if int(f.sensor_id) in known
    ]
    sub_pairs = [
        {"ns": subkeyframe_symbol(f.frame_id), "calib_id": str(int(f.sensor_id))}
        for f in others
        if f.anchor_node_id is not None and int(f.sensor_id) in known
    ]
    n_edges = 0
    for label, pairs in (
        (constants.AGENTS, agent_pairs),
        (constants.SUBKEYFRAMES, sub_pairs),
    ):
        if not pairs:
            continue
        db.execute(
            f"""
        UNWIND $pairs AS pair
        MATCH (n:{label} {{nodeSymbol: pair.ns}})
        MATCH (c:{constants.CAMERA_CALIBS} {{calib_id: pair.calib_id}})
        MERGE (n)-[:{constants.HAS_CALIB}]->(c)
        """,
            pairs=pairs,
        )
        n_edges += len(pairs)
    return n_edges


# ---------------------------------------------------------------------------
# Top-level load: spark_dsg → Neo4j
# ---------------------------------------------------------------------------


def spark_dsg_to_db(G, db, source_file_path=None, image_store=None):
    """Load all nodes and edges from a spark_dsg graph into Neo4j.

    Extracts labelspaces from DSG metadata (embedded ``"labelspaces"`` key)
    and passes them to per-layer functions for semantic label → class name
    mapping.

    If ``source_file_path`` is provided, it is stored as a ``_GraphMetadata``
    node so that downstream tools (e.g., SGET) can locate the original file
    on disk for mesh data and other large assets not stored in Neo4j.

    ``image_store`` (anything :func:`open_image_store` accepts) adds the
    images Hydra recorded for the graph: keyframe paths on Agent nodes,
    SubKeyframe nodes for the other keyframes, Observation nodes for the
    objects, CameraCalib nodes, and the store directories as
    ``_GraphMetadata {key: 'image_store'}.image_store_paths``.
    """
    store = open_image_store(image_store)
    try:
        _spark_dsg_to_db(G, db, source_file_path, store)
    finally:
        # close only a store opened here, not one the caller passed in open
        if store is not None and store is not image_store:
            store.close()


def _spark_dsg_to_db(G, db, source_file_path, store):
    from .utils import extract_labelspaces_from_dsg

    object_ls, room_ls = extract_labelspaces_from_dsg(G)
    keyframes = split_keyframes(G, store) if store is not None else None

    add_agents_from_dsg(G, db, store, keyframes)
    add_objects_from_dsg(G, db, store, object_labelspace=object_ls)
    add_places_from_dsg(G, db)
    add_mesh_places_from_dsg(G, db, object_labelspace=object_ls)
    add_rooms_from_dsg(G, db, room_labelspace=room_ls)
    add_buildings_from_dsg(G, db)
    add_edges_from_dsg(G, db)
    # agents went in before the places: add their edges now that both exist
    insert_agent_edges_to_db(db, *agent_edges(G))

    if store is not None:
        add_subkeyframes_from_dsg(G, db, store, keyframes)
        attach_camera_calibs(db, store, G, keyframes)
        link_observations_to_frames(db)
        db.execute(
            "MERGE (m:_GraphMetadata {key: 'image_store'}) "
            "SET m.image_store_paths = $paths",
            paths=_store_paths(store),
        )

    # Store labelspaces in Neo4j so db_to_spark_dsg() can reconstruct
    # without requiring external YAML files or function arguments.
    if object_ls:
        ids = [int(k) for k in object_ls.keys()]
        names = list(object_ls.values())
        db.execute(
            "MERGE (m:_Labelspace {layer: 'object'}) "
            "SET m.ids = $ids, m.names = $names",
            ids=ids,
            names=names,
        )
    if room_ls:
        ids = [int(k) for k in room_ls.keys()]
        names = list(room_ls.values())
        db.execute(
            "MERGE (m:_Labelspace {layer: 'room'}) SET m.ids = $ids, m.names = $names",
            ids=ids,
            names=names,
        )

    if source_file_path is not None:
        abs_path = os.path.abspath(source_file_path)
        db.execute(
            "MERGE (m:_GraphMetadata {key: 'source'}) SET m.file_path = $path",
            path=abs_path,
        )


# ---------------------------------------------------------------------------
# Edge insertion (unchanged — layer-aware, not type-aware)
# ---------------------------------------------------------------------------


def add_edges_from_dsg(G, db):
    print("Adding Edges")
    meta = G.metadata.get() or {}
    if "LayerIdToHeraclesLayerStr" in meta:
        layer_id_to_layer_str = meta["LayerIdToHeraclesLayerStr"]
    else:
        # Build fallback: map both "N" (integer form, from .layer.layer) and
        # "N[P]" (full LayerKey form, from .layer) to heracles layer strings.
        _dsg_to_heracles = [
            ("OBJECTS", constants.OBJECTS),
            ("PLACES", constants.PLACES),
            ("MESH_PLACES", constants.MESH_PLACES),
            ("ROOMS", constants.ROOMS),
            ("BUILDINGS", constants.BUILDINGS),
            ("AGENTS", constants.AGENTS),
        ]
        layer_id_to_layer_str = {}
        for dsg_name, heracles_name in _dsg_to_heracles:
            lk = spark_dsg.DsgLayers.name_to_layer_id(dsg_name)
            if lk is not None:
                # setdefault so earlier entries (OBJECTS before AGENTS) win
                layer_id_to_layer_str.setdefault(str(lk), heracles_name)
                layer_id_to_layer_str.setdefault(str(lk.layer), heracles_name)

    object_object_edges = []
    for n in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        from_ns = n.id.str(True)
        for sid in n.siblings():
            to_ns = spark_dsg.NodeSymbol(sid).str(True)
            object_object_edges.append({"from": from_ns, "to": to_ns})

    insert_edges(
        db,
        "OBJECT_CONNECTED",
        constants.OBJECTS,
        constants.OBJECTS,
        object_object_edges,
    )

    print("Finished Object Edges")

    place_place_edges = []
    place_object_edges = []
    for n in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes:
        from_ns = n.id.str(True)
        for sid in n.siblings():
            to_ns = spark_dsg.NodeSymbol(sid).str(True)
            place_place_edges.append({"from": from_ns, "to": to_ns})

        for cid in n.children():
            to_ns = spark_dsg.NodeSymbol(cid).str(True)
            to_layer_id = G.get_node(cid).layer.layer
            to_layer_str = layer_id_to_layer_str[str(to_layer_id)]
            assert to_layer_str == constants.OBJECTS, (
                "Currently Places can only have Objects as children"
            )
            place_object_edges.append({"from": from_ns, "to": to_ns})

    insert_edges(
        db, "PLACE_CONNECTED", constants.PLACES, constants.PLACES, place_place_edges
    )
    insert_edges(
        db, "CONTAINS", constants.PLACES, constants.OBJECTS, place_object_edges
    )

    print("Finished Place Edges")

    mp_mp_edges = []
    mesh_place_object_edges = []
    try:
        mesh_place_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except IndexError:
        mesh_place_layer = G.get_layer(20)
    for n in mesh_place_layer.nodes:
        from_ns = n.id.str(True)
        for sid in n.siblings():
            to_ns = spark_dsg.NodeSymbol(sid).str(True)
            mp_mp_edges.append({"from": from_ns, "to": to_ns})
        for cid in n.children():
            to_ns = spark_dsg.NodeSymbol(cid).str(True)
            to_layer_id = G.get_node(cid).layer.layer
            to_layer_str = layer_id_to_layer_str[str(to_layer_id)]
            assert to_layer_str == constants.OBJECTS, (
                "Currently MeshPlaces can only have Objects as children"
            )
            mesh_place_object_edges.append({"from": from_ns, "to": to_ns})

    insert_edges(
        db,
        "MESH_PLACE_CONNECTED",
        constants.MESH_PLACES,
        constants.MESH_PLACES,
        mp_mp_edges,
    )
    insert_edges(
        db,
        "CONTAINS",
        constants.MESH_PLACES,
        constants.OBJECTS,
        mesh_place_object_edges,
    )
    print("Finished Mesh Place Edges")

    room_room_edges = []
    room_place_edges = []
    room_mesh_place_edges = []
    for n in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        from_ns = n.id.str(True)
        for sid in n.siblings():
            to_ns = spark_dsg.NodeSymbol(sid).str(True)
            room_room_edges.append({"from": from_ns, "to": to_ns})
        for cid in n.children():
            to_ns = spark_dsg.NodeSymbol(cid).str(True)
            to_layer_id = G.get_node(cid).layer
            to_layer_str = layer_id_to_layer_str[str(to_layer_id)]
            assert to_layer_str in [
                constants.PLACES,
                constants.MESH_PLACES,
            ], "Currently Rooms can only have Places or MeshPlaces as children"
            if to_layer_str == constants.PLACES:
                room_place_edges.append({"from": from_ns, "to": to_ns})
            elif to_layer_str == constants.MESH_PLACES:
                room_mesh_place_edges.append({"from": from_ns, "to": to_ns})

    insert_edges(
        db, "ROOM_CONNECTED", constants.ROOMS, constants.ROOMS, room_room_edges
    )
    insert_edges(db, "CONTAINS", constants.ROOMS, constants.PLACES, room_place_edges)
    insert_edges(
        db, "CONTAINS", constants.ROOMS, constants.MESH_PLACES, room_mesh_place_edges
    )

    print("Finished Room Edges")

    building_building_edges = []
    building_room_edges = []

    for n in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes:
        from_ns = n.id.str(True)
        for sid in n.siblings():
            to_ns = spark_dsg.NodeSymbol(sid).str(True)
            building_building_edges.append({"from": from_ns, "to": to_ns})
        for cid in n.children():
            to_ns = spark_dsg.NodeSymbol(cid).str(True)
            to_layer_id = G.get_node(cid).layer
            to_layer_str = layer_id_to_layer_str[str(to_layer_id)]
            assert to_layer_str == constants.ROOMS, (
                "Currently Buildings can only have Rooms as children"
            )
            building_room_edges.append({"from": from_ns, "to": to_ns})

    insert_edges(
        db,
        "BUILDING_CONNECTED",
        constants.BUILDINGS,
        constants.BUILDINGS,
        building_building_edges,
    )
    insert_edges(
        db, "CONTAINS", constants.BUILDINGS, constants.ROOMS, building_room_edges
    )
    print("Finished Building Edges")


def insert_edges(db, edge_type, from_label, to_label, connections):
    # MERGE, not CREATE: inserting an edge that is already there (loading the
    # same graph again) must not add a parallel duplicate relationship.
    query = f"""
    WITH $connections AS connections
    UNWIND connections AS connection
    MATCH (n1: {from_label} {{nodeSymbol: connection.from}})
    MATCH (n2: {to_label} {{nodeSymbol: connection.to}})
    MERGE (n1)-[:{edge_type}]->(n2)
    """
    ret = db.execute(query, connections=connections)
    return ret


# ---------------------------------------------------------------------------
# Generic node read (Neo4j → flat dict)
# ---------------------------------------------------------------------------


def get_layer_nodes(db, layer):
    """Fetch all nodes for a layer, returning all properties generically."""
    records, summary, keys = db.execute(
        f"""
        MATCH (p:{layer})
        RETURN properties(p) AS props
        """
    )
    return [dict(r["props"]) for r in records], summary, keys


def get_db_edges(db, edge_type, from_label, to_label):
    query = f"""
    MATCH (a:{from_label})-[:{edge_type}]->(b:{to_label})
    RETURN a.nodeSymbol as from, b.nodeSymbol as to
    """
    records, summary, keys = db.execute(query)
    return records, summary, keys


# ---------------------------------------------------------------------------
# Generic reconstruction (Neo4j → spark_dsg)
# ---------------------------------------------------------------------------


def db_record_to_spark_attrs(record, object_labelspace, room_labelspace):
    """Create a spark_dsg attribute object from a Neo4j property dict.

    Uses ``attr_type`` (stored on the node) to pick the right C++ class.
    Raises ValueError if attr_type is missing — the database must be
    reloaded from the source JSON to populate it.
    """
    attr_type = record.get("attr_type")
    if attr_type is None:
        raise ValueError(
            f"Node {record.get('nodeSymbol', '?')} has no attr_type property. "
            f"This database may have been populated by an older version of heracles. "
            f"Please reload the source JSON file to update the database."
        )
    cls = ATTR_TYPE_REGISTRY.get(attr_type, spark_dsg.NodeAttributes)
    attrs = cls()

    # Position (all nodes).
    if "center" in record:
        attrs.position = record["center"]

    # Name (SemanticNodeAttributes).
    if hasattr(attrs, "name") and "name" in record:
        attrs.name = record["name"]

    # Keep unmapped IDs; an explicitly mapped class still takes precedence.
    if hasattr(attrs, "semantic_label") and "semantic_label" in record:
        attrs.semantic_label = int(record["semantic_label"])

    # Semantic label → class (SemanticNodeAttributes).
    if hasattr(attrs, "semantic_label") and "class" in record:
        ls = (
            room_labelspace
            if isinstance(attrs, spark_dsg.RoomNodeAttributes)
            else object_labelspace
        )
        if record["class"] in ls:
            attrs.semantic_label = ls[record["class"]]

    # Color (SemanticNodeAttributes).
    if hasattr(attrs, "color") and "color_r" in record:
        import numpy as np

        attrs.color = np.array(
            [record["color_r"], record["color_g"], record["color_b"]], dtype=np.uint8
        )

    # Bounding box (ObjectNodeAttributes).
    if hasattr(attrs, "bounding_box") and "bbox_dim" in record:
        attrs.bounding_box = spark_dsg.BoundingBox(
            [record["bbox_dim"][0], record["bbox_dim"][1], record["bbox_dim"][2]],
            [
                record["bbox_center"][0],
                record["bbox_center"][1],
                record["bbox_center"][2],
            ],
        )

    # Registered flag (ObjectNodeAttributes).
    if hasattr(attrs, "registered") and "registered" in record:
        attrs.registered = bool(record["registered"])

    # Distance (PlaceNodeAttributes).
    if hasattr(attrs, "distance") and "distance" in record:
        attrs.distance = float(record["distance"])

    # Observation timestamps.
    # KhronosObjectAttributes exposes these as read-only lists; standard types
    # use a settable scalar. Try to set, skip gracefully if read-only.
    for ts_field in ("first_observed_ns", "last_observed_ns"):
        if hasattr(attrs, ts_field) and ts_field in record:
            try:
                setattr(attrs, ts_field, int(record[ts_field]))
            except (AttributeError, TypeError):
                pass  # Read-only on this attr type (e.g., KhronosObjectAttributes)

    # TravNodeAttributes boundary.
    if hasattr(attrs, "radii") and "radii" in record:
        attrs.radii = [float(r) for r in record["radii"]]
        if "states" in record:
            # Convert ints back to TraversabilityState enums for the C++ binding.
            attrs.states = [
                spark_dsg.TraversabilityState(int(s)) for s in record["states"]
            ]
        if "min_radius" in record:
            attrs.min_radius = float(record["min_radius"])
        if "max_radius" in record:
            attrs.max_radius = float(record["max_radius"])

    # Place2dNodeAttributes boundary (stored as Point3D list in Neo4j).
    if hasattr(attrs, "boundary") and isinstance(
        getattr(attrs, "boundary", None), list
    ):
        if "boundary" in record and isinstance(record["boundary"], list):
            import numpy as np

            attrs.boundary = [
                np.array([float(pt[0]), float(pt[1]), float(pt[2])])
                for pt in record["boundary"]
            ]

    return attrs


def insert_edges_to_spark(G, records):
    for record in records:
        G.insert_edge(str_to_ns_value(record["from"]), str_to_ns_value(record["to"]))
    return


def str_to_ns_value(string):
    p = parse.parse("{:l}({:d})", string)  # idx in parenthesis
    if not p:
        p = parse.parse("{:l}{:d}", string)  # idx not in parenthesis
    if not p:
        raise ValueError(f"Unexpected Node ID: {string}")

    key = p.fixed[0]
    idx = int(p.fixed[1])
    ns = spark_dsg.NodeSymbol(key, idx)
    return ns.value


def add_edges_from_db(db, G):
    #### INTRALAYER EDGES
    records, _, _ = get_db_edges(
        db, "OBJECT_CONNECTED", constants.OBJECTS, constants.OBJECTS
    )
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(
        db, "MESH_PLACE_CONNECTED", constants.MESH_PLACES, constants.MESH_PLACES
    )
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(
        db, "PLACE_CONNECTED", constants.PLACES, constants.PLACES
    )
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(db, "ROOM_CONNECTED", constants.ROOMS, constants.ROOMS)
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(
        db, "BUILDING_CONNECTED", constants.BUILDINGS, constants.BUILDINGS
    )
    insert_edges_to_spark(G, records)

    #### INTERLAYER EDGES
    records, _, _ = get_db_edges(
        db, "CONTAINS", constants.MESH_PLACES, constants.OBJECTS
    )
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(db, "CONTAINS", constants.PLACES, constants.OBJECTS)
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(db, "CONTAINS", constants.ROOMS, constants.PLACES)
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(db, "CONTAINS", constants.ROOMS, constants.MESH_PLACES)
    insert_edges_to_spark(G, records)
    records, _, _ = get_db_edges(db, "CONTAINS", constants.BUILDINGS, constants.ROOMS)
    insert_edges_to_spark(G, records)
    return


_DEFAULT_LAYER_MAP = {
    2: constants.OBJECTS,
    3: constants.PLACES,
    4: constants.ROOMS,
    5: constants.BUILDINGS,
    20: constants.MESH_PLACES,
}


def _read_labelspace_from_db(db, layer_key):
    """Read a labelspace from Neo4j's _Labelspace node. Returns {name: id}."""
    records, _, _ = db.execute(
        "MATCH (m:_Labelspace {layer: $layer}) RETURN m.ids AS ids, m.names AS names",
        layer=layer_key,
    )
    if not records:
        logger.warning("No _Labelspace node found for layer '%s' in Neo4j", layer_key)
        return {}
    ids = records[0]["ids"]
    names = records[0]["names"]
    return {name: sid for sid, name in zip(ids, names)}


def db_to_spark_dsg(
    db, layer_id_to_layer_name=None, object_labelspace=None, room_labelspace=None
):
    """Reconstruct a spark_dsg DynamicSceneGraph from Neo4j.

    Reads labelspaces from ``_Labelspace`` nodes in the database.
    Uses ``attr_type`` stored on each node to create the correct attribute
    class.

    The optional arguments keep the older four-argument form working:
    ``object_labelspace`` / ``room_labelspace`` (``{name: id}``) are used for a
    layer whose ``_Labelspace`` the database does not have, and
    ``layer_id_to_layer_name`` is ignored (the layers are fixed).
    """
    label_to_semantic_id = _read_labelspace_from_db(db, "object") or dict(
        object_labelspace or {}
    )
    room_label_to_semantic_id = _read_labelspace_from_db(db, "room") or dict(
        room_labelspace or {}
    )

    new_scene_graph = spark_dsg.DynamicSceneGraph()
    new_scene_graph.clear(True)

    # Write labelspaces so they persist in the exported DSG metadata.
    if label_to_semantic_id:
        object_labelspace = spark_dsg.Labelspace(
            {v: k for k, v in label_to_semantic_id.items()}
        )
        new_scene_graph.set_labelspace(object_labelspace, 2, 0)

    if room_label_to_semantic_id:
        room_labelspace = spark_dsg.Labelspace(
            {v: k for k, v in room_label_to_semantic_id.items()}
        )
        new_scene_graph.set_labelspace(room_labelspace, 4, 0)

    for spark_layer_id, heracles_layer_name in _DEFAULT_LAYER_MAP.items():
        if spark_layer_id == 20:
            new_scene_graph.add_layer(
                3,
                1,
                constants.HERACLES_TO_SPARK_LAYER_NAMES[heracles_layer_name],
            )
        else:
            new_scene_graph.add_layer(
                spark_layer_id,
                0,
                constants.HERACLES_TO_SPARK_LAYER_NAMES[heracles_layer_name],
            )

        records, summary, keys = get_layer_nodes(db, heracles_layer_name)

        for record in records:
            attrs = db_record_to_spark_attrs(
                record, label_to_semantic_id, room_label_to_semantic_id
            )
            new_scene_graph.add_node(
                constants.HERACLES_TO_SPARK_LAYER_NAMES[heracles_layer_name],
                str_to_ns_value(record["nodeSymbol"]),
                attrs,
            )

    add_edges_from_db(db, new_scene_graph)
    return new_scene_graph


# ---------------------------------------------------------------------------
# Mesh path (the mesh stays on disk; Neo4j stores where it is)
# ---------------------------------------------------------------------------


def store_mesh_path(db, mesh_path, mesh_format=None):
    """Record the scene mesh file so viewers can load it directly.

    Stored as ``file_path`` / ``format`` on ``_GraphMetadata {key: 'mesh'}``.
    ``mesh_format`` defaults to the file extension (e.g. ``"ply"``).
    """
    import os

    abs_path = os.path.abspath(str(mesh_path))
    fmt = mesh_format or os.path.splitext(abs_path)[1].lstrip(".").lower() or "ply"
    db.execute(
        "MERGE (m:_GraphMetadata {key: 'mesh'}) SET m.file_path = $path, m.format = $fmt",
        path=abs_path,
        fmt=fmt,
    )


def read_mesh_path(db):
    """Return ``(file_path, format)`` of the scene mesh, or ``(None, None)``."""
    records, _, _ = db.execute(
        "MATCH (m:_GraphMetadata {key: 'mesh'}) RETURN m.file_path AS p, m.format AS f"
    )
    if records:
        return records[0]["p"], records[0]["f"]
    return None, None
