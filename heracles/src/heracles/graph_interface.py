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
- AgentNodeAttributes (world_R_body, dbow) — not stored

The source JSON file should be considered the authoritative copy.  Neo4j
stores the subset needed for visualization and editing.

Edge handling remains layer-aware since edge types are determined by the
layer relationship (intralayer vs interlayer), not by attribute types.
"""

import datetime
import json
import logging
import os
import re
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


def initialize_db(db):
    def try_drop_index(db, index_name):
        try:
            db.execute(f"DROP INDEX {index_name}")
        except neo4j.exceptions.DatabaseError:
            print(f"No index `{index_name}`")

    db.execute("MATCH (n) DETACH DELETE n")

    try_drop_index(db, "object_node_symbol")
    try_drop_index(db, "place_node_symbol")
    try_drop_index(db, "mesh_place_node_symbol")
    try_drop_index(db, "room_node_symbol")
    try_drop_index(db, "building_node_symbol")
    try_drop_index(db, "observation_node_symbol")
    try_drop_index(db, "agent_node_symbol")

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
        f"CREATE INDEX observation_node_symbol FOR (n:{constants.OBSERVATIONS}) ON (n.nodeSymbol)"
    )
    db.execute(
        f"CREATE INDEX agent_node_symbol FOR (n:{constants.AGENTS}) ON (n.nodeSymbol)"
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

    # Explicit layer/partition ints so the renderer can group/Z-offset without
    # inferring them from the node label. (node.layer is a LayerKey.)
    try:
        lk = node.layer
        d["layer"] = int(getattr(lk, "layer", lk))
        d["partition"] = int(getattr(lk, "partition", 0))
    except Exception:
        pass

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
            # prompts, not a labelspace; the matched prompt is the node's name
            pass
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
                # Oriented box: store rotation as a quaternion (w,x,y,z) so the
                # renderer can orient the box. Axis-aligned boxes (has_rotation
                # False) stay implicit-identity to keep the row small.
                try:
                    if bb.has_rotation():
                        import numpy as _np
                        import trimesh.transformations as _tt
                        _M = _np.eye(4)
                        _M[:3, :3] = _np.array(bb.world_R_center)
                        _q = _tt.quaternion_from_matrix(_M)  # [w, x, y, z]
                        d["bbox_qw"] = float(_q[0])
                        d["bbox_qx"] = float(_q[1])
                        d["bbox_qy"] = float(_q[2])
                        d["bbox_qz"] = float(_q[3])
                except Exception:
                    pass
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


def obj_to_dict(node_classes, obj):
    """Legacy per-type conversion for ObjectNodeAttributes.

    Kept for backward compatibility (imported by test_db.py).
    New code should use node_to_dict() instead.
    """
    attrs = obj.attributes
    d = {}
    d["nodeSymbol"] = obj.id.str(True)
    d["pos_x"] = attrs.position[0]
    d["pos_y"] = attrs.position[1]
    d["pos_z"] = attrs.position[2]
    d["bbox_x"] = attrs.bounding_box.world_P_center[0]
    d["bbox_y"] = attrs.bounding_box.world_P_center[1]
    d["bbox_z"] = attrs.bounding_box.world_P_center[2]
    d["bbox_l"] = attrs.bounding_box.dimensions[0]
    d["bbox_w"] = attrs.bounding_box.dimensions[1]
    d["bbox_h"] = attrs.bounding_box.dimensions[2]
    d["class"] = node_classes[str(attrs.semantic_label)]
    d["name"] = attrs.name  # SemanticNodeAttribute::name

    # Specific to Khronos objects
    if hasattr(attrs, "details"):
        d["details"] = json.dumps(attrs.details)

    if hasattr(attrs, "first_observed_ns"):
        d["first_observed_ns"] = attrs.first_observed_ns

    if hasattr(attrs, "last_observed_ns"):
        d["last_observed_ns"] = attrs.last_observed_ns

    return d


# ---------------------------------------------------------------------------
# Generic bulk insert (flat dicts → Neo4j nodes)
# ---------------------------------------------------------------------------


def insert_nodes_to_db(db, layer_label, node_dicts, default_properties=None):
    """Bulk insert nodes into Neo4j for a given layer.

    Uses MERGE on nodeSymbol (idempotent).  Position is stored as a Neo4j
    Point3D.  All other dict keys are set as scalar properties via ``n += node``.

    ``default_properties`` are *defaults*, not values: each is written with
    ``coalesce(n.<key>, $default)``, so a property already on the node survives
    the re-ingest. ``n += node`` cannot express that -- it overwrites -- which
    is exactly how a re-ingest of the same nodeSymbol used to reset a graded
    ``admission_status`` back to ``trusted_prior`` (see add_objects_from_dsg).
    Backfills nodes written before the property existed, unlike ON CREATE SET.

    If any nodes have boundary_x/y/z (Place2d polygon points), a follow-up
    query converts them to a native Neo4j Point3D list for spatial queries.
    """
    if not node_dicts:
        logger.info("insert_nodes_to_db: no nodes to insert for layer %s", layer_label)
        return

    defaults_clause = ""
    if default_properties:
        for key in default_properties:
            # These become Cypher identifiers; they are module constants, but
            # refuse anything that could escape the property name position.
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Unsafe default property name: {key!r}")
        defaults_clause = "\nSET " + ", ".join(
            f"n.{key} = coalesce(n.{key}, $defaults.{key})"
            for key in default_properties
        )

    db.execute(
        f"""
        WITH $nodes AS nodes
        UNWIND nodes AS node
        WITH point({{x: node.pos_x, y: node.pos_y, z: node.pos_z}}) AS p3d, node
        MERGE (n:{layer_label} {{nodeSymbol: node.nodeSymbol}})
        SET n.center = p3d, n += node{defaults_clause}
        """,
        nodes=node_dicts,
        defaults=dict(default_properties or {}),
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
# Image store access
# ---------------------------------------------------------------------------


class ImageStoreError(ValueError):
    """Raised when an ``image_store`` argument does not point at an image store."""


def open_image_store(image_store):
    """Return a read-only ``hydra_image_store.ImageStore`` for ``image_store``.

    ``image_store`` is one store directory, a list/tuple of store directories (one per
    robot), or an already-open ``ImageStore`` (returned as is). A run directory that
    contains ``image_store/`` is accepted too. A legacy run directory (images/, agents/,
    subkeyframes/ folders and no ``store.sqlite``) raises :class:`ImageStoreError`.
    Returns None for None / empty input.
    """
    if image_store is None or (
        isinstance(image_store, (list, tuple)) and len(image_store) == 0
    ):
        return None
    try:
        from hydra_image_store import ImageStore
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "heracles needs the `hydra_image_store` package to ingest images "
            "(pip install -e <Hydra>/hydra_image_store)"
        ) from e
    if isinstance(image_store, ImageStore):
        return image_store
    if isinstance(image_store, (str, os.PathLike)):
        image_store = [image_store]
    resolved = []
    for p in image_store:
        base = os.path.abspath(os.path.expanduser(str(p)))
        candidates = [base, os.path.join(base, "image_store")]
        found = None
        for c in candidates:
            if os.path.isfile(c) and c.endswith(".sqlite"):
                found = c
            elif os.path.isfile(os.path.join(c, "store.sqlite")):
                found = c
            if found:
                break
        if found is None:
            raise ImageStoreError(
                f"{base} is not an image store (no store.sqlite). If it is a legacy "
                "run directory with image folders, convert it first: "
                f"`hydra-image-store import-legacy --run-dir {base} --out <store_dir> "
                "--dsg <dsg.json>`, then pass the new store as `image_store`."
            )
        resolved.append(found)
    return ImageStore.open(*resolved)


def _store_paths(store):
    return [str(r) for r in store.roots]


def _node_symbol_str(node_id):
    return spark_dsg.NodeSymbol(int(node_id)).str(True)


def subkeyframe_symbol(frame_id):
    """``nodeSymbol`` of a SubKeyframe node: ``s`` + the decimal store ``frame_id``.

    Frame ids embed the robot id (``robot << 48 | seq``), so the symbol is unique across
    robot stores and maps back with ``store.frame(int(symbol[1:]))``.
    """
    return f"s{int(frame_id)}"


def _path_or_none(p):
    return None if p is None else str(p)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def agent_to_dict(agent):
    attrs = agent.attributes
    d = {}
    d["nodeSymbol"] = agent.id.str(True)
    d["pos_x"] = attrs.position[0]
    d["pos_y"] = attrs.position[1]
    d["pos_z"] = attrs.position[2]

    # Orientation from the DSG attribute (optimized in place by backend PGO).
    rot = attrs.world_R_body
    d["rot_w"] = rot.w
    d["rot_x"] = rot.x
    d["rot_y"] = rot.y
    d["rot_z"] = rot.z

    # The python binding exposes `timestamp` as datetime.timedelta (chrono
    # caster) and DEFAULTS to timedelta(0) when unset; treat 0 as unset (the
    # image store's keyframe frame supplies the timestamp then).
    ts = getattr(attrs, "timestamp", None)
    ts_ns = (ts // datetime.timedelta(microseconds=1)) * 1000 if ts is not None else 0
    if ts_ns > 0:
        d["timestamp_ns"] = ts_ns

    return d


def subkeyframe_to_dict(frame, pose):
    """One SubKeyframe row from a store sub-keyframe ``Frame``.

    ``pose`` is the 4x4 corrected world_T_body (anchor pose composed with the stored
    anchor_T_body, or odometry when the anchor is unknown).
    """
    from heracles.pose_math import rot_to_quat

    q = rot_to_quat(pose[:3, :3])
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


def insert_agents_to_db(db, agents):
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


def _collect_agents(G, store=None):
    """Return agent dicts for every agent node of the DSG.

    Agents share layer id 2 with OBJECTS but live in a non-zero partition (robot
    prefix 'a' -> ord('a')). ``G.get_layer(DsgLayers.AGENTS)`` resolves to layer 2 /
    partition 0 == the OBJECTS layer, so it must NOT be used here. Enumerate the
    partitions instead (handles multi-robot prefixes), guarded by an
    AgentNodeAttributes type check.

    With a ``store``, each agent gets ``frame_id`` / ``rgb_path`` / ``depth_path`` from
    its keyframe frame (absolute paths), if the store has one; the frame timestamp
    fills in a missing attribute timestamp.
    """
    from hydra_image_store import FrameKind  # only reached with a store

    agents_layer = spark_dsg.DsgLayers.name_to_layer_id("AGENTS").layer  # == 2
    keyframes = {}
    if store is not None:
        for f in store.frames(kind=FrameKind.KEYFRAME):
            if f.anchor_node_id is not None:
                keyframes.setdefault(int(f.anchor_node_id), f)
    agents = []
    for key in G.layer_keys:
        if key.layer != agents_layer or key.partition == 0:
            continue
        for a in G.get_layer(key.layer, key.partition).nodes:
            if not isinstance(a.attributes, spark_dsg.AgentNodeAttributes):
                continue
            d = agent_to_dict(a)
            f = keyframes.get(int(a.id.value))
            if f is not None:
                d["frame_id"] = int(f.frame_id)
                d["rgb_path"] = _path_or_none(f.rgb_path)
                d["depth_path"] = _path_or_none(f.depth_path)
                # The store timestamp is exact; the DSG attribute only has float
                # (microsecond) precision through the python timedelta binding.
                d["timestamp_ns"] = int(f.timestamp_ns)
            agents.append(d)
    return agents


def add_agents_from_dsg(G, db, store=None):
    agents = _collect_agents(G, store)
    if agents:
        insert_agents_to_db(db, agents)
    return len(agents)


def _collect_subkeyframes(G, store):
    """SubKeyframe dicts, one per store frame of kind sub-keyframe.

    The pose is the corrected body pose: the anchor agent's pose in ``G`` composed with
    the stored anchor_T_body (``store.body_pose``), falling back to odometry when the
    anchor is not in the graph.
    """
    from hydra_image_store import FrameKind

    out = []
    for f in store.frames(kind=FrameKind.SUBKEYFRAME):
        pose, _ = store.body_pose(f, G)
        out.append(subkeyframe_to_dict(f, pose))
    return out


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


def add_subkeyframes_from_dsg(G, db, store):
    """Create ``(:SubKeyframe)-[:ANCHORED_TO]->(:Agent)`` from the store's sub-keyframes."""
    if store is None:
        return 0
    subframes = _collect_subkeyframes(G, store)
    if not subframes:
        return 0
    insert_subkeyframes_to_db(db, subframes)
    edges = [
        {"from": s["nodeSymbol"], "to": s["anchor_symbol"]}
        for s in subframes
        if "anchor_symbol" in s
    ]
    insert_edges(db, constants.ANCHORED_TO, constants.SUBKEYFRAMES, constants.AGENTS, edges)
    return len(subframes)


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def insert_observations_to_db(db, observations):
    # MERGE on nodeSymbol only, then SET the rest — so re-observing the same
    # (object, keyframe) updates in place instead of creating a duplicate node
    # whenever the 2D bbox / mask differs slightly between queries.
    return db.execute(
        f"""
    WITH $observations AS observations
    UNWIND observations AS obs
    MERGE (o:{constants.OBSERVATIONS} {{nodeSymbol: obs.nodeSymbol}})
    SET o.timestamp_ns = obs.timestamp_ns,
        o.mask_file = obs.mask_file,
        o.obs_id = coalesce(obs.obs_id, o.obs_id),
        o.frame_id = coalesce(obs.frame_id, o.frame_id),
        o.rgb_path = coalesce(obs.rgb_path, o.rgb_path),
        o.depth_path = coalesce(obs.depth_path, o.depth_path),
        o.mask_path = coalesce(obs.mask_path, o.mask_path),
        o.bbox_2d = coalesce(obs.bbox_2d, o.bbox_2d),
        o.kept = coalesce(obs.kept, o.kept),
        o.bbox_2d_min_x = obs.bbox_2d_min_x,
        o.bbox_2d_min_y = obs.bbox_2d_min_y,
        o.bbox_2d_max_x = obs.bbox_2d_max_x,
        o.bbox_2d_max_y = obs.bbox_2d_max_y,
        o.score = coalesce(obs.score, o.score),
        o.detector = coalesce(obs.detector, o.detector),
        o.mechanism = coalesce(obs.mechanism, o.mechanism),
        o.target_prompt = coalesce(obs.target_prompt, o.target_prompt),
        o.target_score = coalesce(obs.target_score, o.target_score),
        o.control_max_score = coalesce(obs.control_max_score, o.control_max_score),
        o.margin = coalesce(obs.margin, o.margin),
        o.control_scores_json = coalesce(obs.control_scores_json, o.control_scores_json),
        o.depth_valid_px = coalesce(obs.depth_valid_px, o.depth_valid_px),
        o.view_x = coalesce(obs.view_x, o.view_x),
        o.view_y = coalesce(obs.view_y, o.view_y),
        o.view_z = coalesce(obs.view_z, o.view_z)
    """,
        observations=observations,
    )


# ---------------------------------------------------------------------------
# Evidence provenance: Observation -> source frame, frame -> camera calibration
# ---------------------------------------------------------------------------

# Frame-bearing labels an Observation can be joined to by capture timestamp.
_FRAME_LABELS = (constants.AGENTS, constants.SUBKEYFRAMES, constants.TRAJECTORY_FRAMES)


def ensure_provenance_indexes(db):
    """Idempotent b-tree indexes backing the timestamp joins below."""
    for label in _FRAME_LABELS + (constants.OBSERVATIONS,):
        db.execute(
            f"CREATE INDEX {label.lower()}_timestamp_ns IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.timestamp_ns)"
        )
    db.execute(
        f"CREATE INDEX {constants.CAMERA_CALIBS.lower()}_calib_id IF NOT EXISTS "
        f"FOR (n:{constants.CAMERA_CALIBS}) ON (n.calib_id)"
    )


def link_observations_to_frames(db):
    """MERGE ``(:Observation)-[:OBSERVED_IN]->(frame)`` by exact capture timestamp.

    Observation symbols are ``<objectSymbol>_<timestamp_ns>`` where the timestamp
    is the source keyframe's — so an equality join on ``timestamp_ns`` against
    Agent / SubKeyframe / TrajectoryFrame recovers the missing provenance hop.
    Best-effort: frames without a matching timestamp simply get no edge.
    Returns the total number of edges present after the merge.
    """
    ensure_provenance_indexes(db)
    total = 0
    for label in _FRAME_LABELS:
        records, _, _ = db.execute(
            f"""
        MATCH (o:{constants.OBSERVATIONS}) WHERE o.timestamp_ns IS NOT NULL
        MATCH (f:{label} {{timestamp_ns: o.timestamp_ns}})
        MERGE (o)-[:{constants.OBSERVED_IN}]->(f)
        RETURN count(*) AS n
        """
        )
        total += records[0]["n"] if records else 0
    return total


def _calib_to_dict(calib):
    """Flat CameraCalib property dict from a store ``Calib`` (sensor row)."""
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
        "body_T_sensor": [float(v) for v in np.asarray(calib.body_T_sensor).reshape(-1)],
    }


def attach_camera_calibs(db, store):
    """Materialize ``(:CameraCalib)`` nodes from the store's sensors, with
    ``HAS_CALIB`` edges from every Agent (keyframe frame) and SubKeyframe.

    ``calib_id`` is the decimal store ``sensor_id``. Returns the number of
    frame->calib edges written.
    """
    from hydra_image_store import FrameKind

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
    agent_pairs, sub_pairs = [], []
    for f in store.frames(kind=[FrameKind.KEYFRAME, FrameKind.SUBKEYFRAME]):
        if f.anchor_node_id is None or int(f.sensor_id) not in known:
            continue
        cid = str(int(f.sensor_id))
        if f.kind == FrameKind.KEYFRAME:
            agent_pairs.append({"ns": _node_symbol_str(f.anchor_node_id), "calib_id": cid})
        else:
            sub_pairs.append({"ns": subkeyframe_symbol(f.frame_id), "calib_id": cid})
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
# Per-layer wrappers (know which spark_dsg layer to iterate)
# ---------------------------------------------------------------------------


def _observation_dicts(obj_symbol, observations):
    """Observation property dicts for one object node, one per store observation.

    ``nodeSymbol`` is ``<obj>_<timestamp_ns>``; when two observations of the same object
    share a timestamp the later one gets ``_<obs_id>`` appended so none is lost.
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
            "score": float(ob.score),
            "kept": bool(ob.kept),
        }
        if ob.bbox2d is not None:
            x0, y0, x1, y1 = (int(v) for v in ob.bbox2d)
            d["bbox_2d"] = [x0, y0, x1, y1]
            d["bbox_2d_min_x"], d["bbox_2d_min_y"] = x0, y0
            d["bbox_2d_max_x"], d["bbox_2d_max_y"] = x1, y1
        out.append(d)
    return out


def add_objects_from_dsg(G, db, store=None, object_labelspace=None):
    """Insert Object nodes (and their Observation nodes) into Neo4j.

    Parameters
    ----------
    G : spark_dsg.DynamicSceneGraph
    db : Neo4jWrapper
    store : hydra_image_store.ImageStore or None
        When given, every object gets ``(:Object)-[:HAS_OBSERVATION]->(:Observation)``
        for each store observation of the node (merged children included).
    object_labelspace : dict or None
        ``{str(int_id): class_name}`` mapping used by node_to_dict().
    """
    if object_labelspace is None:
        object_labelspace = {}
    dsg_objects = list(G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes)
    nodes = [node_to_dict(o, object_labelspace=object_labelspace) for o in dsg_objects]

    # Evidence-admission gate (G1): objects arriving here come from a COMMITTED
    # scene graph -- the prior map -- so they are trusted priors, stamped
    # explicitly rather than left to a reader's default. Best-effort and
    # additive, like the other provenance hooks: a downstream reader that
    # doesn't know about admission is unaffected by two extra properties.
    #
    # Passed as DEFAULTS, not as node properties. Stamping the dicts here made
    # the properties unconditional (setdefault on a dict node_to_dict has just
    # built always fires), and insert_nodes_to_db's "MERGE ... SET n += node"
    # then overwrote whatever the node already carried: re-ingesting a map
    # whose object had been graded "rejected" reset it to "trusted_prior" while
    # leaving its admission_reason attached. A grading decision outranks an
    # ingest stamp, so it must only be filled in where it is missing.
    admission_defaults = {
        constants.ADMISSION_STATUS: constants.TRUSTED_PRIOR,
        constants.ADMISSION_POLICY_VERSION: constants.PRIOR_MAP_POLICY_VERSION,
    }

    insert_nodes_to_db(
        db, constants.OBJECTS, nodes, default_properties=admission_defaults
    )

    if store is not None:
        observations = []
        edges = []
        for o in dsg_objects:
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
    nodes = [
        node_to_dict(p) for p in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes
    ]
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


def add_rooms_from_dsg(G, db, room_labelspace=None):
    if room_labelspace is None:
        room_labelspace = {}
    nodes = [
        node_to_dict(r, room_labelspace=room_labelspace)
        for r in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    ]
    insert_nodes_to_db(db, constants.ROOMS, nodes)


def add_buildings_from_dsg(G, db):
    nodes = [
        node_to_dict(b) for b in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes
    ]
    insert_nodes_to_db(db, constants.BUILDINGS, nodes)


# ---------------------------------------------------------------------------
# Top-level load: spark_dsg → Neo4j
# ---------------------------------------------------------------------------


def spark_dsg_to_db(G, db, source_file_path=None, image_store=None, mesh_path=None):
    """Load all nodes and edges from a spark_dsg graph into Neo4j.

    Extracts labelspaces from DSG metadata (embedded ``"labelspaces"`` key)
    and passes them to per-layer functions for semantic label -> class name
    mapping.

    If ``source_file_path`` is provided, it is stored as a ``_GraphMetadata``
    node so that downstream tools (e.g., SGET) can locate the original file
    on disk for mesh data and other large assets not stored in Neo4j.

    ``image_store`` is one image-store directory, a list of them (one per robot) or an
    open ``hydra_image_store.ImageStore``. With a store, the ingest adds:
    Observation nodes (``Object-[:HAS_OBSERVATION]->Observation``), ``frame_id`` /
    ``rgb_path`` / ``depth_path`` on Agent nodes, SubKeyframe nodes
    (``-[:ANCHORED_TO]->Agent``), CameraCalib nodes (``HAS_CALIB``) and
    ``_GraphMetadata {key: 'image_store'}.image_store_paths``. A legacy run directory
    without a store raises :class:`ImageStoreError`.
    """
    from .utils import extract_labelspaces_from_dsg

    store = open_image_store(image_store)
    object_ls, room_ls = extract_labelspaces_from_dsg(G)

    add_agents_from_dsg(G, db, store)
    add_objects_from_dsg(G, db, store, object_labelspace=object_ls)
    add_places_from_dsg(G, db)
    add_mesh_places_from_dsg(G, db, object_labelspace=object_ls)
    add_rooms_from_dsg(G, db, room_labelspace=room_ls)
    add_buildings_from_dsg(G, db)
    if store is not None:
        add_subkeyframes_from_dsg(G, db, store)
    add_edges_from_dsg(G, db)

    # Evidence provenance (additive, best-effort): Observation -> source-frame
    # edges by capture timestamp, and CameraCalib nodes from the store sensors so
    # visibility reasoning is possible from the DB.
    try:
        if store is not None:
            attach_camera_calibs(db, store)
        link_observations_to_frames(db)
    except Exception:
        logger.exception("evidence-provenance materialization failed (non-fatal)")

    if store is not None:
        db.execute(
            "MERGE (m:_GraphMetadata {key: 'image_store'}) SET m.image_store_paths = $paths",
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
            "MERGE (m:_Labelspace {layer: 'room'}) "
            "SET m.ids = $ids, m.names = $names",
            ids=ids,
            names=names,
        )

    if source_file_path is not None:
        import os

        abs_path = os.path.abspath(source_file_path)
        db.execute(
            "MERGE (m:_GraphMetadata {key: 'source'}) SET m.file_path = $path",
            path=abs_path,
        )

    if mesh_path is not None:
        store_mesh_path(db, mesh_path)


# ---------------------------------------------------------------------------
# Mesh path (geometry lives on disk, not in Neo4j — store a pointer to the .ply)
# ---------------------------------------------------------------------------


def store_mesh_path(db, mesh_path, mesh_format=None):
    """Record the scene mesh file location so the renderer can load it directly
    (e.g. ``trimesh.load(path)``) without going through spark_dsg.

    Stored on a ``_GraphMetadata {key:'mesh'}`` node as ``file_path``/``format``.
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
    """Return ``(file_path, format)`` for the scene mesh, or ``(None, None)``."""
    records, _, _ = db.execute(
        "MATCH (m:_GraphMetadata {key: 'mesh'}) RETURN m.file_path AS p, m.format AS f"
    )
    if records:
        return records[0]["p"], records[0]["f"]
    return None, None


# ---------------------------------------------------------------------------
# Incremental writes (write only a changeset — for the online ingester and
# single mutations; the full-graph spark_dsg_to_db re-writes everything).
# ---------------------------------------------------------------------------

# (layer, partition) -> heracles node label.
_LAYER_PARTITION_TO_LABEL = {
    (2, 0): constants.OBJECTS,
    (3, 0): constants.PLACES,
    (3, 1): constants.MESH_PLACES,
    (4, 0): constants.ROOMS,
    (5, 0): constants.BUILDINGS,
}


def update_db_from_spark_dsg(
    G, db, node_ids, object_labelspace=None, room_labelspace=None, bump=True
):
    """Write only ``node_ids`` (symbol strings) from G into Neo4j (idempotent MERGE).

    Incremental counterpart to :func:`spark_dsg_to_db`. Groups the ids by layer
    label and bulk-MERGEs each group. Ids not present in G are skipped (handle
    deletions via :func:`remove_nodes`). Returns the new ``_State.version`` if
    ``bump`` (so callers can fan out a GraphEvent).
    """
    if object_labelspace is None or room_labelspace is None:
        try:
            from .utils import extract_labelspaces_from_dsg
            o_ls, r_ls = extract_labelspaces_from_dsg(G)
            object_labelspace = object_labelspace or o_ls
            room_labelspace = room_labelspace or r_ls
        except Exception:
            pass

    by_label: dict[str, list] = {}
    for nid in node_ids:
        try:
            node = G.get_node(str_to_ns_value(nid))
        except Exception:
            continue
        if node is None:
            continue
        key = (int(node.layer.layer), int(node.layer.partition))
        label = _LAYER_PARTITION_TO_LABEL.get(key)
        if label is None:
            continue
        d = node_to_dict(
            node, object_labelspace=object_labelspace, room_labelspace=room_labelspace
        )
        by_label.setdefault(label, []).append(d)

    for label, dicts in by_label.items():
        insert_nodes_to_db(db, label, dicts)

    return bump_version(db) if bump else read_version(db)


def remove_nodes(db, node_ids, bump=True):
    """Detach-delete ``node_ids`` (symbol strings). Returns new version if ``bump``."""
    ids = list(node_ids)
    if not ids:
        return read_version(db)
    db.execute(
        "UNWIND $ids AS ns MATCH (n {nodeSymbol: ns}) DETACH DELETE n",
        ids=ids,
    )
    return bump_version(db) if bump else read_version(db)


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
    # MERGE (not CREATE) so re-inserting the same edge is idempotent — CREATE made
    # re-runs (e.g. re-querying SAM3) accumulate duplicate parallel relationships.
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
            attrs.states = [spark_dsg.TraversabilityState(int(s)) for s in record["states"]]
        if "min_radius" in record:
            attrs.min_radius = float(record["min_radius"])
        if "max_radius" in record:
            attrs.max_radius = float(record["max_radius"])

    # Place2dNodeAttributes boundary (stored as Point3D list in Neo4j).
    if hasattr(attrs, "boundary") and isinstance(getattr(attrs, "boundary", None), list):
        if "boundary" in record and isinstance(record["boundary"], list):
            import numpy as np

            attrs.boundary = [
                np.array([float(pt[0]), float(pt[1]), float(pt[2])])
                for pt in record["boundary"]
            ]

    return attrs




# ---------------------------------------------------------------------------
# Embedding + versioning helpers (T3)
# ---------------------------------------------------------------------------


def create_vector_indexes(db, dim: int, model_name: str) -> None:
    """Create Neo4j vector indices on Object/Observation/TrajectoryFrame.embedding properties.

    Idempotent: uses ``CREATE VECTOR INDEX ... IF NOT EXISTS``.

    The index name is derived from the node label: ``object_embedding``,
    ``observation_embedding``, ``trajectoryframe_embedding``.
    Cosine similarity, dimension = ``dim``.
    """
    for label, index_name in [
        ("Object", "object_embedding"),
        ("Observation", "observation_embedding"),
        ("TrajectoryFrame", "trajectoryframe_embedding"),
    ]:
        cypher = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.embedding) "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}"
        )
        db.execute(cypher, dim=dim)


def create_epistemic_indexes(db) -> None:
    """Constraints + lookup indexes for the epistemic layer (SearchEpisode /
    Assertion nodes written by ``agentic_navigation.evidence.store``).

    Idempotent. These are the schema's first uniqueness constraints:
    ``episode_id`` / ``assertion_id`` are stable external identities that a
    JSONL re-import (wipe survival) must MERGE onto, never duplicate. Note
    ``initialize_db`` wipes nodes but leaves constraints in place, so calling
    this once per database outlives ingest cycles.
    """
    db.execute(
        f"CREATE CONSTRAINT search_episode_id_unique IF NOT EXISTS "
        f"FOR (n:{constants.SEARCH_EPISODES}) REQUIRE n.episode_id IS UNIQUE"
    )
    db.execute(
        f"CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS "
        f"FOR (n:{constants.ASSERTIONS}) REQUIRE n.assertion_id IS UNIQUE"
    )
    db.execute(
        f"CREATE INDEX search_episode_query IF NOT EXISTS "
        f"FOR (n:{constants.SEARCH_EPISODES}) ON (n.predicate, n.args_json)"
    )
    db.execute(
        f"CREATE INDEX assertion_query IF NOT EXISTS "
        f"FOR (n:{constants.ASSERTIONS}) ON (n.predicate, n.args_json)"
    )


def set_node_embedding(db, node_symbol: str, vec, model_name: str) -> None:
    """Set the ``embedding``, ``embedding_dim``, ``embedding_model`` properties on a node
    identified by ``nodeSymbol``. ``vec`` is a list[float] or 1D numpy array.
    """
    vec_list = list(map(float, vec))
    db.execute(
        "MATCH (n {nodeSymbol: $ns}) SET n.embedding = $v, n.embedding_dim = $d, n.embedding_model = $m",
        ns=node_symbol,
        v=vec_list,
        d=len(vec_list),
        m=model_name,
    )


def set_observation_embedding(db, observation_symbol: str, vec, model_name: str) -> None:
    """Same as set_node_embedding but scoped to Observation nodes by their ``nodeSymbol``.

    Observations are identified by ``nodeSymbol`` (see ``insert_observations_to_db``).
    """
    set_node_embedding(db, observation_symbol, vec, model_name)


def query_similar_nodes(
    db, label: str, query_vec, k: int = 10, filter_cypher: str | None = None
) -> list[tuple[str, float]]:
    """Return list of (node_symbol, score) tuples for the top-k nodes most similar to query_vec.

    ``label`` is one of ``"Object"`` | ``"Observation"`` | ``"TrajectoryFrame"``.
    ``filter_cypher`` is an optional Cypher predicate fragment, e.g. ``"n.class = 'chair'"``,
    inserted after the vector search as a ``WHERE`` clause. WARNING: this fragment is
    interpolated directly into the query; callers must ensure it is not derived from
    untrusted input (no user-controlled cypher).
    """
    index_name = {
        "Object": "object_embedding",
        "Observation": "observation_embedding",
        "TrajectoryFrame": "trajectoryframe_embedding",
    }[label]
    where = f"WHERE {filter_cypher} " if filter_cypher else ""
    cypher = (
        f"CALL db.index.vector.queryNodes('{index_name}', $k, $v) YIELD node, score "
        f"{where}"
        f"RETURN node.nodeSymbol AS ns, score ORDER BY score DESC"
    )
    vec_list = list(map(float, query_vec))
    records, _, _ = db.execute(cypher, k=k, v=vec_list)
    return [(r["ns"], r["score"]) for r in records]


def insert_trajectory_frames_to_db(db, frames: list) -> int:
    """Insert TrajectoryFrame nodes. Each frame dict has keys:
      nodeSymbol, timestamp_ns, path, pos_x, pos_y, pos_z,
      pose_qw, pose_qx, pose_qy, pose_qz, place_id (str|None),
      embedding (list[float]|None), embedding_dim (int|None), embedding_model (str|None).

    Uses MERGE on nodeSymbol so the operation is idempotent. Sets ``center = point(...)``.
    Returns the number of rows actually written.
    """
    if not frames:
        return 0
    cypher = """
    UNWIND $frames AS f
    MERGE (t:TrajectoryFrame {nodeSymbol: f.nodeSymbol})
    SET t.timestamp_ns = f.timestamp_ns,
        t.path = f.path,
        t.pos_x = f.pos_x, t.pos_y = f.pos_y, t.pos_z = f.pos_z,
        t.pose_qw = f.pose_qw, t.pose_qx = f.pose_qx, t.pose_qy = f.pose_qy, t.pose_qz = f.pose_qz,
        t.place_id = f.place_id,
        t.center = point({x: f.pos_x, y: f.pos_y, z: f.pos_z}),
        t.embedding = f.embedding,
        t.embedding_dim = f.embedding_dim,
        t.embedding_model = f.embedding_model
    RETURN count(t) AS n
    """
    records, _, _ = db.execute(cypher, frames=frames)
    return records[0]["n"] if records else 0


def insert_frame_edges(
    db,
    frame_to_place_edges: list,
    frame_to_object_edges: list,
) -> None:
    """Create OBSERVED_AT edges from TrajectoryFrame to Place and DEPICTS edges from
    TrajectoryFrame to Object.

    Each tuple is (frame_symbol, target_symbol). MERGE-based so idempotent.
    """
    if frame_to_place_edges:
        db.execute(
            "UNWIND $pairs AS p "
            "MATCH (f:TrajectoryFrame {nodeSymbol: p[0]}), (place {nodeSymbol: p[1]}) "
            "MERGE (f)-[:OBSERVED_AT]->(place)",
            pairs=[list(t) for t in frame_to_place_edges],
        )
    if frame_to_object_edges:
        db.execute(
            "UNWIND $pairs AS p "
            "MATCH (f:TrajectoryFrame {nodeSymbol: p[0]}), (obj:Object {nodeSymbol: p[1]}) "
            "MERGE (f)-[:DEPICTS]->(obj)",
            pairs=[list(t) for t in frame_to_object_edges],
        )


def bump_version(db) -> int:
    """Atomically increment ``_State {key:'global'}.version`` and return the new value.

    The first call creates the node with ``version = 1`` and returns ``1``.
    Subsequent calls increment and return the new value. ``read_version`` returns
    ``0`` when the node does not yet exist (i.e. before the first bump).
    """
    records, _, _ = db.execute(
        "MERGE (s:_State {key: 'global'}) "
        "ON CREATE SET s.version = 1 "
        "ON MATCH SET s.version = s.version + 1 "
        "RETURN s.version AS v"
    )
    return records[0]["v"] if records else 0


def read_version(db) -> int:
    """Return the current ``_State.version`` (0 if the node doesn't exist yet)."""
    records, _, _ = db.execute(
        "MATCH (s:_State {key: 'global'}) RETURN s.version AS v"
    )
    return records[0]["v"] if records else 0


def record_change(db, dirty_layers=None, affected_ids=None) -> int:
    """Bump the version AND record which layers/ids changed on ``_State``.

    Lets a cross-process reader (e.g. viz_agent's SyncSupervisor) re-render only
    the dirty layers instead of the whole graph. ``dirty_layers`` is a list of
    ``"layer:partition"`` strings. Returns the new version.
    """
    records, _, _ = db.execute(
        "MERGE (s:_State {key: 'global'}) "
        "ON CREATE SET s.version = 1 "
        "ON MATCH SET s.version = s.version + 1 "
        "SET s.dirty_layers = $dl, s.affected_ids = $ai "
        "RETURN s.version AS v",
        dl=list(dirty_layers or []),
        ai=list(affected_ids or []),
    )
    return records[0]["v"] if records else 0


_MAP_STATE = "_MapState"


def get_map_version(db) -> int:
    """``_MapState.version``: the counter the heracles publisher polls (0 before any bump)."""
    records, _, _ = db.execute(f"MATCH (m:{_MAP_STATE}) RETURN m.version AS v")
    return int(records[0]["v"]) if records and records[0]["v"] is not None else 0


def bump_map_version(db) -> int:
    """Increment ``_MapState.version`` (heracles_ros publisher republishes within its
    ``version_poll_period_s``) and return the new value."""
    records, _, _ = db.execute(
        f"MERGE (m:{_MAP_STATE}) "
        "SET m.version = coalesce(m.version, 0) + 1 "
        "RETURN m.version AS v"
    )
    return int(records[0]["v"])


def notify_map_change(db, dirty_layers=None, affected_ids=None) -> tuple:
    """Tell every reader that the map changed: bump BOTH version counters at once.

    * ``_State {key:'global'}.version`` (+ ``dirty_layers`` / ``affected_ids``, see
      :func:`record_change`) -- watched by the web console and viz agents.
    * ``_MapState.version`` (see :func:`bump_map_version`) -- polled by the heracles
      publisher, which republishes Neo4j -> DSG as soon as it changes (planners wait on it).

    Every Neo4j map writer should call this after its write. One statement, so both
    counters move together. Returns ``(state_version, map_version)``.
    """
    records, _, _ = db.execute(
        "MERGE (s:_State {key: 'global'}) "
        "ON CREATE SET s.version = 1 "
        "ON MATCH SET s.version = s.version + 1 "
        "SET s.dirty_layers = $dl, s.affected_ids = $ai "
        f"WITH s MERGE (m:{_MAP_STATE}) "
        "SET m.version = coalesce(m.version, 0) + 1 "
        "RETURN s.version AS sv, m.version AS mv",
        dl=list(dirty_layers or []),
        ai=list(affected_ids or []),
    )
    return int(records[0]["sv"]), int(records[0]["mv"])


def read_change(db):
    """Return ``(version, dirty_layers, affected_ids)`` from ``_State``.

    ``dirty_layers`` is a list of ``"layer:partition"`` strings (empty if the
    last write didn't record any — treat that as "re-render everything").
    """
    records, _, _ = db.execute(
        "MATCH (s:_State {key: 'global'}) "
        "RETURN s.version AS v, s.dirty_layers AS dl, s.affected_ids AS ai"
    )
    if not records:
        return 0, [], []
    r = records[0]
    return r["v"] or 0, list(r["dl"] or []), list(r["ai"] or [])


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


def db_to_spark_dsg(db):
    """Reconstruct a spark_dsg DynamicSceneGraph from Neo4j.

    Reads labelspaces from ``_Labelspace`` nodes in the database.
    Uses ``attr_type`` stored on each node to create the correct attribute
    class.
    """
    label_to_semantic_id = _read_labelspace_from_db(db, "object")
    room_label_to_semantic_id = _read_labelspace_from_db(db, "room")

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