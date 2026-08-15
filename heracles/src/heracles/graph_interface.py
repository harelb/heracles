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

import glob
import json
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
        ls = (
            room_labelspace
            if isinstance(attrs, spark_dsg.RoomNodeAttributes)
            else object_labelspace
        )
        label_key = str(attrs.semantic_label)
        if ls and label_key in ls:
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
    if hasattr(attrs, "image_folder"):
        d["image_folder"] = attrs.image_folder
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
# Agent helpers
# ---------------------------------------------------------------------------


def _timestamp_from_agent_folder(image_folder):
    """Parse the keyframe timestamp from an ``.../agent_<ts>`` file prefix.

    Returns int nanoseconds or None. The suffix is the authoritative capture
    timestamp baked in by hydra's AgentImageExtractor, so it can stand in for
    graphs whose agent attributes carry no timestamp.
    """
    if not image_folder:
        return None
    tail = os.path.basename(str(image_folder)).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def agent_to_dict(agent):
    attrs = agent.attributes
    d = {}
    d["nodeSymbol"] = agent.id.str(True)
    d["pos_x"] = attrs.position[0]
    d["pos_y"] = attrs.position[1]
    d["pos_z"] = attrs.position[2]

    # Orientation from the DSG attribute (optimized in place by backend PGO).
    # The baked _meta.json world_T_body is a stale snapshot and is not used.
    rot = attrs.world_R_body
    d["rot_w"] = rot.w
    d["rot_x"] = rot.x
    d["rot_y"] = rot.y
    d["rot_z"] = rot.z

    if hasattr(attrs, "image_folder"):
        d["image_folder"] = attrs.image_folder

    # The python binding exposes `timestamp` as datetime.timedelta (chrono
    # caster) and DEFAULTS to timedelta(0) when unset — treat 0 as unset and
    # fall back to the agent_<ts> folder suffix so Observation nodes can be
    # joined to their source keyframe in Cypher.
    ts = getattr(attrs, "timestamp", None)
    ts_ns = round(ts.total_seconds() * 1e9) if ts is not None else 0
    if ts_ns > 0:
        d["timestamp_ns"] = ts_ns
    else:
        parsed = _timestamp_from_agent_folder(d.get("image_folder"))
        if parsed is not None:
            d["timestamp_ns"] = parsed

    return d


def subkeyframe_to_dict(subframe, anchor_pose):
    """anchor_pose = (world_t_anchor (3,), world_R_anchor (w,x,y,z))."""
    from heracles.pose_math import compose_pose

    attrs = subframe.attributes
    anchor_R = attrs.anchor_R_subframe
    world_t, world_R = compose_pose(
        world_t_anchor=anchor_pose[0],
        world_R_anchor=anchor_pose[1],
        anchor_t_sub=np.array(attrs.anchor_t_subframe),
        anchor_R_sub=(anchor_R.w, anchor_R.x, anchor_R.y, anchor_R.z),
    )
    d = {
        "nodeSymbol": subframe.id.str(True),
        "anchor_symbol": spark_dsg.NodeSymbol(attrs.anchor_node_id).str(True),
        "pos_x": float(world_t[0]),
        "pos_y": float(world_t[1]),
        "pos_z": float(world_t[2]),
        "rot_w": float(world_R[0]),
        "rot_x": float(world_R[1]),
        "rot_y": float(world_R[2]),
        "rot_z": float(world_R[3]),
        "image_folder": attrs.image_folder,
        # NOTE: the python binding exposes `timestamp` as datetime.timedelta
        # (chrono caster), NOT an int. Convert to integer nanoseconds:
        "timestamp_ns": round(attrs.timestamp.total_seconds() * 1e9),
    }
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
        n.image_folder = agent.image_folder,
        n.timestamp_ns = coalesce(agent.timestamp_ns, n.timestamp_ns)
    """,
        agents=agents,
    )


def _collect_keyframe_agents(G):
    """Return agent dicts (with a non-empty image_folder) from every keyframe
    partition of the agents layer.

    Agents share layer id 2 with OBJECTS but live in a non-zero partition (robot
    prefix 'a' -> ord('a')). ``G.get_layer(DsgLayers.AGENTS)`` resolves to layer 2 /
    partition 0 == the OBJECTS layer, so it must NOT be used here — iterating it
    mislabels Objects as Agents. Enumerate the partitions instead and read agents
    from every non-zero partition (handles multi-robot prefixes), guarded by an
    AgentNodeAttributes type check. Agents with an empty image_folder (no extracted
    keyframe) are skipped.
    """
    agents_layer = spark_dsg.DsgLayers.name_to_layer_id("AGENTS").layer  # == 2
    agents = []
    for key in G.layer_keys:
        if key.layer != agents_layer or key.partition == 0:
            continue
        for a in G.get_layer(key.layer, key.partition).nodes:
            if not isinstance(a.attributes, spark_dsg.AgentNodeAttributes):
                continue
            d = agent_to_dict(a)
            # Agent image_folder is already the absolute on-disk agents/ prefix;
            # do NOT rebase it onto image_folder_root (the object-crop dir).
            if d.get("image_folder"):
                agents.append(d)
    return agents


def add_agents_from_dsg(G, image_folder_root, db):
    agents = _collect_keyframe_agents(G)
    if agents:
        insert_agents_to_db(db, agents)


def _collect_subkeyframes(G):
    """Sub-keyframes share layer 2 with agents/objects but use the 's' prefix
    partition. Filter by attribute type; compose world pose from each node's
    anchor agent node (optimized pose = source of truth)."""
    sub_layer = spark_dsg.DsgLayers.name_to_layer_id("AGENTS").layer  # == 2
    out = []
    for key in G.layer_keys:
        if key.layer != sub_layer or key.partition == 0:
            continue
        for n in G.get_layer(key.layer, key.partition).nodes:
            if not isinstance(n.attributes, spark_dsg.SubKeyframeNodeAttributes):
                continue
            anchor_id = n.attributes.anchor_node_id
            if not G.has_node(anchor_id):
                continue  # anchor pruned; skip (orphan)
            a = G.get_node(anchor_id).attributes
            aR = a.world_R_body
            anchor_pose = (np.array(a.position), (aR.w, aR.x, aR.y, aR.z))
            out.append(subkeyframe_to_dict(n, anchor_pose))
    return out


def insert_subkeyframes_to_db(db, subframes):
    return db.execute(
        f"""
    WITH $subframes AS subframes
    UNWIND subframes AS s
    MERGE (n:{constants.SUBKEYFRAMES} {{nodeSymbol: s.nodeSymbol}})
    SET n.center = point({{x: s.pos_x, y: s.pos_y, z: s.pos_z}}),
        n.rot_w = s.rot_w, n.rot_x = s.rot_x, n.rot_y = s.rot_y, n.rot_z = s.rot_z,
        n.image_folder = s.image_folder,
        n.timestamp_ns = s.timestamp_ns
    """,
        subframes=subframes,
    )


def add_subkeyframes_from_dsg(G, db):
    subframes = _collect_subkeyframes(G)
    if not subframes:
        return 0
    insert_subkeyframes_to_db(db, subframes)
    edges = [{"from": s["nodeSymbol"], "to": s["anchor_symbol"]} for s in subframes]
    insert_edges(db, constants.ANCHORED_TO, constants.SUBKEYFRAMES, constants.AGENTS, edges)
    return len(subframes)


def merge_agent_image_folders(G, db):
    """Upsert ``image_folder`` onto Agent nodes by nodeSymbol from G (no rebasing).

    Hydra's backend pose-graph optimization drops ``image_folder`` on roughly half
    the agent (keyframe) nodes; the sibling *frontend* DSG retains all of them. This
    merges the frontend folders in by symbol so every keyframe is queryable, WITHOUT
    disturbing the optimized ``center`` already written for existing backend nodes
    (only newly-created frontend-only nodes get their center set). Returns the number
    of agent rows upserted.
    """
    agents = _collect_keyframe_agents(G)
    if not agents:
        return 0
    db.execute(
        f"""
    WITH $agents AS agents
    UNWIND agents AS agent
    MERGE (n:{constants.AGENTS} {{nodeSymbol: agent.nodeSymbol}})
    ON CREATE SET n.center = point({{x: agent.pos_x, y: agent.pos_y, z: agent.pos_z}}),
                  n.rot_w = agent.rot_w, n.rot_x = agent.rot_x,
                  n.rot_y = agent.rot_y, n.rot_z = agent.rot_z,
                  n.image_folder = agent.image_folder,
                  n.timestamp_ns = agent.timestamp_ns
    ON MATCH SET n.image_folder = agent.image_folder,
                 n.timestamp_ns = coalesce(n.timestamp_ns, agent.timestamp_ns)
    """,
        agents=agents,
    )
    return len(agents)


def _merge_agent_folders_from_frontend_sibling(db, source_file_path):
    """Best-effort: if loading a hydra *backend* DSG, fill agent image_folders the
    backend dropped from the sibling ``../frontend/dsg.json`` (see
    :func:`merge_agent_image_folders`). No-op if the path is not a backend DSG or the
    frontend sibling is absent.
    """
    if not source_file_path:
        return 0
    src = os.path.abspath(source_file_path)
    parent = os.path.dirname(src)
    if os.path.basename(parent) != "backend":
        return 0
    frontend = os.path.join(os.path.dirname(parent), "frontend", "dsg.json")
    if not os.path.exists(frontend):
        return 0
    try:
        G_fe = spark_dsg.DynamicSceneGraph.load(frontend)
    except Exception:
        logger.exception("could not load frontend DSG %s for agent merge", frontend)
        return 0
    n = merge_agent_image_folders(G_fe, db)
    logger.info("merged %d agent image_folders from frontend DSG %s", n, frontend)
    return n


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
        o.bbox_2d_min_x = obs.bbox_2d_min_x,
        o.bbox_2d_min_y = obs.bbox_2d_min_y,
        o.bbox_2d_max_x = obs.bbox_2d_max_x,
        o.bbox_2d_max_y = obs.bbox_2d_max_y,
        o.score = coalesce(obs.score, o.score),
        o.detector = coalesce(obs.detector, o.detector),
        o.mechanism = coalesce(obs.mechanism, o.mechanism)
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


def backfill_agent_timestamps(db):
    """Parse ``timestamp_ns`` from the ``agent_<ts>`` image_folder suffix for
    Agent nodes that lack it (pre-existing databases). Returns rows updated."""
    records, _, _ = db.execute(
        f"""
    MATCH (n:{constants.AGENTS})
    WHERE n.timestamp_ns IS NULL AND n.image_folder IS NOT NULL
    RETURN n.nodeSymbol AS ns, n.image_folder AS folder
    """
    )
    updates = []
    for r in records:
        ts = _timestamp_from_agent_folder(r["folder"])
        if ts is not None:
            updates.append({"ns": r["ns"], "ts": ts})
    if updates:
        db.execute(
            f"""
        UNWIND $updates AS u
        MATCH (n:{constants.AGENTS} {{nodeSymbol: u.ns}})
        SET n.timestamp_ns = u.ts
        """,
            updates=updates,
        )
    return len(updates)


def _load_calib_dict(calib_path):
    """Read a hydra ``camera_calib.json`` into a flat CameraCalib property dict.

    ``calib_id`` is a content hash so identical calibrations (multi-run maps)
    collapse onto one node.
    """
    import hashlib

    with open(calib_path, "r") as f:
        data = json.load(f)
    body_T_sensor = list(np.asarray(data["body_T_sensor"], dtype=float).reshape(-1))
    d = {
        "fx": float(data["fx"]),
        "fy": float(data["fy"]),
        "cx": float(data["cx"]),
        "cy": float(data["cy"]),
        "width": int(data["width"]),
        "height": int(data["height"]),
        "depth_scale": float(data.get("depth_scale", 1e-3)),
        "body_T_sensor": body_T_sensor,
    }
    canonical = json.dumps(d, sort_keys=True)
    d["calib_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return d


def attach_camera_calibs(db):
    """Materialize ``(:CameraCalib)`` nodes + ``HAS_CALIB`` edges from on-disk
    ``camera_calib.json`` files.

    Agent/SubKeyframe ``image_folder`` values are file *prefixes* and
    TrajectoryFrame ``path`` values are RGB files; the calibration sits once in
    each parent directory. Missing/unreadable files are skipped (best-effort) —
    frustum/visibility reasoning is only possible for frames that get an edge.
    Returns the number of frame->calib edges written.
    """
    ensure_provenance_indexes(db)
    label_prop = [
        (constants.AGENTS, "image_folder"),
        (constants.SUBKEYFRAMES, "image_folder"),
        (constants.TRAJECTORY_FRAMES, "path"),
    ]
    calib_by_dir = {}
    pairs_by_label = {}
    for label, prop in label_prop:
        records, _, _ = db.execute(
            f"MATCH (n:{label}) WHERE n.{prop} IS NOT NULL "
            f"RETURN n.nodeSymbol AS ns, n.{prop} AS p"
        )
        pairs = []
        for r in records:
            parent = os.path.dirname(str(r["p"]))
            if parent not in calib_by_dir:
                calib_path = os.path.join(parent, "camera_calib.json")
                calib = None
                if os.path.exists(calib_path):
                    try:
                        calib = _load_calib_dict(calib_path)
                    except Exception:
                        logger.exception("unreadable camera calib %s", calib_path)
                calib_by_dir[parent] = calib
            calib = calib_by_dir[parent]
            if calib is not None:
                pairs.append({"ns": r["ns"], "calib_id": calib["calib_id"]})
        if pairs:
            pairs_by_label[label] = pairs

    # Calib nodes must exist before the MATCH-based edge merge below.
    calibs = {c["calib_id"]: c for c in calib_by_dir.values() if c is not None}
    if calibs:
        db.execute(
            f"""
        UNWIND $calibs AS c
        MERGE (n:{constants.CAMERA_CALIBS} {{calib_id: c.calib_id}})
        SET n.fx = c.fx, n.fy = c.fy, n.cx = c.cx, n.cy = c.cy,
            n.width = c.width, n.height = c.height,
            n.depth_scale = c.depth_scale, n.body_T_sensor = c.body_T_sensor
        """,
            calibs=list(calibs.values()),
        )

    n_edges = 0
    for label, pairs in pairs_by_label.items():
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


def add_objects_from_dsg(G, image_folder_root, db, object_labelspace=None):
    """Insert Object nodes (and their Observation nodes) into Neo4j.

    Parameters
    ----------
    G : spark_dsg.DynamicSceneGraph
    image_folder_root : str or None
        Root directory containing per-object image folders.  When provided,
        each object's ``image_folder`` attribute is rebased onto this root
        and ``*_meta.json`` files are read to create Observation nodes.
    db : Neo4jWrapper
    object_labelspace : dict or None
        ``{str(int_id): class_name}`` mapping used by node_to_dict().
    """
    if object_labelspace is None:
        object_labelspace = {}
    nodes = [
        node_to_dict(o, object_labelspace=object_labelspace)
        for o in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
    ]

    # Rebase image_folder paths onto image_folder_root if provided.
    if image_folder_root:
        for d in nodes:
            if d.get("image_folder"):
                d["image_folder"] = os.path.join(
                    image_folder_root, os.path.basename(d["image_folder"])
                )

    insert_nodes_to_db(db, constants.OBJECTS, nodes)

    # Process observations from per-object image folders.
    if image_folder_root:
        observations = []
        object_observation_edges = []

        for obj in nodes:
            image_folder = obj.get("image_folder")
            if not image_folder:
                continue

            meta_files = glob.glob(os.path.join(image_folder, "*_meta.json"))
            for meta_file in meta_files:
                try:
                    with open(meta_file, "r") as f:
                        data = json.load(f)

                    timestamp_ns = data.get("timestamp_ns")
                    if timestamp_ns is None:
                        continue

                    obs_symbol = f"{obj['nodeSymbol']}_{timestamp_ns}"
                    obs_dict = {
                        "nodeSymbol": obs_symbol,
                        "timestamp_ns": timestamp_ns,
                        "mask_file": data.get("mask_file", ""),
                    }

                    if "bbox_2d" in data:
                        bbox_2d = data["bbox_2d"]
                        obs_dict["bbox_2d_min_x"] = bbox_2d.get("min_x")
                        obs_dict["bbox_2d_min_y"] = bbox_2d.get("min_y")
                        obs_dict["bbox_2d_max_x"] = bbox_2d.get("max_x")
                        obs_dict["bbox_2d_max_y"] = bbox_2d.get("max_y")

                    observations.append(obs_dict)
                    object_observation_edges.append(
                        {"from": obj["nodeSymbol"], "to": obs_symbol}
                    )
                except Exception as e:
                    print(f"Failed to parse observation file {meta_file}: {e}")

        if observations:
            insert_observations_to_db(db, observations)
            insert_edges(
                db,
                constants.HAS_OBSERVATION,
                constants.OBJECTS,
                constants.OBSERVATIONS,
                object_observation_edges,
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


def spark_dsg_to_db(G, db, source_file_path=None, image_folder_root=None, mesh_path=None):
    """Load all nodes and edges from a spark_dsg graph into Neo4j.

    Extracts labelspaces from DSG metadata (embedded ``"labelspaces"`` key)
    and passes them to per-layer functions for semantic label → class name
    mapping.

    If ``source_file_path`` is provided, it is stored as a ``_GraphMetadata``
    node so that downstream tools (e.g., SGET) can locate the original file
    on disk for mesh data and other large assets not stored in Neo4j.

    If ``image_folder_root`` is provided, per-object image folders are rebased
    onto this root and Observation nodes are created from ``*_meta.json`` files.
    """
    from .utils import extract_labelspaces_from_dsg

    object_ls, room_ls = extract_labelspaces_from_dsg(G)

    add_agents_from_dsg(G, image_folder_root, db)
    # Backend DSGs lose agent image_folder on ~half the keyframe nodes during
    # pose-graph optimization; recover them from the sibling frontend DSG so all
    # keyframes are queryable. Best-effort, keyed off the backend source path.
    _merge_agent_folders_from_frontend_sibling(db, source_file_path)
    add_objects_from_dsg(G, image_folder_root, db, object_labelspace=object_ls)
    add_places_from_dsg(G, db)
    add_mesh_places_from_dsg(G, db, object_labelspace=object_ls)
    add_rooms_from_dsg(G, db, room_labelspace=room_ls)
    add_buildings_from_dsg(G, db)
    add_subkeyframes_from_dsg(G, db)
    add_edges_from_dsg(G, db)

    # Evidence provenance (additive, best-effort): Observation -> source-frame
    # edges by capture timestamp, and CameraCalib nodes from the on-disk
    # camera_calib.json files so visibility reasoning is possible from the DB.
    try:
        backfill_agent_timestamps(db)
        attach_camera_calibs(db)
        link_observations_to_frames(db)
    except Exception:
        logger.exception("evidence-provenance materialization failed (non-fatal)")

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