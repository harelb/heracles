from types import MappingProxyType

import spark_dsg

# Constant strings for referencing layers in heracles
MESH_PLACES = "MeshPlace"
PLACES = "Place"
OBJECTS = "Object"
AGENTS = "Agent"
OBSERVATIONS = "Observation"
ROOMS = "Room"
BUILDINGS = "Building"
HAS_OBSERVATION = "HAS_OBSERVATION"
TRAJECTORY_FRAMES = "TrajectoryFrame"
STATE_NODE = "_State"
OBSERVED_AT = "OBSERVED_AT"
DEPICTS = "DEPICTS"
SUBKEYFRAMES = "SubKeyframe"
ANCHORED_TO = "ANCHORED_TO"
CAMERA_CALIBS = "CameraCalib"
HAS_CALIB = "HAS_CALIB"
OBSERVED_IN = "OBSERVED_IN"
SEARCH_EPISODES = "SearchEpisode"
ASSERTIONS = "Assertion"
COVERED = "COVERED"
SUPPORTS = "SUPPORTS"
CONTRADICTS = "CONTRADICTS"
VIEWS = "VIEWS"

# ---------------------------------------------------------------------------
# Evidence-admission gate (G1). An :Object row is only a *planning fact* once
# something explicitly admitted it; `admission_status` records that decision
# and `admission_policy_version` pins the policy that made it. Kept here so the
# schema has one owner -- agentic_navigation.evidence.admission mirrors these
# and a drift-guard test asserts the two stay equal.
# ---------------------------------------------------------------------------
ADMISSION_STATUS = "admission_status"
ADMISSION_POLICY_VERSION = "admission_policy_version"
ADMISSION_OBSERVATIONS = "admission_observation_ids"
ADMISSION_REASON = "admission_reason"
# Measured confidence attached by the admitting policy (M2); absent when the
# admitting authority had no graded signal.
ADMISSION_BELIEF = "admission_belief"

# Values of `admission_status`.
TRUSTED_PRIOR = "trusted_prior"   # ingested from a committed scene graph
CANDIDATE = "candidate"           # a detector wrote it; nobody graded it
ADMITTED = "admitted"             # an authority graded it and accepted it
REJECTED = "rejected"             # an authority graded it and refused it

# The policy under which objects ingested by spark_dsg_to_db are trusted.
PRIOR_MAP_POLICY_VERSION = "prior_map_legacy_v0"

# Mappings to/from heracles and spark_dsg
SPARK_TO_HERACLES_LAYER_NAMES = MappingProxyType(
    {
        spark_dsg.DsgLayers.MESH_PLACES: MESH_PLACES,
        spark_dsg.DsgLayers.PLACES: PLACES,
        spark_dsg.DsgLayers.OBJECTS: OBJECTS,
        spark_dsg.DsgLayers.AGENTS: AGENTS,
        spark_dsg.DsgLayers.ROOMS: ROOMS,
        spark_dsg.DsgLayers.BUILDINGS: BUILDINGS,
    }
)
HERACLES_TO_SPARK_LAYER_NAMES = MappingProxyType(
    {
        MESH_PLACES: spark_dsg.DsgLayers.MESH_PLACES,
        PLACES: spark_dsg.DsgLayers.PLACES,
        OBJECTS: spark_dsg.DsgLayers.OBJECTS,
        AGENTS: spark_dsg.DsgLayers.AGENTS,
        ROOMS: spark_dsg.DsgLayers.ROOMS,
        BUILDINGS: spark_dsg.DsgLayers.BUILDINGS,
    }
)
