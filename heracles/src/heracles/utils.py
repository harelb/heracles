import os
from importlib.resources import as_file, files

import yaml

import heracles
import heracles.resources
from heracles.graph_interface import (
    initialize_db,
    spark_dsg_to_db,
)
from heracles.map_reload import KEEP_LABELS, MAP_STATE, reloading_map
from heracles.query_interface import Neo4jWrapper


def extract_labelspaces_from_dsg(G):
    """Extract object and room labelspaces from DSG metadata.

    Reads the new embedded format written by spark_dsg's ``set_labelspace``::

        metadata["labelspaces"]["_l2p0"] = [[0, "unknown"], [1, "chair"], ...]
        metadata["labelspaces"]["_l4p0"] = [[0, "lounge"], [1, "hallway"], ...]

    Returns
    -------
    (object_ls, room_ls) : tuple[dict | None, dict | None]
        Each is ``{str(int_id): name}`` or ``None`` if not found.
    """
    meta = G.metadata.get()
    labelspaces = meta.get("labelspaces")
    if not labelspaces:
        return None, None

    object_ls = None
    room_ls = None

    obj_data = labelspaces.get("_l2p0")  # Objects: layer 2, partition 0
    if obj_data:
        object_ls = {str(pair[0]): pair[1] for pair in obj_data}

    room_data = labelspaces.get("_l4p0")  # Rooms: layer 4, partition 0
    if room_data:
        room_ls = {str(pair[0]): pair[1] for pair in room_data}

    return object_ls, room_ls


def load_dsg_to_db(neo4j_uri, neo4j_creds, scene_graph, source_file_path=None):
    """Load a DSG into Neo4j.

    Labelspaces must be embedded in the DSG metadata via ``set_labelspace()``.
    Use ``extract_labelspaces_from_dsg()`` to verify before calling.

    Pass ``source_file_path`` when the DSG was loaded from a file: it is
    recorded so that the map can later be rebuilt from that file (undo, reset).
    """
    # Set the layer id to layer name mappings
    spark_layer_id_to_heracles_layer_str = {
        2: "Object",
        5: "Building",
        20: "MeshPlace",
        "3[1]": "MeshPlace",
        3: "Place",
        4: "Room",
    }
    scene_graph.metadata.add(
        {"LayerIdToHeraclesLayerStr": spark_layer_id_to_heracles_layer_str}
    )

    with Neo4jWrapper(
        neo4j_uri, neo4j_creds, atomic_queries=True, print_profiles=False
    ) as db:
        # Held as rebuilding throughout, so the publisher keeps the last map;
        # the version, robots and holding survive the wipe (map_reload).
        with reloading_map(db, "Loading a scene graph into Neo4j") as reload:
            # Clear any existing content from the DB & initialize with the schema
            print("Initializing the database.")
            initialize_db(db, keep_labels=KEEP_LABELS)
            reload.wiped()
            # Load the scene graph into the DB
            print("Loading the scene graph into the database.")
            spark_dsg_to_db(scene_graph, db, source_file_path=source_file_path)
            # _MapState survived the wipe: the file it names must be this one,
            # or a rebuild would start from the previous map.
            if source_file_path is not None:
                db.execute(
                    f"MERGE (m:{MAP_STATE}) SET m.prior_path = $p",
                    p=os.path.abspath(source_file_path),
                )
            else:
                db.execute(f"MATCH (m:{MAP_STATE}) REMOVE m.prior_path")
