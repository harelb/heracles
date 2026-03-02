import os
import glob
import json
import neo4j
import parse
import spark_dsg

from . import constants


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


# Insert all nodes and edges for each layer
def spark_dsg_to_db(G, image_folder_root, db):
    add_agents_from_dsg(G, image_folder_root, db)
    add_objects_from_dsg(G, image_folder_root, db)
    add_places_from_dsg(G, db)
    add_mesh_places_from_dsg(G, db)
    add_rooms_from_dsg(G, db)
    add_buildings_from_dsg(G, db)
    add_edges_from_dsg(G, db)


# Inserting agents
def add_agents_from_dsg(G, image_folder_root, db):
    agents = []
    layer = G.get_layer(spark_dsg.DsgLayers.AGENTS)
    if layer is None:
        return
        
    for a in layer.nodes:
        d = agent_to_dict(a)
        if "image_folder" in d and d["image_folder"]:
            d["image_folder"] = os.path.join(
                image_folder_root, os.path.basename(d["image_folder"])
            )
        agents.append(d)

    if agents:
        insert_agents_to_db(db, agents)

def agent_to_dict(agent):
    attrs = agent.attributes
    d = {}
    d["nodeSymbol"] = agent.id.str(True)
    d["pos_x"] = attrs.position[0]
    d["pos_y"] = attrs.position[1]
    d["pos_z"] = attrs.position[2]
    
    if hasattr(attrs, "image_folder"):
        d["image_folder"] = attrs.image_folder
        
    return d

def insert_agents_to_db(db, agents):
    return db.execute(
        f"""
    WITH $agents AS agents
    UNWIND agents AS agent
    WITH point({{x: agent.pos_x, y: agent.pos_y, z: agent.pos_z}}) AS p3d, agent
    MERGE (n:{constants.AGENTS} {{nodeSymbol: agent.nodeSymbol}})
    SET n.center = p3d,
        n.image_folder = agent.image_folder
    """,
        agents=agents,
    )

# Inserting objects
def add_objects_from_dsg(G, image_folder_root, db):
    objects = []
    for o in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        d = obj_to_dict(G.metadata.get()["labelspace"], o)
        if "image_folder" in d and d["image_folder"]:
            d["image_folder"] = os.path.join(
                image_folder_root, os.path.basename(d["image_folder"])
            )
        objects.append(d)

    insert_objects_to_db(db, objects)
    
    # Process observations
    observations = []
    object_observation_edges = []
    
    for obj in objects:
        if "image_folder" not in obj or not obj["image_folder"]:
            continue
            
        image_folder = obj["image_folder"]
        meta_files = glob.glob(os.path.join(image_folder, "*_meta.json"))
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r') as f:
                    data = json.load(f)
                    
                # Construct observation ID/Symbol: ObjectSymbol_Timestamp
                timestamp_ns = data.get("timestamp_ns")
                if timestamp_ns is None:
                    continue
                    
                obs_symbol = f"{obj['nodeSymbol']}_{timestamp_ns}"
                
                obs_dict = {
                    "nodeSymbol": obs_symbol,
                    "timestamp_ns": timestamp_ns,
                    "mask_file": data.get("mask_file", ""),
                }
                
                # Extract 2D Bounding Box
                if "bbox_2d" in data:
                     bbox_2d = data["bbox_2d"]
                     obs_dict["bbox_2d_min_x"] = bbox_2d.get("min_x")
                     obs_dict["bbox_2d_min_y"] = bbox_2d.get("min_y")
                     obs_dict["bbox_2d_max_x"] = bbox_2d.get("max_x")
                     obs_dict["bbox_2d_max_y"] = bbox_2d.get("max_y")
                
                observations.append(obs_dict)
                object_observation_edges.append({"from": obj["nodeSymbol"], "to": obs_symbol})
                
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


def insert_observations_to_db(db, observations):
    return db.execute(
        f"""
    WITH $observations AS observations
    UNWIND observations AS obs
    MERGE (:{constants.OBSERVATIONS} {{
        nodeSymbol: obs.nodeSymbol, 
        timestamp_ns: obs.timestamp_ns,
        mask_file: obs.mask_file,
        bbox_2d_min_x: obs.bbox_2d_min_x,
        bbox_2d_min_y: obs.bbox_2d_min_y,
        bbox_2d_max_x: obs.bbox_2d_max_x,
        bbox_2d_max_y: obs.bbox_2d_max_y
    }})
    """,
        observations=observations,
    )


def obj_to_dict(node_classes, obj):
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
    # d["color_r"] = attrs.color[0]
    # d["color_g"] = attrs.color[1]
    # d["color_b"] = attrs.color[2]

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


def insert_objects_to_db(db, objects):
    return db.execute(
        #    f"""
        #    WITH $objects AS objects
        #    UNWIND objects AS object
        #    WITH point({{x: object.pos_x, y: object.pos_y, z: object.pos_z}}) AS p3d, point({{x: object.bbox_x, y: object.bbox_y, z: object.bbox_z}}) AS bb3d, point({{x: object.bbox_l, y: object.bbox_w, z: object.bbox_h}}) AS bbdim, object
        #    MERGE (:{constants.OBJECTS} {{nodeSymbol: object.nodeSymbol, center: p3d, bbox_center: bb3d, bbox_dim: bbdim, class: object.class}})
        #    """,
        f"""
    WITH $objects AS objects
    UNWIND objects AS object
    WITH point({{x: object.pos_x, y: object.pos_y, z: object.pos_z}}) AS p3d, point({{x: object.bbox_x, y: object.bbox_y, z: object.bbox_z}}) AS bb3d, point({{x: object.bbox_l, y: object.bbox_w, z: object.bbox_h}}) AS bbdim,  object
    MERGE (n:Object {{nodeSymbol: object.nodeSymbol}})
    SET n.center = p3d,
        n.bbox_center = bb3d,
        n.bbox_dim = bbdim,
        n.class = object.class,
        n.name = object.name,
        n.image_folder = object.image_folder,
        n.details = object.details,
        n.first_observed_ns = object.first_observed_ns,
        n.last_observed_ns = object.last_observed_ns
    """,
        objects=objects,
    )


# Inserting places
def place_to_dict(place):
    attrs = place.attributes
    d = {}
    d["nodeSymbol"] = place.id.str(True)
    d["x"] = attrs.position[0]
    d["y"] = attrs.position[1]
    d["z"] = attrs.position[2]
    return d


def add_places_from_dsg(G, db):
    places = [place_to_dict(p) for p in G.get_layer(spark_dsg.DsgLayers.PLACES).nodes]
    insert_places_to_db(db, places)


def insert_places_to_db(db, places):
    return db.execute(
        f"""
    WITH $places AS places
    UNWIND places AS place
    WITH point({{x: place.x, y: place.y, z: place.z}}) AS p3d, place
    MERGE (:{constants.PLACES} {{nodeSymbol: place.nodeSymbol, center: p3d}})
    """,
        places=places,
    )


# Inserting mesh places


def mesh_place_to_dict(node_classes, mesh_place):
    attrs = mesh_place.attributes
    d = {}
    d["nodeSymbol"] = mesh_place.id.str(True)
    d["x"] = attrs.position[0]
    d["y"] = attrs.position[1]
    d["z"] = attrs.position[2]
    d["class"] = node_classes[str(attrs.semantic_label)]
    return d


def add_mesh_places_from_dsg(G, db):
    try:
        mesh_place_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except IndexError:
        mesh_place_layer = G.get_layer(20)

    mesh_places = [
        mesh_place_to_dict(G.metadata.get()["labelspace"], p)
        for p in mesh_place_layer.nodes
    ]
    insert_mesh_places_to_db(db, mesh_places)


def insert_mesh_places_to_db(db, mesh_places):
    return db.execute(
        f"""
    WITH $mesh_places AS places
    UNWIND places AS place
    WITH point({{x: place.x, y: place.y, z: place.z}}) AS p3d, place
    MERGE (:{constants.MESH_PLACES} {{nodeSymbol: place.nodeSymbol, center: p3d, class: place.class}})
    """,
        mesh_places=mesh_places,
    )


# Inserting Rooms


def add_rooms_from_dsg(G, db):
    if "room_labelspace" in G.metadata.get():
        labelspace = G.metadata.get()["room_labelspace"]
    else:
        labelspace = {"0": "Unknown"}

    rooms = [
        room_to_dict(labelspace, r)
        for r in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes
    ]
    insert_rooms_to_db(db, rooms)


def room_to_dict(node_classes, room):
    attrs = room.attributes
    d = {}
    d["nodeSymbol"] = room.id.str(True)
    d["x"] = attrs.position[0]
    d["y"] = attrs.position[1]
    d["z"] = attrs.position[2]
    d["class"] = node_classes[str(attrs.semantic_label)]
    return d


def insert_rooms_to_db(db, rooms):
    return db.execute(
        f"""
    WITH $rooms AS rooms
    UNWIND rooms AS room
    WITH point({{x: room.x, y: room.y, z: room.z}}) AS p3d, room
    MERGE (:{constants.ROOMS} {{nodeSymbol: room.nodeSymbol, center: p3d, class: room.class}})
    """,
        rooms=rooms,
    )


# Inserting Buildings


def building_to_dict(building):
    attrs = building.attributes
    d = {}
    d["nodeSymbol"] = building.id.str(True)
    d["x"] = attrs.position[0]
    d["y"] = attrs.position[1]
    d["z"] = attrs.position[2]
    return d


def add_buildings_from_dsg(G, db):
    buildings = [
        building_to_dict(p) for p in G.get_layer(spark_dsg.DsgLayers.BUILDINGS).nodes
    ]
    insert_buildings_to_db(db, buildings)


def insert_buildings_to_db(db, buildings):
    return db.execute(
        f"""
    WITH $buildings AS buildings
    UNWIND buildings AS building
    WITH point({{x: building.x, y: building.y, z: building.z}}) AS p3d, building
    MERGE (:{constants.BUILDINGS} {{nodeSymbol: building.nodeSymbol, center: p3d}})
    """,
        buildings=buildings,
    )


def add_edges_from_dsg(G, db):
    print("Adding Edges")
    layer_id_to_layer_str = G.metadata.get()["LayerIdToHeraclesLayerStr"]

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
    query = f"""
    WITH $connections AS connections
    UNWIND connections AS connection
    MATCH (n1: {from_label} {{nodeSymbol: connection.from}})
    MATCH (n2: {to_label} {{nodeSymbol: connection.to}})
    CREATE (n1)-[:{edge_type}]->(n2)
    """
    ret = db.execute(query, connections=connections)
    return ret


def get_layer_nodes(db, layer):
    if layer == constants.OBJECTS:
        records, summary, keys = db.execute(
            f"""
          Match (p:{layer})
          RETURN p.nodeSymbol as nodeSymbol, p.class as class, p.center as center, p.bbox_center as bbox_center, p.bbox_dim as bbox_dim, p.name as name
          """
        )
    else:
        records, summary, keys = db.execute(
            f"""
          Match (p:{layer})
          RETURN p.nodeSymbol as nodeSymbol, p.class as class, p.center as center
          """
        )
    return records, summary, keys


def get_db_edges(db, edge_type, from_label, to_label):
    query = f"""
    MATCH (a:{from_label})-[:{edge_type}]->(b:{to_label})
    RETURN a.nodeSymbol as from, b.nodeSymbol as to
    """
    records, summary, keys = db.execute(query)
    return records, summary, keys


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
    # Add the Object-Object edges
    records, _, _ = get_db_edges(
        db, "OBJECT_CONNECTED", constants.OBJECTS, constants.OBJECTS
    )
    insert_edges_to_spark(G, records)
    # Add the MeshPlace-MeshPlace edges
    records, _, _ = get_db_edges(
        db, "MESH_PLACE_CONNECTED", constants.MESH_PLACES, constants.MESH_PLACES
    )
    insert_edges_to_spark(G, records)
    # Add the Place-Place edges
    records, _, _ = get_db_edges(
        db, "PLACE_CONNECTED", constants.PLACES, constants.PLACES
    )
    insert_edges_to_spark(G, records)
    # Add the Room-Room edges
    records, _, _ = get_db_edges(db, "ROOM_CONNECTED", constants.ROOMS, constants.ROOMS)
    insert_edges_to_spark(G, records)
    # Add the Building-Building edges
    records, _, _ = get_db_edges(
        db, "BUILDING_CONNECTED", constants.BUILDINGS, constants.BUILDINGS
    )
    insert_edges_to_spark(G, records)

    #### INTERLAYER EDGES
    # Add the MeshPlace-Object edges
    records, _, _ = get_db_edges(
        db, "CONTAINS", constants.MESH_PLACES, constants.OBJECTS
    )
    insert_edges_to_spark(G, records)
    # Add the Place-Object edges
    records, _, _ = get_db_edges(db, "CONTAINS", constants.PLACES, constants.OBJECTS)
    insert_edges_to_spark(G, records)
    # Add the Room-Place edges
    records, _, _ = get_db_edges(db, "CONTAINS", constants.ROOMS, constants.PLACES)
    insert_edges_to_spark(G, records)
    # Add the Room-Place edges
    records, _, _ = get_db_edges(db, "CONTAINS", constants.ROOMS, constants.MESH_PLACES)
    insert_edges_to_spark(G, records)
    # Add the Building-Room edges
    records, _, _ = get_db_edges(db, "CONTAINS", constants.BUILDINGS, constants.ROOMS)
    insert_edges_to_spark(G, records)
    return


def db_to_spark_mesh_place(mp, label_to_semantic_id):
    attrs = spark_dsg.Place2dNodeAttributes()
    attrs.name = mp["nodeSymbol"]
    attrs.position = mp["center"]
    attrs.semantic_label = label_to_semantic_id[mp["class"]]
    return attrs


def db_to_spark_place(p, label_to_semantic_id):
    attrs = spark_dsg.PlaceNodeAttributes()
    attrs.name = p["nodeSymbol"]
    attrs.position = p["center"]
    return attrs


def db_to_spark_object(o, label_to_semantic_id):
    attrs = spark_dsg.ObjectNodeAttributes()
    attrs.name = o["nodeSymbol"]
    attrs.position = o["center"]
    
    # Robust Label Lookup
    cls_name = o["class"]
    if cls_name in label_to_semantic_id:
        attrs.semantic_label = label_to_semantic_id[cls_name]
    else:
        # Fallback to 0 if unknown, or try to find a generic one?
        # Assuming 0 is valid or "Unknown"
        # If we can't find it, we just set 0.
        # Ideally we print a warning?
        # print(f"Warning: Unknown class '{cls_name}', using default 0.")
        attrs.semantic_label = 0
        
    attrs.name = o["name"]
    # If name is empty, maybe use class?
    if not attrs.name:
         attrs.name = cls_name
         
    attrs.bounding_box = spark_dsg.BoundingBox(
        [o["bbox_dim"][0], o["bbox_dim"][1], o["bbox_dim"][2]],  # dimensions
        [o["bbox_center"][0], o["bbox_center"][1], o["bbox_center"][2]],  # center
    )
    return attrs


def db_to_spark_room(r, room_label_to_semantic_id):
    attrs = spark_dsg.RoomNodeAttributes()
    attrs.name = r["nodeSymbol"]
    attrs.position = r["center"]
    attrs.semantic_label = room_label_to_semantic_id[r["class"]]
    attrs.bounding_box = spark_dsg.BoundingBox([0.1, 0.1, 0.1])
    return attrs


def db_to_spark_dsg(
    db, spark_layer_id_to_layer_name, label_to_semantic_id, room_label_to_semantic_id
):
    # Initialize the spark_dsg scene graph object
    new_scene_graph = spark_dsg.DynamicSceneGraph()
    new_scene_graph.clear(True)  # Removes all layers

    # -- Dynamic Label Expansion Logic --
    # Hack: Pre-fetch objects to find unknown classes so we can add them to labelspace
    # We iterate layers, if we match OBJECTS, we fetch records early.
    
    # We need to process layers in order or just find the Objects layer key
    # spark_layer_id_to_layer_name maps generic ID to heracles string
    # We look for constants.OBJECTS
    
    # Let's pre-scan objects
    records_map = {} # Cache records to avoid repeated DB calls
    
    for spark_layer_id, heracles_layer_name in spark_layer_id_to_layer_name.items():
        records, summary, keys = get_layer_nodes(db, heracles_layer_name)
        # Consure iterator to list so we can iterate twice
        records_list = list(records)
        records_map[heracles_layer_name] = records_list
        
        if heracles_layer_name == constants.OBJECTS:
            # Update label map
            next_id = max(label_to_semantic_id.values()) + 1 if label_to_semantic_id else 1
            
            for rec in records_list:
                c = rec["class"]
                if c and c not in label_to_semantic_id:
                    # Assign new ID
                    label_to_semantic_id[c] = next_id
                    print(f"Assigning new class '{c}' -> {next_id}")
                    next_id += 1

    object_labelspace = spark_dsg.Labelspace(
        {v: k for k, v in label_to_semantic_id.items()}
    )
    new_scene_graph.set_labelspace(object_labelspace, 2, 0)
    
    # Add each layer (LayerID, PythonPartitionID, Name)
    for spark_layer_id, heracles_layer_name in spark_layer_id_to_layer_name.items():
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
            
        # Use cached records
        records = records_map.get(heracles_layer_name, [])
        # records, summary, keys = get_layer_nodes(db, heracles_layer_name) # OLD call
        # Assign the function to get the attributes
        # TODO - Can we have a generic function for retreiving all of the attributes?
        attr_func = None
        match heracles_layer_name:
            case constants.MESH_PLACES:
                attr_func = db_to_spark_mesh_place
            case constants.PLACES:
                attr_func = db_to_spark_place
            case constants.OBJECTS:
                attr_func = db_to_spark_object
            case constants.ROOMS:
                attr_func = db_to_spark_room
            case constants.BUILDINGS:
                # attr_func = db_to_spark_building
                pass
            case _:
                raise ValueError(
                    f'Unexpected heracles layer name "{heracles_layer_name}"'
                )
        for record in records:
            attrs = []
            if heracles_layer_name == constants.ROOMS:
                attrs = attr_func(record, room_label_to_semantic_id)
            else:
                attrs = attr_func(record, label_to_semantic_id)
            new_scene_graph.add_node(
                constants.HERACLES_TO_SPARK_LAYER_NAMES[heracles_layer_name],
                str_to_ns_value(record["nodeSymbol"]),
                attrs,
            )
    # Add all of the edges
    add_edges_from_db(db, new_scene_graph)
    return new_scene_graph