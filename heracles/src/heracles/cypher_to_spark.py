import spark_dsg


def spark_object_from_db(db_object, label_to_id):
    try:
        attrs = spark_dsg.KhronosObjectAttributes()
    except AttributeError:
        # Fallback for versions without Khronos attributes
        attrs = spark_dsg.ObjectNodeAttributes()

    attrs.name = db_object["name"]
    attrs.position = [db_object["pos_x"], db_object["pos_y"], db_object["pos_z"]]
    attrs.semantic_label = label_to_id[db_object["class"]]
    attrs.bounding_box = spark_dsg.BoundingBox(
        [db_object["bbox_l"], db_object["bbox_w"], db_object["bbox_h"]],  # dimensions
        [db_object["bbox_x"], db_object["bbox_y"], db_object["bbox_z"]],  # center
    )

    if "image_folder" in db_object:
        attrs.image_folder = db_object["image_folder"]

    try:
        if "details" in db_object:
            attrs.details = db_object["details"]
    except AttributeError:
        pass
        
    try:
        if "observations" in db_object:
            attrs.observations = db_object["observations"]
    except AttributeError:
        pass

    return attrs


def spark_place_from_db(db_place):
    attrs = spark_dsg.PlaceNodeAttributes()
    # Assuming standard place attributes
    # attrs.name = db_place.get("nodeSymbol", "") # Not always set or needed?
    attrs.position = [db_place["x"], db_place["y"], db_place["z"]]
    return attrs


def spark_mesh_place_from_db(db_mesh_place, label_to_id):
    attrs = spark_dsg.Place2dNodeAttributes()
    attrs.position = [db_mesh_place["x"], db_mesh_place["y"], db_mesh_place["z"]]
    # attrs.semantic_label = label_to_id[db_mesh_place["class"]] # Verify if MeshPlaces have semantic labels in DB mapping
    # Looking at graph_interface.py:168, mesh places have 'class' and 'center' (x,y,z)
    if "class" in db_mesh_place:
         attrs.semantic_label = label_to_id[db_mesh_place["class"]]
    return attrs


def spark_room_from_db(db_room, label_to_id):
    attrs = spark_dsg.RoomNodeAttributes()
    attrs.position = [db_room["x"], db_room["y"], db_room["z"]]
    if "class" in db_room:
        attrs.semantic_label = label_to_id[db_room["class"]]
    return attrs


def spark_building_from_db(db_building):
    # Buildings usually just have position
    # graph_interface.py uses a generic dict, BuildingNodeAttributes likely similar to others
    # Using generic/base attributes if specific one doesn't exist or isn't needed with much detail
    # But spark_dsg likely has BuildingNodeAttributes
    # Check graph_interface.py imports... it doesn't show building attributes usage explicitly 
    # except in add_buildings_from_dsg -> insert_buildings_to_db.
    # We will assume a basic structure or return a dict if attributes are not strictly typed in python bindings
    # However, to be consistent with others, let's look for attributes.
    # Since I don't see BuildingNodeAttributes explicitly used in graph_interface.py (it just does building_to_dict),
    # I'll Assume standard NodeAttributes or similar.
    # Actually, let's check if spark_dsg has BuildingNodeAttributes.
    # Based on other types, it's probable.
    # If not, we'll use NodeAttributes.
    try:
        attrs = spark_dsg.BuildingNodeAttributes()
    except AttributeError:
        # Fallback if class doesn't exist
        attrs = spark_dsg.NodeAttributes()
        
    attrs.position = [db_building["x"], db_building["y"], db_building["z"]]
    return attrs
