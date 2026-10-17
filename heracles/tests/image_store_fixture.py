"""Tiny image store + matching DSG shared by the image-store ingest tests."""

import pathlib

import numpy as np
import spark_dsg

hydra_image_store = __import__("pytest").importorskip("hydra_image_store")
PIL = __import__("pytest").importorskip("PIL.Image")

from hydra_image_store import FrameKind, ImageStore, make_node_id  # noqa: E402

AGENT0 = make_node_id("a", 0)
AGENT1 = make_node_id("a", 1)
OBJ1 = make_node_id("O", 1)
OBJ2 = make_node_id("O", 2)  # merged into OBJ1 (not in the DSG)
BODY_T_SENSOR = np.array(
    [[0, 0, 1, 0.4], [-1, 0, 0, 0.0], [0, -1, 0, 0.1], [0, 0, 0, 1.0]]
)


def _img(path, mode="RGB"):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "I;16":
        PIL.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(path)
    elif mode == "L":
        PIL.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(path)
    else:
        PIL.fromarray(np.full((6, 8, 3), 128, dtype=np.uint8)).save(path)
    return path


def _T(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def build_store(root, obj1=OBJ1, obj2=OBJ2):
    """Store: 2 keyframes (a0, a1), 2 sub-keyframes on a1, 2 tracks -> O1 (O2 merged)."""
    root = pathlib.Path(root)
    store_dir = root / "image_store"
    src = root / "src"
    w = ImageStore.open_writer(store_dir, robot_id=0)
    sid = w.add_sensor("cam", 8, 6, 5.0, 5.0, 4.0, 3.0, 0.001, BODY_T_SENSOR)
    kw = dict(link="copy")
    kf0 = w.add_frame_ref(
        sid, 100, kind=FrameKind.KEYFRAME, world_T_body_odom=_T((1, 0, 0)),
        rgb=_img(src / "kf0.jpg"), depth=_img(src / "kf0.png", "I;16"),
        anchor_node_id=AGENT0, **kw)
    kf1 = w.add_frame_ref(
        sid, 200, kind=FrameKind.KEYFRAME, world_T_body_odom=_T((2, 0, 0)),
        rgb=_img(src / "kf1.jpg"), anchor_node_id=AGENT1, **kw)
    w.add_frame_ref(
        sid, 150, kind=FrameKind.SUBKEYFRAME, world_T_body_odom=_T((9, 9, 9)),
        rgb=_img(src / "s0.jpg"), anchor_node_id=AGENT1,
        anchor_T_body=_T((0.5, 0, 0)), **kw)
    w.add_frame_ref(
        sid, 160, kind=FrameKind.SUBKEYFRAME, world_T_body_odom=_T((3, 3, 3)),
        rgb=_img(src / "s1.jpg"), **kw)  # unanchored
    of = w.add_frame_ref(
        sid, 100 + 1, kind=FrameKind.OBSERVATION, world_T_body_odom=_T((1, 0, 0)),
        rgb=_img(src / "o0.jpg"), anchor_node_id=AGENT0, **kw)
    w.bind_track(7, obj1)
    w.bind_track(8, obj2)
    w.record_merges({obj2: obj1})
    mask_png = _img(src / "m.png", "L")
    w.add_observation(7, kf0, bbox2d=(1, 2, 5, 6), score=0.9, kept=True, mask_path=mask_png)
    w.add_observation(8, of, bbox2d=(0, 0, 3, 3), score=0.4, kept=False)
    w.close()
    return store_dir


def build_dsg():
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, 0, spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(2, ord("a"), spark_dsg.DsgLayers.AGENTS)
    G.add_layer(3, "p", spark_dsg.DsgLayers.PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, "B", spark_dsg.DsgLayers.BUILDINGS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)
    obj = spark_dsg.ObjectNodeAttributes()
    obj.position = np.array([1.0, 2.0, 3.0])
    G.add_node(2, OBJ1, obj, 0)
    for nid, pos in ((AGENT0, (0.0, 0.0, 0.0)), (AGENT1, (4.0, 5.0, 6.0))):
        a = spark_dsg.AgentNodeAttributes()
        a.position = np.array(pos)
        a.world_R_body = spark_dsg.Quaternion(1.0, 0.0, 0.0, 0.0)
        G.add_node(2, nid, a, ord("a"))
    return G
