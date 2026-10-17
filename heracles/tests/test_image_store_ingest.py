"""Image-store ingest tests that need no Neo4j (capture-DB stubs)."""

import numpy as np
import pytest
import spark_dsg
from image_store_fixture import AGENT0, AGENT1, build_dsg, build_store

from heracles import graph_interface as gi
from heracles.pose_math import rot_to_quat


class _CaptureDB:
    def __init__(self):
        self.calls = []

    def execute(self, query, **params):
        self.calls.append((query, params))
        return ([], None, None)


@pytest.fixture
def store(tmp_path):
    from hydra_image_store import ImageStore

    return ImageStore.open(build_store(tmp_path))


def test_open_image_store_accepts_path_list_and_object(tmp_path):
    from hydra_image_store import ImageStore

    d = build_store(tmp_path)
    assert gi.open_image_store(None) is None
    assert gi.open_image_store([]) is None
    assert gi.open_image_store(str(d)).roots
    assert gi.open_image_store([str(d)]).roots
    # A run dir containing image_store/ resolves too.
    assert gi.open_image_store(str(tmp_path)).roots
    s = ImageStore.open(d)
    assert gi.open_image_store(s) is s


def test_legacy_run_dir_raises_pointing_at_import_legacy(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "images").mkdir()
    with pytest.raises(gi.ImageStoreError, match="import-legacy"):
        gi.open_image_store(str(tmp_path))
    with pytest.raises(gi.ImageStoreError, match="import-legacy"):
        gi.spark_dsg_to_db(build_dsg(), _CaptureDB(), image_store=str(tmp_path))


def test_agents_get_keyframe_paths_from_store(store):
    agents = {a["nodeSymbol"]: a for a in gi._collect_agents(build_dsg(), store)}
    assert set(agents) == {"a0", "a1"}
    a0 = agents["a0"]
    assert a0["rgb_path"].endswith("100_rgb.jpg") and a0["depth_path"].endswith("100_depth.png")
    assert "image_folder" not in a0
    assert agents["a1"]["depth_path"] is None
    assert agents["a1"]["timestamp_ns"] == 200


def test_agents_without_store_have_no_frame_properties():
    agents = gi._collect_agents(build_dsg(), None)
    assert len(agents) == 2 and all("rgb_path" not in a for a in agents)


def test_subkeyframe_pose_composed_from_anchor_or_odom(store):
    sub = {s["timestamp_ns"]: s for s in gi._collect_subkeyframes(build_dsg(), store)}
    # anchored: a1 at (4,5,6) identity + (0.5,0,0)
    assert (sub[150]["pos_x"], sub[150]["pos_y"], sub[150]["pos_z"]) == (4.5, 5.0, 6.0)
    assert sub[150]["anchor_symbol"] == "a1" and sub[150]["rot_w"] == 1.0
    assert sub[150]["nodeSymbol"].startswith("s")
    # unanchored: odometry
    assert (sub[160]["pos_x"], sub[160]["pos_y"], sub[160]["pos_z"]) == (3.0, 3.0, 3.0)
    assert "anchor_symbol" not in sub[160]
    assert sub[160]["depth_path"] is None


def test_observation_dicts_merged_children_and_collisions(store):
    from heracles.graph_interface import _observation_dicts

    obs = store.observations(int(spark_dsg.NodeSymbol("O", 1).value))
    assert len(obs) == 2
    dicts = _observation_dicts("O1", obs)
    best = dicts[0]
    assert best["nodeSymbol"] == "O1_100" and best["kept"] is True
    assert best["bbox_2d"] == [1, 2, 5, 6] and best["mask_file"] == best["mask_path"].rsplit("/", 1)[-1]
    assert best["mask_path"].endswith(".png") and best["rgb_path"].endswith("100_rgb.jpg")
    assert dicts[1]["mask_path"] is None and dicts[1]["mask_file"] is None
    # Same object+timestamp twice -> second symbol disambiguated by obs_id.
    twice = _observation_dicts("O1", [obs[0], obs[0]])
    assert twice[0]["nodeSymbol"] != twice[1]["nodeSymbol"]


def test_rot_to_quat_roundtrip():
    for q in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]):
        q = np.array(q, float)
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        out = rot_to_quat(R)
        assert np.allclose(out, q) or np.allclose(out, -q)
