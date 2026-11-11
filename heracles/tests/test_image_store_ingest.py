"""Ingesting a Hydra image store (hydra_python.image_store) alongside a DSG.

The database tests require a disposable Neo4j instance (its contents are
cleared). Configure it with HERACLES_TEST_NEO4J_URI / HERACLES_TEST_NEO4J_PASSWORD.
"""

import os
import sqlite3

import numpy as np
import pytest
from dsg_fixture import AGENT0, AGENT1, OBJ1, OBJ2, build_dsg

from heracles import constants
from heracles import graph_interface as gi
from heracles.query_interface import Neo4jWrapper
from heracles.utils import extract_labelspaces_from_dsg

image_store = pytest.importorskip("hydra_python.image_store")
iio = pytest.importorskip("imageio.v3")

NEO4J_URI = os.environ.get("HERACLES_TEST_NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"))


def _img(path, value=128, channels=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (6, 8, channels) if channels > 1 else (6, 8)
    iio.imwrite(path, np.full(shape, value, dtype=np.uint8))
    return path


def _T(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def build_store(run_dir):
    """Image store matching dsg_fixture.build_dsg().

    Keyframes: a0's own (100 ns, a0 has no stamp: identity anchor_T_body), a1's
    own (2000 ns = a1's stamp), one more anchored to a1 (1500 ns, 0.5 m ahead)
    and an unanchored one (1600 ns). Observations: O1 in a0's keyframe (kept,
    mask PNG) and O2 (merged into O1) in an observation-only frame (RLE).
    """
    src = run_dir / "src"
    store_dir = run_dir / "image_store"
    w = image_store.ImageStore.open_writer(store_dir, robot_id=0)
    sid = w.add_sensor("cam", 8, 6, 5.0, 5.0, 4.0, 3.0, 0.001, _T((0.4, 0, 0.1)))
    kf = dict(kind=image_store.FrameKind.KEYFRAME, link="copy")
    kf0 = w.add_frame_ref(
        sid, 100, world_T_body_odom=_T((1, 0, 0)), rgb=_img(src / "kf0.png"),
        anchor_node_id=AGENT0, **kf,
    )  # fmt: skip
    w.add_frame_ref(
        sid, 2000, world_T_body_odom=_T((2, 0, 0)), rgb=_img(src / "kf1.png"),
        anchor_node_id=AGENT1, **kf,
    )  # fmt: skip
    w.add_frame_ref(
        sid, 1500, world_T_body_odom=_T((9, 9, 9)), rgb=_img(src / "s0.png"),
        anchor_node_id=AGENT1, anchor_T_body=_T((0.5, 0, 0)), **kf,
    )  # fmt: skip
    w.add_frame_ref(
        sid, 1600, world_T_body_odom=_T((3, 3, 3)), rgb=_img(src / "s1.png"), **kf
    )
    obs_frame = w.add_frame_ref(
        sid, 101, kind=image_store.FrameKind.OBSERVATION, world_T_body_odom=_T((1, 0, 0)),
        rgb=_img(src / "o0.png"), anchor_node_id=AGENT0, link="copy",
    )  # fmt: skip
    w.bind_track(7, OBJ1)
    w.bind_track(8, OBJ2)
    w.record_merges({OBJ2: OBJ1})
    w.add_observation(
        7, kf0, bbox2d=(1, 2, 5, 6), score=0.9, kept=True,
        mask_path=_img(src / "m.png", 255, 1), link="copy",
    )  # fmt: skip
    w.add_observation(8, obs_frame, bbox2d=(0, 0, 3, 3), score=0.4, mask_rle=[2, 3, 4])
    w.close()
    return store_dir


@pytest.fixture()
def store_dir(tmp_path):
    return build_store(tmp_path)


@pytest.fixture()
def store(store_dir):
    s = image_store.ImageStore.open(store_dir)
    yield s
    s.close()


@pytest.fixture()
def db():
    wrapper = Neo4jWrapper(NEO4J_URI, NEO4J_AUTH, atomic_queries=True)
    wrapper.connect()
    gi.initialize_db(wrapper)
    yield wrapper
    wrapper.close()


# ---------------------------------------------------------------------------
# No database
# ---------------------------------------------------------------------------


def test_open_image_store(store_dir, store):
    assert gi.open_image_store(None) is None
    assert gi.open_image_store([]) is None
    assert gi.open_image_store(store) is store
    for arg in (str(store_dir), [store_dir], str(store_dir.parent)):  # run dir too
        s = gi.open_image_store(arg)
        assert gi._store_paths(s) == [str(store_dir)]
        s.close()


def test_open_image_store_errors(tmp_path, store_dir):
    with pytest.raises(gi.ImageStoreError):
        gi.open_image_store(str(tmp_path / "missing"))
    # frame kind 1 (sub-keyframes of early stores) is not in this schema
    with sqlite3.connect(store_dir / "store.sqlite") as c:
        c.execute("UPDATE frames SET kind = 1 WHERE timestamp_ns = 1500")
    with pytest.raises(gi.ImageStoreError):
        gi.open_image_store(str(store_dir))


def test_split_keyframes(store):
    own, others = gi.split_keyframes(build_dsg(), store)
    # a0 has no stamp: its identity-anchored keyframe; a1: the one at its stamp
    assert {a: f.timestamp_ns for a, f in own.items()} == {AGENT0: 100, AGENT1: 2000}
    assert [f.timestamp_ns for f in others] == [1500, 1600]


def test_split_keyframes_stamp_too_far(store, monkeypatch):
    monkeypatch.setattr(gi, "_agent_stamps", lambda G: {AGENT1: 10**9})
    own, others = gi.split_keyframes(build_dsg(), store)
    assert AGENT1 not in own
    assert [f.timestamp_ns for f in others] == [1500, 1600, 2000]


@pytest.mark.parametrize("missing_agent", [AGENT0, AGENT1])
def test_missing_anchor_keyframes_remain_subkeyframes(store, missing_agent):
    G = build_dsg()
    G.remove_node(missing_agent)
    own, others = gi.split_keyframes(G, store)
    assert missing_agent not in own
    frames = store.frames(kind=image_store.FrameKind.KEYFRAME)
    assert {f.frame_id for f in frames} == {
        f.frame_id for f in [*own.values(), *others]
    }
    assert {f.frame_id for f in frames if f.anchor_node_id == missing_agent} <= {
        f.frame_id for f in others
    }


def test_rot_to_quat():
    for q in ([1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 1, 0, 0], [0, 0, 0, 1]):
        w, x, y, z = q
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        out = gi._rot_to_quat(R)
        assert np.allclose(out, q) or np.allclose(out, -np.array(q))


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _rows(db, query):
    records, _, _ = db.execute(query)
    return [dict(r) for r in records]


def test_spark_dsg_to_db_with_image_store(db, store_dir, store):
    G = build_dsg()
    gi.spark_dsg_to_db(G, db, image_store=str(store_dir))

    agents = _rows(
        db,
        f"MATCH (a:{constants.AGENTS}) RETURN a.nodeSymbol AS s, a.rgb_path AS p, "
        "a.timestamp_ns AS t ORDER BY s",
    )
    assert [(a["s"], a["t"]) for a in agents] == [("a0", 100), ("a1", 2000)]
    assert all(os.path.isfile(a["p"]) for a in agents)

    subs = _rows(
        db,
        f"MATCH (k:{constants.SUBKEYFRAMES}) "
        f"OPTIONAL MATCH (k)-[:{constants.ANCHORED_TO}]->(a:{constants.AGENTS}) "
        "RETURN k.timestamp_ns AS t, k.center AS c, a.nodeSymbol AS a ORDER BY t",
    )
    assert [(s["t"], tuple(s["c"]), s["a"]) for s in subs] == [
        (1500, (4.5, 5.0, 6.0), "a1"),  # a1's pose composed with anchor_T_body
        (1600, (3.0, 3.0, 3.0), None),  # unanchored: odometry
    ]

    obs = _rows(
        db,
        f"MATCH (:{constants.OBJECTS} {{nodeSymbol: 'O1'}})"
        f"-[:{constants.HAS_OBSERVATION}]->(o:{constants.OBSERVATIONS}) "
        "RETURN o.timestamp_ns AS t, o.kept AS kept, o.mask_path AS mp, "
        "o.mask_rle AS rle, o.bbox_2d AS bb, o.rgb_path AS p ORDER BY t",
    )
    assert [(o["t"], o["kept"], o["bb"]) for o in obs] == [
        (100, True, [1, 2, 5, 6]),
        (101, False, [0, 0, 3, 3]),  # O2's, merged into O1
    ]
    assert os.path.isfile(obs[0]["mp"]) and obs[1]["mp"] is None
    assert obs[1]["rle"] and all(os.path.isfile(o["p"]) for o in obs)

    links = _rows(
        db,
        f"MATCH (o:{constants.OBSERVATIONS})-[:{constants.OBSERVED_IN}]->(f) "
        "RETURN o.timestamp_ns AS t, f.nodeSymbol AS f",
    )
    assert links == [{"t": 100, "f": "a0"}]

    calibs = _rows(
        db,
        f"MATCH (n)-[:{constants.HAS_CALIB}]->(c:{constants.CAMERA_CALIBS}) "
        "RETURN n.nodeSymbol AS n, c.fx AS fx ORDER BY n",
    )
    anchored_sub = gi.split_keyframes(G, store)[1][0]  # 1500 ns, anchored to a1
    assert [c["n"] for c in calibs] == [
        "a0",
        "a1",
        gi.subkeyframe_symbol(anchored_sub.frame_id),
    ]
    assert all(c["fx"] == 5.0 for c in calibs)

    meta = _rows(
        db,
        "MATCH (m:_GraphMetadata {key: 'image_store'}) RETURN m.image_store_paths AS p",
    )
    assert meta == [{"p": [str(store_dir)]}]

    # The image nodes stay out of the reconstructed graph.
    assert gi.db_to_spark_dsg(db).num_nodes() == G.num_nodes() - 2


@pytest.mark.parametrize(
    "missing_agent, expected_stamp, expected_position",
    [(AGENT0, 100, (1.0, 0.0, 0.0)), (AGENT1, 2000, (2.0, 0.0, 0.0))],
)
def test_missing_anchor_keyframes_are_ingested_with_odometry(
    db, store_dir, store, missing_agent, expected_stamp, expected_position
):
    G = build_dsg()
    G.remove_node(missing_agent)
    gi.spark_dsg_to_db(G, db, image_store=store_dir)
    rows = _rows(
        db,
        "MATCH (f) WHERE f:Agent OR f:SubKeyframe "
        "RETURN f.frame_id AS id, f.timestamp_ns AS stamp, f.center AS center, "
        "labels(f) AS labels",
    )
    assert {r["id"] for r in rows} == {
        f.frame_id for f in store.frames(kind=image_store.FrameKind.KEYFRAME)
    }
    recovered = next(r for r in rows if r["stamp"] == expected_stamp)
    assert constants.SUBKEYFRAMES in recovered["labels"]
    np.testing.assert_allclose(tuple(recovered["center"]), expected_position)
    assert _rows(
        db,
        "MATCH (f:SubKeyframe)-[:ANCHORED_TO]->(a:Agent) "
        f"WHERE a.nodeSymbol = '{gi._node_symbol_str(missing_agent)}' "
        "RETURN count(*) AS n",
    ) == [{"n": 0}]


def test_staged_ingest_matches_one_shot(db, store_dir):
    """The image steps can run after a graph-only ingest (e.g. to show the map
    before its images are loaded) and give the same database."""
    G = build_dsg()

    def snapshot():
        rows = _rows(
            db,
            "MATCH (n) WHERE NOT n:_GraphMetadata OPTIONAL MATCH (n)-[r]->(m) "
            "RETURN labels(n) AS l, properties(n) AS n, type(r) AS r, "
            "m.nodeSymbol AS m",
        )
        return sorted(rows, key=repr)

    gi.spark_dsg_to_db(G, db, image_store=store_dir)
    expected = snapshot()

    gi.initialize_db(db)
    gi.spark_dsg_to_db(G, db)
    store = gi.open_image_store(store_dir)
    gi.add_agents_from_dsg(G, db, store)
    object_ls, _ = extract_labelspaces_from_dsg(G)
    gi.add_objects_from_dsg(G, db, store, object_labelspace=object_ls)
    gi.add_subkeyframes_from_dsg(G, db, store)
    assert gi.attach_camera_calibs(db, store, G) == 3
    assert gi.link_observations_to_frames(db) == 1
    store.close()
    assert snapshot() == expected


def test_observations_link_to_their_own_frame_not_one_at_the_same_stamp(db, store_dir):
    """Another robot's keyframe taken at the same nanosecond gets no edge."""
    gi.spark_dsg_to_db(build_dsg(), db, image_store=store_dir)
    db.execute(
        f"CREATE (:{constants.SUBKEYFRAMES} "
        "{nodeSymbol: 's_other_robot', timestamp_ns: 100, frame_id: -1})"
    )
    assert gi.link_observations_to_frames(db) == 1
    links = _rows(
        db,
        f"MATCH (o:{constants.OBSERVATIONS})-[:{constants.OBSERVED_IN}]->(f) "
        "RETURN o.timestamp_ns AS t, f.nodeSymbol AS f",
    )
    assert links == [{"t": 100, "f": "a0"}]


def test_keyframes_are_split_once_per_ingest(db, store_dir, monkeypatch):
    calls = []
    split = gi.split_keyframes

    def counting(G, store):
        calls.append(1)
        return split(G, store)

    monkeypatch.setattr(gi, "split_keyframes", counting)
    gi.spark_dsg_to_db(build_dsg(), db, image_store=store_dir)
    assert len(calls) == 1


def test_a_store_opened_by_the_load_is_closed_when_it_fails(
    db, store_dir, store, monkeypatch
):
    closed = []
    close = image_store.ImageStore.close

    def spy(self):
        closed.append(self)
        return close(self)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(image_store.ImageStore, "close", spy)
    monkeypatch.setattr(gi, "add_subkeyframes_from_dsg", boom)
    with pytest.raises(RuntimeError, match="boom"):
        gi.spark_dsg_to_db(build_dsg(), db, image_store=store_dir)
    assert len(closed) == 1 and closed[0] is not store

    # a store the caller opened stays open for the caller to close
    with pytest.raises(RuntimeError, match="boom"):
        gi.spark_dsg_to_db(build_dsg(), db, image_store=store)
    assert len(closed) == 1
