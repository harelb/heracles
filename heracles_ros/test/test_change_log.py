"""The map as prior + logged changes: rebuilds, undo, and change detection snapshots.

Wipes the database it connects to: uses the dedicated test instance
(HERACLES_TEST_NEO4J_URI) and skips without it.
"""

# The db fixture comes from test_map_state; ruff takes each test using it for a redefinition.
# ruff: noqa: F811

from types import SimpleNamespace as NS

import pytest
import spark_dsg
from geometry_msgs.msg import Point
from heracles import graph_edit
from heracles.graph_interface import db_to_spark_dsg
from heracles_ros_interfaces.msg import SceneChange, SceneChangeMsg
from test_map_state import (  # noqa: F401
    AUTH,
    DSG_PATH,
    ROBOT,
    URI,
    _connect,
    db,  # noqa: F401  fixture
)

from heracles_ros import change_log
from heracles_ros.map_state import (
    get_holding,
    get_map_version,
    get_prior_path,
    is_rebuilding,
    map_edit,
    notify_map_change,
)
from heracles_ros.scene_change import (
    ADDED,
    REMOVED,
    add_object,
    add_place,
    apply_changes,
    object_center,
    place_exists,
    remove_connection,
    remove_place,
)
from heracles_ros.scene_change_writer_node import SceneChangeWriter


@pytest.fixture
def prior():
    return spark_dsg.DynamicSceneGraph.load(DSG_PATH)


def _edges(conn, place):
    rows = conn.query(
        f"MATCH (p:MeshPlace {{nodeSymbol: '{place}'}})-[r]-() RETURN count(r) AS n"
    )
    return rows[0]["n"]


def _a_place_with_neighbours(conn):
    rows = conn.query(
        "MATCH (p:MeshPlace)-[:MESH_PLACE_CONNECTED]->(q:MeshPlace) "
        "RETURN p.nodeSymbol AS p, collect(q.nodeSymbol) AS q LIMIT 1"
    )
    return rows[0]["p"], rows[0]["q"]


def test_rebuild_restores_a_removed_object_and_keeps_an_added_one(db, prior):
    removed = apply_changes(db, [(REMOVED, "O4", "", (0, 0, 0))])
    change_log.record_applied(db, "sget/chat", removed)
    added = add_object(db, "", "cone", (1.0, 2.0, 0.0))
    seq = change_log.record_applied(db, "sget/chat", [added])[0]
    assert object_center(db, "O4") is None

    change_log.deactivate(
        db, [e["seq"] for e in change_log.entries(db) if e["op"] == "remove_object"]
    )
    restored = change_log.rebuild(db, prior)

    assert object_center(db, "O4") is not None  # back from the prior
    assert object_center(db, added["symbol"]) == pytest.approx((1.0, 2.0, 0.0))
    assert [(a["kind"], a["symbol"]) for a in restored] == [(ADDED, "O4")]
    assert [e["seq"] for e in change_log.entries(db)] == [seq]
    db_to_spark_dsg(db)  # the rebuilt map can still be published


def test_rebuild_discards_console_edits_and_retired_copies(db, prior):
    original = object_center(db, "O0")
    out = graph_edit.apply_edit(
        db,
        [
            {"op": "update_node", "symbol": "O0", "set": {"center": [91, 92, 0]}},
            {
                "op": "create_node",
                "layer": "Object",
                "class": "crate",
                "pos": [93, 94, 0],
            },
            {"op": "retire_node", "symbol": "O2"},
        ],
        provenance={"source": "test"},
    )
    crate = next(c["symbol"] for c in out["changes"] if c["kind"] == "create")
    assert db.query("MATCH (n:Retired {nodeSymbol: 'O2'}) RETURN n")
    # Active logged changes must still replay while console edits disappear.
    logged = add_object(db, "", "cone", (95.0, 96.0, 0.0))
    change_log.record_applied(db, "sget/chat", [logged])

    change_log.rebuild(db, prior)

    assert object_center(db, "O0") == pytest.approx(original)
    assert object_center(db, crate) is None
    assert object_center(db, logged["symbol"]) == pytest.approx((95, 96, 0))
    assert not db.query("MATCH (n:Retired) RETURN n")
    nodes = db.query("MATCH (n {nodeSymbol: 'O2'}) RETURN labels(n) AS labels")
    assert [r["labels"] for r in nodes] == [["Object"]]
    assert db.query("MATCH (m:GraphMutation) RETURN m")
    db_to_spark_dsg(db)


@pytest.mark.parametrize("writer", ["scene_change", "rebuild"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_console_rejects_map_version_stale_after_another_writer(
    db, prior, writer, dry_run
):
    version = get_map_version(db)
    revision = graph_edit.graph_revision(db)
    if writer == "scene_change":
        with map_edit(db) as tx:
            applied = add_object(tx, "", "cone", (95.0, 96.0, 0.0))
            change_log.record_applied(tx, "awcd/test", [applied])
            notify_map_change(tx)
    else:
        change_log.rebuild(db, prior)
    assert graph_edit.graph_revision(db) == revision
    current = get_map_version(db)
    assert current > version
    ops = [{"op": "retire_node", "symbol": "O0"}]
    with pytest.raises(graph_edit.EditConflict, match="expected map version"):
        graph_edit.apply_edit(
            db,
            ops,
            provenance={"source": "test"},
            expected_revision=revision,
            expected_map_version=version,
            dry_run=dry_run,
        )
    assert object_center(db, "O0") is not None
    assert not db.query("MATCH (m:GraphMutation) RETURN m")
    assert graph_edit.graph_revision(db) == revision
    assert get_map_version(db) == current
    out = graph_edit.apply_edit(
        db,
        ops,
        provenance={"source": "test"},
        expected_revision=revision,
        expected_map_version=current,
        dry_run=dry_run,
    )
    assert out["map_version"] == (None if dry_run else current + 1)


def test_rebuild_restores_a_deleted_place_with_its_edges(db, prior):
    place, _ = _a_place_with_neighbours(db)
    edges = _edges(db, place)
    done = remove_place(db, place)
    change_log.record(db, "sget/chat", "remove_place", {"symbol": place})
    assert done["edges"] == edges and not place_exists(db, place)

    change_log.deactivate(db, [e["seq"] for e in change_log.entries(db)])
    change_log.rebuild(db, prior)

    assert place_exists(db, place) and _edges(db, place) == edges


def test_replayed_place_edits_survive_a_rebuild(db, prior):
    place, neighbours = _a_place_with_neighbours(db)
    added = add_place(db, "", 0.5, 0.5, neighbours[:2])
    change_log.record(db, "sget/chat", "add_place", added)
    remove_connection(db, place, neighbours[0])
    change_log.record(
        db, "sget/chat", "remove_connection", {"a": place, "b": neighbours[0]}
    )

    change_log.rebuild(db, prior)  # nothing undone: both edits replayed

    assert place_exists(db, added["symbol"])
    gone = db.query(
        f"MATCH (:MeshPlace {{nodeSymbol: '{place}'}})-[r:MESH_PLACE_CONNECTED]-"
        f"(:MeshPlace {{nodeSymbol: '{neighbours[0]}'}}) RETURN count(r) AS n"
    )[0]["n"]
    assert gone == 0
    db_to_spark_dsg(db)


def test_rebuild_keeps_holding_robots_and_bumps_the_version_once(db, prior):
    db.query(
        f"MATCH (r:Robot {{name: '{ROBOT}'}}), (o:Object {{nodeSymbol: 'O4'}}) "
        "MERGE (r)-[:HOLDS]->(o)"
    )
    v0 = get_map_version(db)
    change_log.rebuild(db, prior)
    assert get_holding(db) == {ROBOT: ["O4"]}
    assert get_map_version(db) == v0 + 1
    assert db.query(f"MATCH (r:Robot {{name: '{ROBOT}'}}) RETURN r.name AS n")
    assert not is_rebuilding(db)


def test_undone_object_symbols_are_not_reused(db, prior):
    first = add_object(db, "", "cone", (1.0, 1.0, 0.0))
    change_log.record_applied(db, "sget/chat", [first])
    change_log.deactivate(db, [e["seq"] for e in change_log.entries(db)])
    change_log.rebuild(db, prior)
    second = add_object(db, "", "cone", (1.0, 1.0, 0.0))
    assert second["symbol"] != first["symbol"]


# ---------- change detection snapshots through the writer ----------


class _Writer:
    """SceneChangeWriter's snapshot logic, without a ROS node."""

    _on_snapshot = SceneChangeWriter._on_snapshot
    _wanted = SceneChangeWriter._wanted
    _can_rebuild = SceneChangeWriter._can_rebuild
    _prior_problem = SceneChangeWriter._prior_problem
    _rebuild_log = SceneChangeWriter._rebuild_log
    _rebuild = SceneChangeWriter._rebuild
    _prior_graph = SceneChangeWriter._prior_graph
    _withdrawn_keys = SceneChangeWriter._withdrawn_keys
    _withdraw = SceneChangeWriter._withdraw
    _log_error = SceneChangeWriter._log_error
    _drop = SceneChangeWriter._drop
    on_reset = SceneChangeWriter.on_reset
    _clear_awcd = SceneChangeWriter._clear_awcd
    _announce = SceneChangeWriter._announce

    _prior_path = SceneChangeWriter._prior_path
    on_keep_detections = SceneChangeWriter.on_keep_detections

    def __init__(self, db, prior, withdraw_after_msgs=1, withdraw_after_s=0.0):
        self.db = db
        self.prior_param = ""  # use the path the fixture seeded from
        self._prior = (DSG_PATH, prior)
        self.min_confidence = 0.0
        self.snapshot_move_m = 0.3
        self.keep_match_m = 1.0
        self.snapshot_sources = ["awcd/"]
        self.withdraw_after_msgs = withdraw_after_msgs
        self.withdraw_after_s = withdraw_after_s
        self._missing = {}
        self.now = 0.0
        self._now = lambda: self.now
        self._awcd_clear = None
        self._lock = __import__("threading").Lock()
        self._queued = __import__("collections").deque()
        self.published = []
        self.applied_pub = NS(publish=self.published.append)
        self.get_logger = lambda: NS(
            info=lambda *_, **__: None,
            warn=lambda *_, **__: None,
            error=lambda *_, **__: None,
        )
        self.get_clock = lambda: NS(
            now=lambda: NS(
                to_msg=lambda: __import__(
                    "builtin_interfaces.msg", fromlist=["Time"]
                ).Time()
            )
        )


def _snapshot(removed=(), added=()):
    msg = SceneChangeMsg()
    msg.source = "awcd/reconciled"
    for sym in removed:
        msg.changes.append(SceneChange(kind=REMOVED, symbol=sym, confidence=1.0))
    for key, label, xyz in added:
        msg.changes.append(
            SceneChange(
                kind=ADDED,
                symbol=key,
                semantic_label=label,
                new_position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                confidence=1.0,
            )
        )
    return msg


def test_a_withdrawn_detection_is_undone(db, prior):
    w = _Writer(db, prior)
    w._on_snapshot(
        _snapshot(
            removed=["O4"], added=[("awcd:reconciled:7", "cone", (2.0, 1.0, 0.0))]
        )
    )
    assert object_center(db, "O4") is None
    added = [e for e in change_log.entries(db) if e["op"] == "add_object"][0]["data"][
        "symbol"
    ]
    assert object_center(db, added) is not None

    # Same set again: nothing to do.
    n = len(w.published)
    w._on_snapshot(
        _snapshot(
            removed=["O4"], added=[("awcd:reconciled:7", "cone", (2.0, 1.0, 0.0))]
        )
    )
    assert len(w.published) == n

    # AWCD reset: the set is empty, the map is the prior again.
    w._on_snapshot(_snapshot())
    assert object_center(db, "O4") is not None
    assert object_center(db, added) is None
    assert change_log.entries(db) == []
    kinds = {(c.kind, c.symbol) for c in w.published[-1].changes}
    assert kinds == {(ADDED, "O4"), (REMOVED, added)}


def test_kept_detections_survive_a_change_detection_reset(db, prior):
    w = _Writer(db, prior)
    gone = db.query("MATCH (o:Object) RETURN o.nodeSymbol AS s ORDER BY s LIMIT 1")[0][
        "s"
    ]
    w._on_snapshot(
        _snapshot(
            removed=[gone], added=[("awcd:reconciled:7", "cone", (2.0, 1.0, 0.0))]
        )
    )
    added = [e for e in change_log.entries(db) if e["op"] == "add_object"][0]["data"][
        "symbol"
    ]
    response = w.on_keep_detections(None, NS(success=False, message=""))
    assert response.success and "Kept 2" in response.message
    assert {e["source"] for e in change_log.entries(db)} == {"kept/awcd/reconciled"}

    # A robot reset later still reports them: nothing is applied twice.
    n = len(w.published)
    w._on_snapshot(
        _snapshot(
            removed=[gone], added=[("awcd:reconciled:7", "cone", (2.2, 1.0, 0.0))]
        )
    )
    assert len(w.published) == n and len(change_log.entries(db)) == 2

    # The detectors' reset: an empty set withdraws nothing.
    w._on_snapshot(_snapshot())
    assert object_center(db, gone) is None
    assert object_center(db, added) is not None

    # A rebuild (an undo elsewhere) replays the kept changes.
    change_log.rebuild(db, prior)
    assert object_center(db, gone) is None
    assert object_center(db, added) is not None

    # New detections after the reset are change detection's again.
    w._on_snapshot(_snapshot(added=[("awcd:reconciled:7", "cone", (9.0, 1.0, 0.0))]))
    assert [e["source"] for e in change_log.entries(db)][-1] == "awcd/reconciled"


def test_a_detection_reported_elsewhere_moves(db, prior):
    w = _Writer(db, prior)
    w._on_snapshot(_snapshot(added=[("awcd:reconciled:7", "cone", (2.0, 1.0, 0.0))]))
    w._on_snapshot(_snapshot(added=[("awcd:reconciled:7", "cone", (4.0, 1.0, 0.0))]))
    entry = change_log.entries(db)[0]
    assert object_center(db, entry["data"]["symbol"]) == pytest.approx((4.0, 1.0, 0.0))
    assert len(change_log.entries(db)) == 1


def test_detections_do_not_undo_map_edits(db, prior):
    w = _Writer(db, prior)
    place, _ = _a_place_with_neighbours(db)
    remove_place(db, place)
    change_log.record(db, "sget/chat", "remove_place", {"symbol": place})
    w._on_snapshot(_snapshot(removed=["O4"]))
    w._on_snapshot(_snapshot())  # withdrawn: rebuild
    assert object_center(db, "O4") is not None
    assert not place_exists(db, place)  # the edit is replayed


def test_an_object_a_robot_placed_stays_there_after_a_rebuild(db, prior):
    from test_map_state import _holding_call

    assert _holding_call(db, True, "O4")
    assert _holding_call(db, False, "O4", position=(3.0, -1.0, 0.0))
    change_log.rebuild(db, prior)
    assert object_center(db, "O4") == pytest.approx((3.0, -1.0, 0.0))
    assert get_holding(db) == {}


def test_the_writer_rebuilds_only_from_the_file_neo4j_was_seeded_from(db, prior):
    import os

    assert get_prior_path(db) == os.path.abspath(DSG_PATH)
    w = _Writer(db, prior)
    assert w._can_rebuild()
    # filled some other way: neither the seed's path nor the loader's source
    db.query("MATCH (m:_MapState) REMOVE m.prior_path")
    db.query("MATCH (m:_GraphMetadata {key: 'source'}) REMOVE m.file_path")
    assert not w._can_rebuild()
    w.prior_param = DSG_PATH  # unless set explicitly
    assert w._can_rebuild()


def test_an_edit_and_its_log_entry_commit_together_in_map_edit(db):
    """Found in the local e2e: the first SGET edit failed with "Transaction
    failed", because logging it tried to create the log's seq constraint
    (schema) inside the edit's write transaction, which Neo4j refuses."""
    from heracles_ros.map_state import map_edit, notify_map_change

    db.execute("DROP CONSTRAINT scene_change_seq IF EXISTS")
    fresh = _connect()  # a new connection: it has not made the schema yet
    v = get_map_version(fresh)
    with map_edit(fresh) as tx:
        assert tx is not None
        # as SGET's remove_scene_place logs it
        place = _a_place_with_neighbours(tx)[0]
        assert remove_place(tx, place) is not None
        change_log.record(tx, "sget/test", "remove_place", {"symbol": place})
        notify_map_change(tx)
    fresh.close()
    assert get_map_version(db) == v + 1
    (entry,) = change_log.entries(db)
    assert entry["op"] == "remove_place" and entry["source"] == "sget/test"
    rows = db.query(
        "SHOW CONSTRAINTS YIELD name WHERE name = 'scene_change_seq' RETURN name"
    )
    assert rows  # made, outside the edit


# ---------- withdrawn detections: undone in place, after a grace period ----------


def _no_rebuild(monkeypatch):
    def fail(*a, **k):
        pytest.fail("a withdrawn detection must not rebuild the map")

    monkeypatch.setattr(change_log, "rebuild", fail)


def _node(db, sym):
    rows = db.query(
        f"MATCH (n {{nodeSymbol: '{sym}'}}) OPTIONAL MATCH (p)-[:CONTAINS]->(n) "
        "RETURN labels(n) AS l, properties(n) AS p, collect(p.nodeSymbol) AS parents"
    )
    return rows[0] if rows else None


def _moved(msg, sym, xyz):
    msg.changes.append(
        SceneChange(
            kind=2,  # MOVED
            symbol=sym,
            new_position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
            confidence=1.0,
        )
    )
    return msg


def test_a_withdrawn_removal_comes_back_from_the_prior_without_a_rebuild(
    db, prior, monkeypatch
):
    _no_rebuild(monkeypatch)
    before = _node(db, "O4")
    # Written around the change log (as a console edit is): a rebuild drops it.
    other = add_object(db, "O990", "crate", (5.0, 5.0, 0.0))
    w = _Writer(db, prior)
    w._on_snapshot(_snapshot(removed=["O4"]))
    assert object_center(db, "O4") is None
    v = get_map_version(db)

    w._on_snapshot(_snapshot())

    after = _node(db, "O4")
    assert after["l"] == before["l"] and after["parents"] == before["parents"]
    assert after["p"] == before["p"]  # same attributes, position, bbox
    assert object_center(db, other["symbol"]) is not None  # no rebuild
    assert get_map_version(db) == v + 1
    assert w.published[-1].map_version == v + 1
    assert [(c.kind, c.symbol) for c in w.published[-1].changes] == [(ADDED, "O4")]
    assert change_log.entries(db) == []


def test_a_withdrawn_addition_is_deleted_without_a_rebuild(db, prior, monkeypatch):
    _no_rebuild(monkeypatch)
    w = _Writer(db, prior)
    w._on_snapshot(_snapshot(added=[("awcd:reconciled:3", "car", (2.0, 1.0, 0.0))]))
    (entry,) = change_log.entries(db)
    added = entry["data"]["symbol"]
    assert object_center(db, added) is not None
    v = get_map_version(db)

    w._on_snapshot(_snapshot())

    assert object_center(db, added) is None
    assert get_map_version(db) == v + 1
    assert [(c.kind, c.symbol) for c in w.published[-1].changes] == [(REMOVED, added)]


def test_a_withdrawn_move_goes_back_without_a_rebuild(db, prior, monkeypatch):
    _no_rebuild(monkeypatch)
    original = object_center(db, "O4")
    w = _Writer(db, prior)
    w._on_snapshot(_moved(_snapshot(), "O4", (7.0, 8.0, 0.0)))
    assert object_center(db, "O4") == pytest.approx((7.0, 8.0, 0.0))

    w._on_snapshot(_snapshot())

    assert object_center(db, "O4") == pytest.approx(original)
    assert [c.symbol for c in w.published[-1].changes] == ["O4"]
    assert w.published[-1].changes[0].kind == 2  # MOVED


def test_a_withdrawn_removal_of_an_object_not_in_the_prior_comes_back_as_logged(
    db, prior, monkeypatch
):
    _no_rebuild(monkeypatch)
    added = add_object(db, "", "cone", (3.0, 3.0, 0.0))
    change_log.record_applied(db, "sget/chat", [added])
    w = _Writer(db, prior)
    w._on_snapshot(_snapshot(removed=[added["symbol"]]))
    assert object_center(db, added["symbol"]) is None

    w._on_snapshot(_snapshot())

    assert object_center(db, added["symbol"]) == pytest.approx((3.0, 3.0, 0.0))
    assert db.query(
        f"MATCH (o:Object {{nodeSymbol: '{added['symbol']}'}}) RETURN o.class AS c"
    ) == [{"c": "cone"}]


def test_a_flickering_detection_writes_nothing_until_it_stays_away(
    db, prior, monkeypatch
):
    _no_rebuild(monkeypatch)
    w = _Writer(db, prior, withdraw_after_msgs=3, withdraw_after_s=5.0)
    w._on_snapshot(_snapshot(removed=["O4"]))
    v, n = get_map_version(db), len(w.published)

    # Dropped from two messages, reported again: nothing is written.
    for t in (1.0, 2.0):
        w.now = t
        w._on_snapshot(_snapshot())
    w.now = 3.0
    w._on_snapshot(_snapshot(removed=["O4"]))  # back: not applied a second time
    assert object_center(db, "O4") is None
    assert get_map_version(db) == v and len(w.published) == n
    assert len(change_log.entries(db)) == 1

    # Three messages, but only 2 s: still not withdrawn.
    for t in (4.0, 5.0, 6.0):
        w.now = t
        w._on_snapshot(_snapshot())
    assert object_center(db, "O4") is None and get_map_version(db) == v
    # Long enough as well: withdrawn, undone once.
    w.now = 9.5
    w._on_snapshot(_snapshot())
    assert object_center(db, "O4") is not None
    assert get_map_version(db) == v + 1 and len(w.published) == n + 1
    assert change_log.entries(db) == []


# ---------- a rebuild keeps the console's mission-prep files ----------


def _apply_prep_file(db, name, ops, seq, via_console=True):
    """Apply ops as the adt4 console applies a mission-prep file, and record it."""
    import json

    from heracles_ros.map_state import get_map_epoch

    mutation = None
    if via_console:
        out = graph_edit.apply_edit(
            db, ops, provenance={"via": "mission prep (test)"}, summary=name
        )
        mutation = out["mutation_id"]
    rec = {
        "name": name,
        "summary": name,
        "hash": "x",
        "epoch": get_map_epoch(db),
        "t": 0.0,
        "via": "test",
        "counts": "",
        "seq": seq,
        "file_json": json.dumps({"name": name, "intent": [], "ops": ops}),
    }
    if mutation:
        rec["mutation_id"] = mutation
    db.execute("CREATE (r:_PrepApplied) SET r = $p", p=rec)
    return mutation


def _mesh_places(db, n):
    rows = db.query(
        "MATCH (p:MeshPlace) RETURN p.nodeSymbol AS s, p.center AS c ORDER BY s "
        f"LIMIT {n}"
    )
    return [(r["s"], (r["c"].x, r["c"].y, r["c"].z)) for r in rows]


def _prep_ops(db):
    (p1, c1), (p2, _), (p3, _), (p4, _) = _mesh_places(db, 4)
    ops = [
        {"op": "delete_node", "symbol": "O0"},
        {"op": "delete_node", "symbol": "O2"},
        {"op": "create_node", "layer": "Object", "symbol": "O900", "class": "cone",
         "pos": [c1[0], c1[1], c1[2] + 0.35], "size": 0.5, "source": "mission_prep"},
        {"op": "create_edge", "from": p1, "to": "O900", "type": "CONTAINS"},
        {"op": "retire_node", "symbol": p2},
        {"op": "group", "parent_layer": "Room", "symbol": "R900", "class": "base camp",
         "name": "base camp", "members": [p3, p4], "member_layers": ["MeshPlace"]},
    ]  # fmt: skip
    return ops, (p1, p2, p3, p4)


def _assert_prep_applied(db, places, moved=False):
    p1, p2, p3, p4 = places
    assert object_center(db, "O0") is None and object_center(db, "O2") is None
    if not moved:  # (a logged move relinks it to the place nearest its new spot)
        assert _node(db, "O900")["parents"] == [p1]
    assert _node(db, p2)["l"] == ["Retired"]
    room = db.query(
        "MATCH (r:Room {nodeSymbol: 'R900'})-[:CONTAINS]->(p:MeshPlace) "
        "RETURN r.name AS name, collect(p.nodeSymbol) AS members"
    )
    assert room[0]["name"] == "base camp"
    assert sorted(room[0]["members"]) == sorted([p3, p4])


def test_a_reset_reapplies_the_mission_prep_files(db, prior):
    from heracles_ros.map_state import get_map_epoch

    ops, places = _prep_ops(db)
    mutation = _apply_prep_file(db, "mission_4", ops, seq=1)
    # A file whose first op names an object the map does not have: skipped,
    # the rest of the file still applies. Not applied by a console: the ops
    # come from the file.
    p5 = _mesh_places(db, 5)[4][0]
    _apply_prep_file(
        db,
        "stale",
        [
            {"op": "delete_node", "symbol": "O77777"},
            {"op": "retire_node", "symbol": p5},
        ],
        seq=2,
        via_console=False,
    )
    detection = add_object(db, "", "car", (2.0, 1.0, 0.0))
    change_log.record_applied(db, "awcd/reconciled", [detection], "awcd:reconciled:1")
    epoch = get_map_epoch(db)
    mutations = len(db.query("MATCH (m:GraphMutation) RETURN m"))

    w = _Writer(db, prior)
    response = w.on_reset(None, NS(success=False, message=""))

    assert response.success, response.message
    assert object_center(db, detection["symbol"]) is None  # the reset dropped it
    _assert_prep_applied(db, places)
    assert _node(db, p5)["l"] == ["Retired"]
    assert get_map_epoch(db) == epoch
    assert len(db.query("MATCH (r:_PrepApplied) RETURN r")) == 2
    assert len(db.query("MATCH (m:GraphMutation) RETURN m")) == mutations
    db_to_spark_dsg(db)

    # The console's undo of the file still applies to the rebuilt map.
    graph_edit.undo_edit(db, mutation, provenance={"via": "test"})
    assert object_center(db, "O0") is not None and object_center(db, "O900") is None
    assert _node(db, places[1])["l"] == ["MeshPlace"]


def test_a_rebuild_replays_prep_files_before_the_log_and_logs_failures(db, prior):
    ops, places = _prep_ops(db)
    _apply_prep_file(db, "mission_4", ops, seq=1)
    # A logged change on the object the file added: replays after the file.
    change_log.record(
        db, "sget/chat", "move_object", {"symbol": "O900", "xyz": [9.0, 9.0, 0.0]}
    )
    move = __import__("heracles_ros.scene_change", fromlist=["x"]).move_object
    move(db, "O900", (9.0, 9.0, 0.0))
    # A file from an earlier epoch (a reseed since) is not applied.
    db.execute(
        "CREATE (:_PrepApplied {name: 'old', epoch: 'gone', seq: 9, file_json: $f})",
        f='{"ops": [{"op": "delete_node", "symbol": "O4"}]}',
    )
    # A file the console's own undo took back is not applied.
    undone = _apply_prep_file(
        db, "undone", [{"op": "delete_node", "symbol": "O6"}], seq=3
    )
    graph_edit.undo_edit(db, undone, provenance={"via": "test"})
    logged = []

    change_log.rebuild(db, prior, log=logged.append)

    _assert_prep_applied(db, places, moved=True)
    assert object_center(db, "O900") == pytest.approx((9.0, 9.0, 0.0))
    assert object_center(db, "O4") is not None
    assert object_center(db, "O6") is not None
    text = "\n".join(logged)
    assert "re-applied mission-prep file 'mission_4'" in text
    assert "'undone' was undone" in text


def test_the_map_is_rebuilt_only_if_undoing_in_place_fails(db, prior, monkeypatch):
    w = _Writer(db, prior)
    w._on_snapshot(_snapshot(removed=["O4"]))

    def broken(*a, **k):
        raise RuntimeError("lost the connection")

    monkeypatch.setattr(change_log, "revert", broken)
    rebuilds = []
    real = change_log.rebuild
    monkeypatch.setattr(
        change_log, "rebuild", lambda *a, **k: rebuilds.append(1) or real(*a, **k)
    )
    w._on_snapshot(_snapshot())
    assert rebuilds == [1]
    assert object_center(db, "O4") is not None
    assert [(c.kind, c.symbol) for c in w.published[-1].changes] == [(ADDED, "O4")]
