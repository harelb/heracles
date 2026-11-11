"""The change log and the writer under failure, without a database.

The Neo4j-backed behaviour is in test_change_log; here the database is faked
(or the functions that talk to it are), so the failure paths run anywhere.
"""

from types import SimpleNamespace as NS

import pytest
from heracles import graph_interface

from heracles_ros import change_log
from heracles_ros import scene_change_writer_node as writer_node
from heracles_ros.scene_change_writer_node import SceneChangeWriter

# ---------- change_log.record: seq clashes ----------


class ConstraintError(Exception):
    """Named like neo4j.exceptions.ConstraintError."""


class _LogDb:
    def __init__(self, clashes=0):
        self.clashes = clashes
        self.schema = []
        self.creates = 0

    def execute(self, query, **kw):
        if "CREATE CONSTRAINT" in query:
            self.schema.append(query)
            return [], None, None
        self.creates += 1
        if self.clashes:
            self.clashes -= 1
            raise ConstraintError("Node(12) already exists with label `_SceneChange`")
        return [{"seq": 5}], None, None


def test_record_makes_seq_unique_once_and_retries_a_clash():
    db = _LogDb(clashes=2)
    assert change_log.record(db, "sget/chat", "remove_place", {"symbol": "t1"}) == 5
    assert db.creates == 3
    change_log.record(db, "sget/chat", "remove_place", {"symbol": "t2"})
    assert len(db.schema) == 1 and "REQUIRE c.seq IS UNIQUE" in db.schema[0]


def test_record_does_not_retry_other_errors():
    class _Broken(_LogDb):
        def execute(self, query, **kw):
            if "CREATE CONSTRAINT" in query:
                return [], None, None
            raise RuntimeError("connection lost")

    with pytest.raises(RuntimeError):
        change_log.record(_Broken(), "sget/chat", "remove_place", {"symbol": "t1"})


# ---------- change_log.rebuild: what it drops, and failing replays ----------


class _RebuildDb:
    """The map as symbols; rebuild's own queries are patched below."""

    atomic_queries = False

    def __init__(self):
        self.symbols = {"o1", "o2", "o77", "t5"}  # o77: written around the log
        self.links = 4  # e.g. Object -> Observation
        self.state = {"version": 3}
        self.marks = []

    def execute(self, query, **kw):
        if "DETACH DELETE" in query:
            self.symbols, self.links = set(), 0
        elif "rebuilding_since = $t" in query:
            self.marks.append("mark")
        elif "rebuild_failed = $why" in query:
            self.marks.append("failed")
        elif "rebuilding_since = null" in query:
            self.marks.append("unmark")
        elif "SET m.version = CASE" in query:
            self.state["version"] = max(self.state["version"], kw["old"]) + 1
        return [], None, None

    def query(self, query):
        if "m.version AS v" in query:
            return [{"v": self.state["version"]}]
        return []


@pytest.fixture
def rebuild_db(monkeypatch):
    db = _RebuildDb()
    entries = [
        {"seq": 1, "op": "remove_object", "data": {"symbol": "O2"}, "active": True},
        {"seq": 2, "op": "add_object", "data": {"symbol": "O9"}, "active": True},
        {"seq": 3, "op": "add_object", "data": {"symbol": "O8"}, "active": False},
    ]
    monkeypatch.setattr(change_log, "_load_prior", lambda prior: prior)
    monkeypatch.setattr(change_log, "objects", lambda d: {})
    monkeypatch.setattr(change_log, "_map_symbols", lambda d: set(d.symbols))
    monkeypatch.setattr(change_log, "_outside_links", lambda d: d.links)
    monkeypatch.setattr(change_log, "_image_store", lambda d: "")
    monkeypatch.setattr(
        change_log,
        "entries",
        lambda d, active_only=True, source=None: [
            e for e in entries if e["active"] or not active_only
        ],
    )

    def load(graph, d, **kw):
        d.symbols = {"o1", "o2", "t5"}
        d.loaded_with = kw

    monkeypatch.setattr(graph_interface, "spark_dsg_to_db", load)
    return db


def test_a_failing_replay_is_skipped_and_what_was_dropped_is_logged(
    rebuild_db, monkeypatch
):
    def apply_op(d, op, data):
        if op == "add_object":
            raise ValueError("no MeshPlace near it")
        d.symbols.discard(data["symbol"].lower())
        return {}

    monkeypatch.setattr(change_log, "apply_op", apply_op)
    logged = []
    change_log.rebuild(rebuild_db, "prior", log=logged.append)
    text = "\n".join(logged)
    assert "1 change(s) failed to replay" in text and "#2 add_object" in text
    # o77 was neither in the prior nor in the log; o8 was undone, o2 removed.
    assert "dropped 1 map node(s)" in text and "o77" in text
    assert "o8" not in text.split("dropped")[1]
    assert "lost 4 link(s)" in text
    assert rebuild_db.marks[-1] == "unmark" and rebuild_db.state["version"] == 4


def test_a_rebuild_whose_prior_fails_to_load_leaves_the_map_held(
    rebuild_db, monkeypatch
):
    def broken(graph, d, **kw):
        raise OSError("truncated file")

    monkeypatch.setattr(graph_interface, "spark_dsg_to_db", broken)
    with pytest.raises(OSError):
        change_log.rebuild(rebuild_db, "prior", log=lambda *_: None)
    assert rebuild_db.marks[-1] == "failed"
    assert rebuild_db.state["version"] == 3  # not announced as a new map


def test_the_recorded_image_store_is_passed_when_the_loader_takes_one(
    rebuild_db, monkeypatch
):
    monkeypatch.setattr(change_log, "_image_store", lambda d: "/maps/k/image_store")
    monkeypatch.setattr(change_log, "apply_op", lambda d, op, data: {})

    def load(graph, d, source_file_path=None, image_store=None):
        d.symbols = {"o1", "o2", "t5"}
        d.loaded_with = image_store

    monkeypatch.setattr(graph_interface, "spark_dsg_to_db", load)
    change_log.rebuild(rebuild_db, "prior", log=lambda *_: None)
    assert rebuild_db.loaded_with == "/maps/k/image_store"


def test_an_image_store_the_loader_cannot_take_is_reported(rebuild_db, monkeypatch):
    monkeypatch.setattr(change_log, "_image_store", lambda d: "/maps/k/image_store")
    monkeypatch.setattr(change_log, "apply_op", lambda d, op, data: {})
    logged = []
    change_log.rebuild(rebuild_db, "prior", log=logged.append)
    assert rebuild_db.loaded_with == {}
    assert any("image store /maps/k/image_store" in line for line in logged)


# ---------- the writer: queueing, refusals, errors ----------


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, m, **_):
        self.lines.append(("info", m))

    def warn(self, m, **_):
        self.lines.append(("warn", m))

    def error(self, m, **_):
        self.lines.append(("error", m))


class _Writer:
    """SceneChangeWriter's callbacks, without a ROS node."""

    on_changes = SceneChangeWriter.on_changes
    _apply = SceneChangeWriter._apply
    _queue = SceneChangeWriter._queue
    _drain_queued = SceneChangeWriter._drain_queued
    _log_error = SceneChangeWriter._log_error
    _answering = SceneChangeWriter._answering
    _drop = SceneChangeWriter._drop
    _prior_path = SceneChangeWriter._prior_path
    _prior_problem = SceneChangeWriter._prior_problem
    _can_rebuild = SceneChangeWriter._can_rebuild
    on_undo_last_edit = SceneChangeWriter.on_undo_last_edit

    def __init__(self, prior=""):
        import threading
        from collections import deque

        self.db = object()
        self._lock = threading.Lock()
        self._queued = deque()
        self.prior_param = prior
        self.snapshot_sources = ["awcd/"]
        self.applied = []
        self.logger = _Logger()
        self.get_logger = lambda: self.logger

    def _on_snapshot(self, msg):
        self.applied.append(msg)

    _on_new_changes = _on_snapshot


def _msg(source="awcd/testbot_b"):
    return NS(source=source, changes=[])


def test_changes_during_a_reload_are_queued_then_applied(monkeypatch):
    rebuilding = [True]
    monkeypatch.setattr(writer_node, "is_rebuilding", lambda db: rebuilding[0])
    w = _Writer()
    w.on_changes(_msg("awcd/testbot_b"))
    w.on_changes(_msg("executor/testbot_a"))
    assert w.applied == [] and len(w._queued) == 2
    w._drain_queued()  # still reloading: wait
    assert w.applied == []
    rebuilding[0] = False
    w.on_changes(_msg("sget/chat"))  # behind the queue, not before it
    w._drain_queued()
    assert [m.source for m in w.applied] == [
        "awcd/testbot_b",
        "executor/testbot_a",
        "sget/chat",
    ]


def test_a_failing_message_is_logged_and_the_writer_goes_on(monkeypatch):
    monkeypatch.setattr(writer_node, "is_rebuilding", lambda db: False)
    w = _Writer()

    def boom(msg):
        raise RuntimeError("Neo4j went away")

    w._on_snapshot = w._on_new_changes = boom
    w.on_changes(_msg())  # does not raise
    assert any(k == "error" and "Neo4j went away" in m for k, m in w.logger.lines)


def test_a_failing_service_answers_instead_of_dying():
    w = _Writer()

    def broken(request, response):
        raise RuntimeError("Neo4j went away")

    response = w._answering("reset", broken)(None, NS(success=True, message=""))
    assert response.success is False and "Neo4j went away" in response.message


def test_a_missing_prior_file_is_refused_before_anything_is_deleted(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writer_node, "rebuild_state", lambda db: (False, ""))
    w = _Writer(prior=str(tmp_path / "gone.json"))
    w._rebuild = lambda: pytest.fail("must not rebuild")
    response = w._drop("scene_graph/reset", [], NS(success=True, message=""), "x")
    assert response.success is False and "does not exist" in response.message
    assert ("warn", f"scene_graph/reset: {response.message}") in w.logger.lines


def test_undo_and_reset_wait_for_a_reload_someone_else_runs(tmp_path, monkeypatch):
    prior = tmp_path / "dsg.json"
    prior.write_text("{}")
    monkeypatch.setattr(writer_node, "rebuild_state", lambda db: (True, ""))
    w = _Writer(prior=str(prior))
    w._rebuild = lambda: pytest.fail("must not rebuild")
    response = w._drop("scene_graph/reset", [], NS(success=True, message=""), "x")
    assert response.success is False and "being reloaded" in response.message


def test_nothing_to_undo_is_logged(monkeypatch):
    monkeypatch.setattr(writer_node.change_log, "entries", lambda db: [])
    w = _Writer()
    response = w.on_undo_last_edit(None, NS(success=True, message=""))
    assert response.success is False
    assert ("warn", "undo_last_edit: There is no map edit to undo.") in w.logger.lines
