"""Reloads keep the map's own state, without a database.

A small fake answers the Cypher heracles.map_reload sends, so the order of
what a load does -- mark, wipe, re-mark, load, restore, unmark -- is checked
without the Neo4j test instance.
"""

import pytest
from heracles import map_reload
from heracles import utils as heracles_utils
from heracles.map_reload import (
    get_map_epoch,
    get_map_version,
    is_rebuilding,
    rebuild_state,
    reloading_map,
)

from heracles_ros.map_state import METADATA_KEY, map_metadata


class _Db:
    """_MapState, Robot nodes, HOLDS and Object symbols, answering map_reload's Cypher."""

    atomic_queries = False  # no refresh thread in tests

    def __init__(self, version=0, robots=None, holding=None, objects=()):
        self.state = {"version": version} if version else None
        self.robots = dict(robots or {})
        self.holding = {r: list(o) for r, o in (holding or {}).items()}
        self.objects = set(objects)
        self.log = []  # what happened, in order

    def _ms(self):
        if self.state is None:
            self.state = {}
        return self.state

    def wipe(self, keep_robots=True):
        """A full load's wipe; initialize_db(keep_labels=KEEP_LABELS) keeps robots."""
        self.log.append("wipe")
        self.objects = set()
        self.holding = {}
        if not keep_robots:
            self.state = None
            self.robots = {}

    def execute(self, query, **kw):
        rows = []
        if "SET m.rebuilding_since = $t" in query:
            self._ms()["rebuilding_since"] = kw["t"]
            self.log.append("mark")
        elif "SET m.rebuilding_since = null" in query:
            self._ms().pop("rebuilding_since", None)
            self._ms().pop("rebuild_failed", None)
            self.log.append("unmark")
        elif "m.rebuild_failed = $why" in query:
            self._ms()["rebuild_failed"] = kw["why"]
            self._ms().pop("rebuilding_since", None)
            self.log.append("failed")
        elif "MERGE (r:Robot {name: $n})" in query:
            self.robots.setdefault(kw["n"], dict(kw["p"]))
        elif "MERGE (r)-[:HOLDS]->(o)" in query:
            if kw["r"] in self.robots:
                held = [o for o in kw["o"] if o in self.objects]
                if held:
                    self.holding.setdefault(kw["r"], []).extend(held)
                rows = [{"o": o} for o in held]
        elif "SET m.version = CASE" in query:
            ms = self._ms()
            ms["version"] = max(ms.get("version", 0), kw["old"]) + 1
            self.log.append(f"version {ms['version']}")
        elif "SET m.epoch" in query:
            self._ms()["epoch"] = kw["e"]
        elif "SET m.prior_path" in query:
            self._ms()["prior_path"] = kw["p"]
        elif "REMOVE m.prior_path" in query:
            if self.state:
                self.state.pop("prior_path", None)
        else:
            raise AssertionError(f"unexpected Cypher: {query}")
        return rows, None, None

    def query(self, query):
        ms = self.state
        if "m.rebuilding_since AS t" in query:
            return (
                [{"t": ms.get("rebuilding_since"), "f": ms.get("rebuild_failed")}]
                if ms is not None
                else []
            )
        if "m.version AS v" in query:
            return [{"v": ms.get("version")}] if ms is not None else []
        if "m.epoch AS e" in query:
            return [{"e": ms.get("epoch")}] if ms is not None else []
        if "properties(r)" in query:
            return [{"n": n, "p": p} for n, p in self.robots.items()]
        if "HOLDS" in query:
            return [
                {"r": r, "o": o, "robot": r, "object": o}
                for r, held in sorted(self.holding.items())
                for o in held
            ]
        raise AssertionError(f"unexpected query: {query}")


def _load(db, objects, fail=False, keep_robots=True):
    with reloading_map(db, "test load", log=lambda *_: None) as reload:
        db.wipe(keep_robots)
        reload.wiped()
        if fail:
            raise RuntimeError("bad scene graph")
        db.objects = set(objects)
        db.log.append("load")


def test_a_reload_is_held_from_start_to_end_and_counts_the_version_on():
    db = _Db(version=7, robots={"testbot_b": {"name": "testbot_b"}}, objects={"O1"})
    _load(db, {"O1", "O2"})
    assert db.log == ["mark", "wipe", "mark", "load", "version 8", "unmark"]
    assert get_map_version(db) == 8 and not is_rebuilding(db)


def test_every_full_load_gets_a_new_epoch():
    db = _Db(version=3, objects={"O1"})
    _load(db, {"O1"})
    first = get_map_epoch(db)
    _load(db, {"O1"})
    assert len(first) == 32 and get_map_epoch(db) not in ("", first)
    assert get_map_version(db) == 5


def test_a_rebuild_of_the_same_map_keeps_its_epoch():
    db = _Db(version=3, objects={"O1"})
    _load(db, {"O1"})
    epoch = get_map_epoch(db)
    with reloading_map(db, "rebuild", log=lambda *_: None, full=False) as reload:
        reload.wiped()
    assert get_map_epoch(db) == epoch and get_map_version(db) == 5


def test_holding_and_robots_survive_a_wipe_that_took_them():
    pose = {"name": "testbot_a", "qw": 1.0}
    db = _Db(
        version=2,
        robots={"testbot_a": pose, "testbot_b": {"name": "testbot_b"}},
        holding={"testbot_a": ["O58"], "testbot_b": ["O9"]},
        objects={"O58", "O9"},
    )
    logged = []
    with reloading_map(db, "console reset", log=logged.append) as reload:
        db.wipe(keep_robots=False)  # a loader that deletes everything
        reload.wiped()
        db.objects = {"O58"}  # the new map has no O9
    assert db.robots["testbot_a"] == pose and "testbot_b" in db.robots
    assert db.holding == {"testbot_a": ["O58"]}
    assert any("testbot_b:O9" in line for line in logged)
    assert get_map_version(db) == 3  # max(old) + 1, even though _MapState was wiped


def test_a_load_failing_after_the_wipe_leaves_the_map_held():
    db = _Db(version=4, objects={"O1"})
    with pytest.raises(RuntimeError):
        _load(db, set(), fail=True)
    running, failed = rebuild_state(db)
    assert not running and "bad scene graph" in failed
    assert is_rebuilding(db)  # never stale: the publisher keeps the last map
    _load(db, {"O1"})  # a later load that works clears it
    assert not is_rebuilding(db) and get_map_version(db) == 5


def test_a_load_failing_before_the_wipe_changes_nothing():
    db = _Db(version=4, objects={"O1"})
    with pytest.raises(ValueError):
        with reloading_map(db, "test", log=lambda *_: None):
            raise ValueError("file not found")
    assert not is_rebuilding(db) and get_map_version(db) == 4


def test_an_abandoned_mark_goes_stale(monkeypatch):
    db = _Db(version=1)
    map_reload.set_rebuilding(db, True)
    assert is_rebuilding(db)
    db.state["rebuilding_since"] -= map_reload.REBUILD_STALE_S + 1
    assert not is_rebuilding(db)


def test_the_published_metadata_carries_the_epoch():
    db = _Db(version=2, objects={"O1"})
    _load(db, {"O1"})
    state = map_metadata(db)[METADATA_KEY]
    assert state["map_epoch"] == get_map_epoch(db) and state["map_version"] == 3
    assert isinstance(state["map_epoch"], str)


def test_load_dsg_to_db_holds_the_map_and_records_its_file(monkeypatch):
    db = _Db(version=6, robots={"testbot_d": {"name": "testbot_d"}}, objects={"O1"})

    class _Wrapper:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return db

        def __exit__(self, *a):
            pass

    def initialize_db(d, keep_labels=()):
        assert set(keep_labels) == {"_MapState", "Robot"}
        assert is_rebuilding(d)  # marked before anything is deleted
        d.wipe(keep_robots=True)
        d.state = None  # the mark must not depend on _MapState surviving

    def spark_dsg_to_db(graph, d, source_file_path=None):
        assert is_rebuilding(d)  # re-marked right after the wipe
        d.objects = {"O1", "O2"}
        d.log.append("load")

    monkeypatch.setattr(heracles_utils, "Neo4jWrapper", _Wrapper)
    monkeypatch.setattr(heracles_utils, "initialize_db", initialize_db)
    monkeypatch.setattr(heracles_utils, "spark_dsg_to_db", spark_dsg_to_db)
    graph = type("G", (), {"metadata": type("M", (), {"add": lambda s, m: None})()})()
    heracles_utils.load_dsg_to_db("neo4j://x", ("u", "p"), graph, "maps/dsg.json")
    assert db.state["prior_path"].endswith("maps/dsg.json")
    assert get_map_version(db) == 7 and len(get_map_epoch(db)) == 32
    assert not is_rebuilding(db) and "testbot_d" in db.robots


def test_a_long_load_keeps_renewing_its_mark(monkeypatch):
    import time

    monkeypatch.setattr(map_reload, "REBUILD_REFRESH_S", 0.01)
    db = _Db(version=1)
    db.atomic_queries = True
    with reloading_map(db, "big map", log=lambda *_: None) as reload:
        reload.wiped()
        time.sleep(0.2)
    assert db.log.count("mark") > 3 and db.log[-1] == "unmark"
    assert not is_rebuilding(db)


class _Tx:
    """A driver, session and transaction recording what an edit ran."""

    def __init__(self, rebuilding_since=None):
        self.ran, self.committed = [], False
        self.since = rebuilding_since
        self.driver = self
        self.db_name = "neo4j"

    def session(self, database):
        return self

    def begin_transaction(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def commit(self):
        self.committed = True

    def run(self, query, parameters=None, **kw):
        self.ran.append(query)
        if "rebuilding_since AS t" in query:
            return [_Record({"t": self.since, "f": None})]
        return []


class _Record(dict):
    def data(self):
        return dict(self)


def test_an_edit_locks_the_map_state_then_commits_in_one_transaction():
    from heracles_ros.map_state import map_edit

    tx = _Tx()
    with map_edit(tx) as db:
        assert db is not None
        db.execute("MATCH (p:MeshPlace {nodeSymbol: $s}) DETACH DELETE p", s="t1")
    assert "SET m.edits" in tx.ran[0]  # the lock comes first
    assert "rebuilding_since" in tx.ran[1] and "DETACH DELETE" in tx.ran[2]
    assert tx.committed


def test_an_edit_during_a_rebuild_is_refused_and_writes_nothing():
    import time

    from heracles_ros.map_state import map_edit

    tx = _Tx(rebuilding_since=time.time())
    with map_edit(tx) as db:
        assert db is None
    assert not tx.committed  # the lock write is rolled back
