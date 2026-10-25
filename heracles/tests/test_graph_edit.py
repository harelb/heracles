"""heracles.graph_edit: ops, undo, groups, snapshots.

Two kinds of DB test:

* ``HERACLES_EDIT_TEST_NEO4J_URI`` -- a database that may hold someone's map (e.g. the
  web console's :7697).  These tests never wipe: they create their own nodes far from
  any real map, and remove everything they added (nodes, mutations, labelspace
  entries, the revision node if they created it) afterwards.
* ``HERACLES_TEST_NEO4J_URI`` -- a throwaway instance; the snapshot restore test
  wipes and reloads it.
"""

import json
import os

import pytest

from heracles import graph_edit as ge
from heracles.transactions import RevisionConflict

AUTH = (
    os.environ.get("HERACLES_TEST_NEO4J_USER", "neo4j"),
    os.environ.get("HERACLES_TEST_NEO4J_PASSWORD", "neo4j_pw"),
)

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_layer_names_and_edge_types():
    assert ge.layer_label("objects") == "Object"
    assert ge.layer_label("region") == "Room"
    assert ge.layer_label("MeshPlace") == "MeshPlace"
    with pytest.raises(ge.EditError):
        ge.layer_label("kitchen")
    assert ge.determine_edge_type("Object", "Object") == "OBJECT_CONNECTED"
    assert ge.determine_edge_type("Room", "MeshPlace") == "CONTAINS"
    with pytest.raises(ge.EditError):
        ge.determine_edge_type("Place", "MeshPlace")  # same layer, different partition
    assert ge.layer_key("MeshPlace") == "3:1"


def test_symbols():
    assert ge.parse_symbol("O(12)") == ("O", 12)
    assert ge.symbol_variants("R3") == ["R3", "R(3)"]
    with pytest.raises(ge.EditError):
        ge.parse_symbol("O12; MATCH (n) DETACH DELETE n")


def test_polygon_helpers():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert ge.point_in_polygon(5, 5, sq)
    assert not ge.point_in_polygon(11, 5, sq)
    assert ge.polygon_centroid(sq) == pytest.approx((5.0, 5.0))
    tri = [[0, 0], [6, 0], [0, 6]]
    assert ge.polygon_centroid(tri) == pytest.approx((2.0, 2.0))
    assert ge._polygon(sq + [sq[0]]) == [[float(a), float(b)] for a, b in sq]  # closed ring
    with pytest.raises(ge.EditError):
        ge._polygon([[0, 0], [1, 1]])


def test_summarize_reads_like_a_diff():
    lines = ge.summarize([
        {"kind": "update", "symbol": "O41", "layer": "Object", "field": "class",
         "old": "vehicle", "new": "crate"},
        {"kind": "group", "symbol": "R0", "layer": "Room", "name": "courtyard",
         "members": {"MeshPlace": 37, "Object": 5}},
        {"kind": "retire", "symbol": "O7", "layer": "Object", "class": "chair", "name": "chair"},
        *[{"kind": "edge+", "from": "R0", "to": f"O{i}", "type": "CONTAINS"} for i in range(5)],
    ])  # fmt: skip
    assert lines[0] == "O41 class vehicle -> crate"
    assert lines[1] == "new region R0 'courtyard' containing 37 places, 5 objects"
    assert "retired" in lines[2]
    assert lines[-1] == "... and 2 more CONTAINS edges"


class _FakeTx:
    """Answers the applier's node lookup; records writes."""

    def __init__(self, props):
        self.props = props
        self.writes = []

    def run(self, query, **params):
        if "RETURN labels(n) AS labels" in query:
            return [_Rec({"labels": ["Object"], "props": self.props})]
        self.writes.append((query, params))
        return []


class _Rec:
    def __init__(self, d):
        self.d = d

    def data(self):
        return self.d


def test_update_rejects_non_whitelisted_properties():
    ap = ge._Applier(_FakeTx({"nodeSymbol": "O1", "class": "tree"}), 1)
    for bad in ("nodeSymbol", "x}) DETACH DELETE n //", "admission"):
        with pytest.raises(ge.EditError):
            ap.apply({"op": "update_node", "symbol": "O1", "set": {bad: 1}})
    with pytest.raises(ge.EditError):
        ap.apply({"op": "drop_database"})


def test_update_expect_conflict():
    ap = ge._Applier(_FakeTx({"nodeSymbol": "O1", "name": "tree"}), 1)
    with pytest.raises(ge.EditConflict):
        ap.apply({"op": "update_node", "symbol": "O1", "set": {"name": "x"},
                  "expect": {"name": "bush"}})  # fmt: skip


def test_update_no_change_writes_nothing():
    tx = _FakeTx({"nodeSymbol": "O1", "name": "tree"})
    ap = ge._Applier(tx, 1)
    ap.apply({"op": "update_node", "symbol": "O1", "set": {"name": "tree"}})
    assert tx.writes == [] and ap.changes == [] and ap.inverse == []


# ---------------------------------------------------------------------------
# shared database (never wiped)
# ---------------------------------------------------------------------------

SHARED_URI = os.environ.get("HERACLES_EDIT_TEST_NEO4J_URI")
FAR = 5000.0  # metres: nowhere near any real map
PROV = {"source": "heracles-test", "operator": "pytest"}


@pytest.fixture
def shared():
    if not SHARED_URI:
        pytest.skip("set HERACLES_EDIT_TEST_NEO4J_URI (a DB these tests clean up after)")
    from heracles.query_interface import Neo4jWrapper

    db = Neo4jWrapper(SHARED_URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No Neo4j at {SHARED_URI}: {exc}")
    q = lambda c, **k: db.execute(c, **k)[0]  # noqa: E731
    before = {
        "ls": {r["k"]: (r["ids"], r["names"]) for r in q(
            "MATCH (l:_Labelspace) RETURN l.layer AS k, l.ids AS ids, l.names AS names")},
        "rev_node": q("MATCH (s:_State {key: 'open_set_graph'}) RETURN count(s) AS c")[0]["c"],
        "mutations": {r["m"] for r in q("MATCH (m:GraphMutation) RETURN m.mutation_id AS m")},
        "counts": ge.map_counts(db),
    }  # fmt: skip
    yield db, q, before
    # ---- cleanup: everything a test added ----
    q("MATCH (n) WHERE n.source = 'heracles-test' DETACH DELETE n")
    q("MATCH (m:GraphMutation) WHERE NOT m.mutation_id IN $keep DETACH DELETE m",
      keep=list(before["mutations"]))  # fmt: skip
    for key in ("object", "room"):
        if key in before["ls"]:
            ids, names = before["ls"][key]
            q("MATCH (l:_Labelspace {layer: $k}) SET l.ids = $i, l.names = $n", k=key, i=ids, n=names)
        else:
            q("MATCH (l:_Labelspace {layer: $k}) DELETE l", k=key)
    if not before["rev_node"]:
        q("MATCH (s:_State {key: 'open_set_graph'}) DETACH DELETE s")
    assert ge.map_counts(db) == before["counts"]
    db.close()


def _obj(x, y, cls="sget test crate", **kw):
    return {"op": "create_node", "layer": "Object", "pos": [FAR + x, FAR + y, 0.0],
            "class": cls, "source": "heracles-test", **kw}  # fmt: skip


def test_create_update_retire_undo(shared):
    db, q, before = shared
    v0 = q("MATCH (s:_State {key: 'global'}) RETURN s.version AS v")
    out = ge.apply_edit(db, [_obj(0, 0)], provenance=PROV)
    sym = out["affected"][0]
    assert any(x.startswith(f"new object {sym} 'sget test crate'") for x in out["summary"])
    if "object" in before["ls"]:
        assert out["summary"][0] == "new object class 'sget test crate'"
    p = q("MATCH (n:Object {nodeSymbol: $s}) RETURN properties(n) AS p", s=sym)[0]["p"]
    assert p["class"] == "sget test crate" and p["admission_status"] == "admitted"
    assert p["layer"] == 2 and p["attr_type"]
    if "object" in before["ls"]:
        names = q("MATCH (l:_Labelspace {layer: 'object'}) RETURN l.names AS n")[0]["n"]
        assert "sget test crate" in names
        assert p["semantic_label"] == len(before["ls"]["object"][1]) and names[-1] == "sget test crate"
    if v0:
        assert out["state_version"] == v0[0]["v"] + 1
    m = q("MATCH (m:GraphMutation {mutation_id: $m}) RETURN properties(m) AS p", m=out["mutation_id"])
    assert m and m[0]["p"]["kind"] == "map_edit" and json.loads(m[0]["p"]["inverse_json"])

    # reclassify + move: the box follows the object
    up = ge.apply_edit(db, [{"op": "update_node", "symbol": sym,
                             "set": {"class": "tree", "center": [FAR + 2, FAR + 1]}}],
                       provenance=PROV)  # fmt: skip
    assert "class sget test crate -> tree" in up["summary"][0]
    p = q("MATCH (n:Object {nodeSymbol: $s}) RETURN properties(n) AS p", s=sym)[0]["p"]
    assert p["class"] == "tree" and list(p["bbox_center"])[:2] == pytest.approx([FAR + 2, FAR + 1])
    assert p["pos_x"] == pytest.approx(FAR + 2)

    # retire: gone from the layer, restorable
    ge.apply_edit(db, [{"op": "retire_node", "symbol": sym}], provenance=PROV)
    assert not q("MATCH (n:Object {nodeSymbol: $s}) RETURN n", s=sym)
    assert q("MATCH (n:Retired {nodeSymbol: $s}) RETURN n.retired_label AS l", s=sym)[0]["l"] == "Object"
    u = ge.undo_edit(db, provenance=PROV)
    assert u["changes"][0]["kind"] == "restore"
    assert q("MATCH (n:Object {nodeSymbol: $s}) RETURN count(n) AS c", s=sym)[0]["c"] == 1

    # undo the move/reclassify, then the create (history order, skipping undos)
    ge.undo_edit(db, provenance=PROV)
    p = q("MATCH (n:Object {nodeSymbol: $s}) RETURN properties(n) AS p", s=sym)[0]["p"]
    assert p["class"] == "sget test crate" and p["pos_x"] == pytest.approx(FAR)
    ge.undo_edit(db, provenance=PROV)
    assert not q("MATCH (n {nodeSymbol: $s}) RETURN n", s=sym)
    if "object" in before["ls"]:
        assert q("MATCH (l:_Labelspace {layer: 'object'}) RETURN l.names AS n")[0]["n"] == before["ls"]["object"][1]
    hist = ge.edit_history(db, limit=10)
    assert hist[0]["undo_of"] and hist[0]["provenance"]["operator"] == "pytest"
    assert ge.map_counts(db) == before["counts"]


def test_group_polygon_into_region_dry_run_and_undo(shared):
    db, q, before = shared
    ge.apply_edit(db, [_obj(0, 0), _obj(3, 1), _obj(30, 30), _obj(4, 4, cls="tree")], provenance=PROV)
    poly = [[FAR - 1, FAR - 1], [FAR + 6, FAR - 1], [FAR + 6, FAR + 6], [FAR - 1, FAR + 6]]
    op = {"op": "group", "parent_layer": "Room", "name": "sget test yard", "polygon": poly,
          "member_layers": ["Object"], "source": "heracles-test"}  # fmt: skip
    dry = ge.apply_edit(db, [op], provenance=PROV, dry_run=True)
    assert dry["dry_run"] and not q("MATCH (r:Room {name: 'sget test yard'}) RETURN r")
    assert any(line.endswith("containing 3 objects") for line in dry["summary"])
    out = ge.apply_edit(db, [op], provenance=PROV)
    r = out["changes"][-1]["symbol"]
    rows = q("MATCH (r:Room {nodeSymbol: $s})-[:CONTAINS]->(o:Object) RETURN count(o) AS c, "
             "r.boundary AS b, r.center AS c0, r.class AS cls", s=r)  # fmt: skip
    assert rows[0]["c"] == 3 and len(rows[0]["b"]) == 4 and rows[0]["cls"] == "region"
    assert list(rows[0]["c0"])[:2] == pytest.approx([FAR + 2.5, FAR + 2.5])
    regs = [x for x in ge.regions(db) if x["symbol"] == r]
    assert regs and regs[0]["members"] == {"Object": 3} and len(regs[0]["polygon"]) == 4
    # a "how many trees in the yard" Cypher
    assert q("MATCH (r:Room {name: 'sget test yard'})-[:CONTAINS]->(o:Object {class: 'tree'}) "
             "RETURN count(o) AS c")[0]["c"] == 1  # fmt: skip
    ge.undo_edit(db, provenance=PROV)
    assert not q("MATCH (r {nodeSymbol: $s}) RETURN r", s=r)


def test_edges_hard_delete_and_recreate(shared):
    db, q, before = shared
    out = ge.apply_edit(db, [_obj(0, 0), _obj(1, 0)], provenance=PROV)
    a, b = out["affected"]
    e = ge.apply_edit(db, [{"op": "create_edge", "from": a, "to": b}], provenance=PROV)
    assert e["changes"] == [{"kind": "edge+", "from": a, "to": b, "type": "OBJECT_CONNECTED"}]
    again = ge.apply_edit(db, [{"op": "create_edge", "from": b, "to": a}], provenance=PROV)
    assert again["changes"] == [] and again["mutation_id"] is None  # already connected
    d = ge.apply_edit(db, [{"op": "delete_node", "symbol": a}], provenance=PROV)
    assert d["changes"][0]["edges"] == 1
    assert not q("MATCH (n {nodeSymbol: $s}) RETURN n", s=a)
    ge.undo_edit(db, d["mutation_id"], provenance=PROV)
    assert q("MATCH (x:Object {nodeSymbol: $a})-[:OBJECT_CONNECTED]-(y {nodeSymbol: $b}) "
             "RETURN x.center AS c", a=a, b=b)  # fmt: skip
    ge.apply_edit(db, [{"op": "delete_edge", "from": b, "to": a}], provenance=PROV)
    assert not q("MATCH (:Object {nodeSymbol: $a})-[r]-(:Object {nodeSymbol: $b}) RETURN r", a=a, b=b)
    with pytest.raises(ge.EditError):
        ge.apply_edit(db, [{"op": "delete_edge", "from": b, "to": a}], provenance=PROV)


def test_revision_and_undo_conflicts(shared):
    db, q, before = shared
    rev = ge.graph_revision(db)
    out = ge.apply_edit(db, [_obj(0, 0)], provenance=PROV, expected_revision=rev)
    sym = out["affected"][0]
    with pytest.raises(RevisionConflict):
        ge.apply_edit(db, [{"op": "update_node", "symbol": sym, "set": {"name": "x"}}],
                      provenance=PROV, expected_revision=rev)  # fmt: skip
    first = ge.apply_edit(db, [{"op": "update_node", "symbol": sym, "set": {"name": "a"}}], provenance=PROV)
    ge.apply_edit(db, [{"op": "update_node", "symbol": sym, "set": {"name": "b"}}], provenance=PROV)
    assert ge.edits_since(db, rev) == 3
    with pytest.raises(ge.EditConflict):  # undoing the older edit would clobber "b"
        ge.undo_edit(db, first["mutation_id"], provenance=PROV)
    ge.undo_edit(db, first["mutation_id"], provenance=PROV, force=True)
    assert q("MATCH (n:Object {nodeSymbol: $s}) RETURN n.name AS n", s=sym)[0]["n"] == "sget test crate"
    with pytest.raises(ge.EditError):  # an op error rolls the whole batch back
        ge.apply_edit(db, [_obj(9, 9), {"op": "retire_node", "symbol": "O999999"}], provenance=PROV)
    assert q("MATCH (n:Object) WHERE n.pos_x = $x RETURN count(n) AS c", x=FAR + 9)[0]["c"] == 0


# ---------------------------------------------------------------------------
# throwaway database: snapshot round trip
# ---------------------------------------------------------------------------

SCRATCH_URI = os.environ.get("HERACLES_TEST_NEO4J_URI")


@pytest.fixture
def scratch():
    if not SCRATCH_URI:
        pytest.skip("set HERACLES_TEST_NEO4J_URI to a throwaway Neo4j")
    from heracles.query_interface import Neo4jWrapper

    db = Neo4jWrapper(SCRATCH_URI, AUTH, atomic_queries=True, print_profiles=False)
    try:
        db.connect()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No test Neo4j at {SCRATCH_URI}: {exc}")
    yield db
    db.close()


def _small_graph():
    import numpy as np
    import spark_dsg as dsg

    G = dsg.DynamicSceneGraph()
    G.set_labelspace(dsg.Labelspace({0: "unknown", 1: "tree", 2: "car"}), 2, 0)
    G.add_layer(3, 1, dsg.DsgLayers.MESH_PLACES)
    for i in range(12):
        a = dsg.Place2dNodeAttributes()
        a.position = np.array([float(i % 4) * 2, float(i // 4) * 2, 0.0])
        G.add_node(3, dsg.NodeSymbol("P", i), a, 1)
    for i in range(11):
        G.insert_edge(dsg.NodeSymbol("P", i), dsg.NodeSymbol("P", i + 1))
    for i, lab in enumerate([1, 2, 1]):
        o = dsg.ObjectNodeAttributes()
        o.position = np.array([1.0 + 2 * i, 1.0, 0.0])
        o.semantic_label = lab
        o.bounding_box = dsg.BoundingBox(np.array([0.5, 0.5, 0.5], np.float32),
                                         np.array(o.position, np.float32))  # fmt: skip
        G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", i), o)
        G.insert_edge(dsg.NodeSymbol("P", i), dsg.NodeSymbol("O", i))
    return G


def test_snapshot_round_trip(scratch, tmp_path):
    from heracles.graph_interface import initialize_db, spark_dsg_to_db

    db = scratch
    initialize_db(db)
    spark_dsg_to_db(_small_graph(), db)
    ge.apply_edit(db, [
        {"op": "group", "name": "yard", "polygon": [[-1, -1], [3, -1], [3, 3], [-1, 3]]},
        {"op": "update_node", "symbol": "O1", "set": {"class": "crate", "note": "was a car"}},
    ], provenance=PROV)  # fmt: skip
    counts = ge.map_counts(db)
    assert counts["Room"] == 1 and counts["Object"] == 3
    meta = ge.export_snapshot(db, tmp_path / "snap")
    assert meta["counts"] == counts and (tmp_path / "snap" / "dsg.json").is_file()
    extras = json.loads((tmp_path / "snap" / "extras.json").read_text())
    assert any(e["type"] == "CONTAINS" and e["to"].startswith("O") for e in extras["edges"])

    ge.apply_edit(db, [{"op": "retire_node", "symbol": "O0"},
                       {"op": "create_node", "layer": "Object", "pos": [9, 9], "class": "car"}],
                  provenance=PROV)  # fmt: skip
    assert ge.map_counts(db) != counts
    v = db.execute("MATCH (m:_MapState) RETURN m.version AS v")[0][0]["v"]
    restored = ge.restore_snapshot(db, tmp_path / "snap")
    assert restored == counts
    room = db.execute("MATCH (r:Room) RETURN r.name AS n, r.boundary AS b, r.class AS c")[0][0]
    assert room["n"] == "yard" and len(room["b"]) == 4 and room["c"] == "region"
    o1 = db.execute("MATCH (o:Object {nodeSymbol: 'O1'}) RETURN o.class AS c, o.note AS n")[0][0]
    assert o1["c"] == "crate" and o1["n"] == "was a car"
    assert db.execute("MATCH (m:_MapState) RETURN m.version AS v")[0][0]["v"] == v + 1
    assert not db.execute("MATCH (n:Retired) RETURN n")[0]
    # the history survives the restore
    assert db.execute("MATCH (m:GraphMutation) RETURN count(m) AS c")[0][0]["c"] == 2


def test_footprint_polygon_follows_the_places():
    # places along an L: the polygon keeps the L (no hull over the inner corner)
    places = [((x, 0.0), [1.0] * 8) for x in range(0, 20)] + [((19.0, y), [1.0] * 8) for y in range(1, 20)]
    poly = ge.footprint_polygon(places, simplify_m=0.2)
    from shapely.geometry import Point, Polygon

    shape = Polygon(poly)
    assert shape.contains(Point(10, 0)) and shape.contains(Point(19, 10))
    assert not shape.contains(Point(8, 10))  # the inner corner
    assert 60 < shape.area < 120
    assert ge.footprint_polygon([]) is None
    assert ge.footprint_polygon([((0.0, 0.0), None)]) is not None  # no radii: a small disc
