"""Place edits on a map whose places are T(i), as the current Hydra names them.

Place names are lowercase outside the map (t12 in a PDDL goal, in chat, in the
change log). The edits used to match the database with that lowercase name, so on
a T(i) map every one of them failed: "there is no place t417".

Needs the scratch Neo4j of test_map_state (HERACLES_TEST_NEO4J_URI).
"""
# The db fixture comes from test_map_state; ruff takes each test using it for a redefinition.
# ruff: noqa: F811

import pytest
from test_map_state import db  # noqa: F401  fixture

from heracles_ros import change_log
from heracles_ros.scene_change import (
    add_connection,
    add_place,
    place_exists,
    remove_connection,
    remove_place,
)


@pytest.fixture
def upper(db):
    """The test map with its places renamed t(i) -> T(i)."""
    db.query("MATCH (p:MeshPlace) SET p.nodeSymbol = 'T' + substring(p.nodeSymbol, 1)")
    return db


def _stored(conn):
    return {r["s"] for r in conn.query("MATCH (p:MeshPlace) RETURN p.nodeSymbol AS s")}


def _a_place_with_neighbours(conn):
    rows = conn.query(
        "MATCH (p:MeshPlace)-[:MESH_PLACE_CONNECTED]->(q:MeshPlace) "
        "RETURN p.nodeSymbol AS p, collect(DISTINCT q.nodeSymbol) AS q LIMIT 1"
    )
    return rows[0]["p"].lower(), [q.lower() for q in rows[0]["q"]]


def _connected(conn, a, b):
    rows = conn.execute(
        "MATCH (p:MeshPlace)-[r:MESH_PLACE_CONNECTED]-(q:MeshPlace) "
        "WHERE toLower(p.nodeSymbol) = $a AND toLower(q.nodeSymbol) = $b "
        "RETURN count(r) AS n",
        **{"a": a, "b": b},
    )[0]
    return rows[0]["n"]


def test_a_lowercase_name_finds_the_uppercase_place(upper):
    place, _ = _a_place_with_neighbours(upper)
    assert place.startswith("t") and place.upper() in _stored(upper)
    assert place_exists(upper, place)
    assert place_exists(upper, place.upper())
    assert not place_exists(upper, "t99999999")


def test_remove_a_place_by_its_lowercase_name(upper):
    place, _ = _a_place_with_neighbours(upper)
    done = remove_place(upper, place)
    assert done is not None and done["symbol"] == place and done["edges"] > 0
    assert place.upper() not in _stored(upper)
    assert remove_place(upper, place) is None


def test_a_new_place_takes_the_maps_letter(upper):
    _, neighbours = _a_place_with_neighbours(upper)
    added = add_place(upper, "", 0.5, 0.5, neighbours[:2])
    # the name handed back (and logged) is lowercase, the stored node is T(i)
    assert added["symbol"].startswith("t") and added["neighbours"] == neighbours[:2]
    assert added["symbol"].upper() in _stored(upper)
    assert not any(s.startswith("t") for s in _stored(upper))
    for n in neighbours[:2]:
        assert _connected(upper, added["symbol"], n) == 2
    # the name is taken now, under either case
    assert add_place(upper, added["symbol"], 1.0, 1.0, neighbours[:1]) is None
    assert add_place(upper, added["symbol"].upper(), 1.0, 1.0, neighbours[:1]) is None


def test_an_unknown_neighbour_refuses_the_new_place(upper):
    _, neighbours = _a_place_with_neighbours(upper)
    before = _stored(upper)
    assert add_place(upper, "", 0.5, 0.5, [neighbours[0], "t99999999"]) is None
    assert _stored(upper) == before


def test_disconnect_and_connect_by_lowercase_names(upper):
    place, neighbours = _a_place_with_neighbours(upper)
    other = neighbours[0]
    assert remove_connection(upper, place, other) == {"a": place, "b": other}
    assert _connected(upper, place, other) == 0
    assert remove_connection(upper, place, other) is None
    assert add_connection(upper, place, other) == {"a": place, "b": other}
    assert _connected(upper, place, other) == 2
    assert add_connection(upper, place, other) is None


def test_a_logged_edit_replays_on_the_uppercase_map(upper):
    place, _ = _a_place_with_neighbours(upper)
    change_log.record(upper, "sget/test", "remove_place", {"symbol": place})
    (entry,) = change_log.entries(upper)
    assert change_log.apply_op(upper, entry["op"], entry["data"]) is not None
    assert not place_exists(upper, place)


def test_a_lowercase_map_behaves_as_before(db):
    place, neighbours = _a_place_with_neighbours(db)
    added = add_place(db, "", 0.5, 0.5, neighbours[:2])
    assert added["symbol"] in _stored(db)  # stored lowercase on a lowercase map
    assert remove_place(db, place)["symbol"] == place
    assert place not in _stored(db)


def test_a_map_with_both_cases_takes_the_name_as_spelled(upper):
    place, neighbours = _a_place_with_neighbours(upper)
    # an old lowercase place with the same number, as a fused map can have
    upper.execute(
        "MATCH (src:MeshPlace {nodeSymbol: $s}) CREATE (p:MeshPlace) "
        "SET p = properties(src), p.nodeSymbol = toLower($s)",
        **{"s": place.upper()},
    )
    assert {place, place.upper()} <= _stored(upper)
    assert remove_place(upper, place.upper())["symbol"] == place
    assert place.upper() not in _stored(upper) and place in _stored(upper)
    assert remove_place(upper, place)["symbol"] == place
    assert place not in _stored(upper)
