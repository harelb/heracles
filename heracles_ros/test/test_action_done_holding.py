"""Picks and places from /action_done when the holding service call is lost.

The field case: testbot_a picked O58 and placed it, both SUCCESS, but its
update_holding_state calls never reached the base, so the map kept O58 where
it was picked. The state updater now also reads the action reports. The map
is faked (the dsg_updater and scene_change calls it makes), so no Neo4j.
"""

from types import SimpleNamespace as NS

import pytest
from geometry_msgs.msg import Point
from heracles_ros_interfaces.srv import UpdateHoldingState

from heracles_ros import heracles_state_updater_node as hsu

msgs = pytest.importorskip("robot_executor_msgs.msg")


class _Map:
    def __init__(self):
        self.objects = {"O58": (-53.45, 77.75, 0.0)}
        self.holds = {}  # object -> robot
        self.logged = []  # change log moves
        self.version = 4
        self.announced = []


@pytest.fixture
def world(monkeypatch):
    m = _Map()

    class _Wrapper:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return m

        def __exit__(self, *a):
            pass

    def hold(db, robot, obj):
        if obj not in m.objects:
            return False
        m.holds[obj] = robot
        return True

    def unhold(db, robot, obj):
        if m.holds.get(obj) == robot:
            del m.holds[obj]
        return obj in m.objects

    def set_center(db, obj, x, y, z):
        m.objects[obj] = (x, y, z)
        return True

    def bump(db):
        m.version += 1
        return m.version

    monkeypatch.setattr(hsu, "Neo4jWrapper", _Wrapper)
    monkeypatch.setattr(hsu, "holder_of", lambda db, o: m.holds.get(o))
    monkeypatch.setattr(hsu, "object_center", lambda db, o: m.objects.get(o))
    monkeypatch.setattr(hsu, "robot_hold_obj", hold)
    monkeypatch.setattr(hsu, "robot_unhold_obj", unhold)
    monkeypatch.setattr(hsu, "set_obj_center", set_center)
    monkeypatch.setattr(hsu, "record", lambda db, src, op, data: m.logged.append(data))
    monkeypatch.setattr(hsu, "bump_map_version", bump)
    return m


def _updater(world, robot_pose=None):
    lines = []
    logger = NS(
        info=lambda s, **_: lines.append(("info", s)),
        warn=lambda s, **_: lines.append(("warn", s)),
        error=lambda s, **_: lines.append(("error", s)),
    )
    u = NS(
        robot_names=["testbot_a", "testbot_b"],
        _action_types=(msgs.ActionMsg.PICK, msgs.ActionMsg.PLACE),
        dsgdb_conf=NS(
            uri="neo4j://fake",
            username=NS(get_secret_value=lambda: "u"),
            password=NS(get_secret_value=lambda: "p"),
        ),
        _get_robot_pose=lambda name: robot_pose,
        _released={},
        get_logger=lambda: logger,
        applied_pub=NS(publish=world.announced.append),
        get_clock=lambda: NS(
            now=lambda: NS(
                to_msg=lambda: __import__(
                    "builtin_interfaces.msg", fromlist=["Time"]
                ).Time()
            )
        ),
        lines=lines,
    )
    for name in (
        "_on_action_done",
        "action_done_callback",
        "_update_holding_state",
        "_already",
        "update_holding_state_callback",
    ):
        f = getattr(hsu.HeraclesStateUpdater, name)
        setattr(u, name, (lambda f: lambda *a, **kw: f(u, *a, **kw))(f))
    return u


def _done(action, obj="O58", robot="testbot_a", points=(), success=True):
    msg = msgs.ActionDoneMsg()
    msg.robot_name, msg.action_type, msg.object_id = robot, action, obj
    msg.success = success
    msg.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
    return msg


def _service(u, robot, is_holding, obj="O58", position=None):
    req = UpdateHoldingState.Request()
    req.is_holding, req.id = is_holding, obj
    if position is not None:
        req.has_position = True
        req.position = Point(x=position[0], y=position[1], z=position[2])
    return u.update_holding_state_callback(robot, req, UpdateHoldingState.Response())


T2249 = (-51.66, 76.61, 0.0)


def test_pick_and_place_results_move_the_object_without_the_service(world):
    u = _updater(world)
    u.action_done_callback(_done("PICK"))
    assert world.holds == {"O58": "testbot_a"} and world.version == 5
    u.action_done_callback(_done("PLACE", points=[T2249]))
    assert world.holds == {} and world.objects["O58"] == T2249
    assert world.logged == [{"symbol": "O58", "xyz": list(T2249)}]
    assert world.version == 6 and world.announced[-1].changes[0].symbol == "O58"
    assert any(k == "warn" and "has not arrived" in s for k, s in u.lines)


def test_a_place_without_a_point_uses_the_robots_pose(world):
    u = _updater(world, robot_pose=(1.0, 2.0, 0.3, 1, 0, 0, 0))
    u.action_done_callback(_done("PICK"))
    u.action_done_callback(_done("PLACE"))
    assert world.objects["O58"] == (1.0, 2.0, 0.3) and world.holds == {}


def test_the_service_then_the_report_moves_the_object_once(world):
    u = _updater(world)
    assert _service(u, "testbot_a", True).success
    assert _service(u, "testbot_a", False, position=T2249).success
    v = world.version
    u.action_done_callback(_done("PICK"))  # late report of the pick
    u.action_done_callback(_done("PLACE", points=[T2249]))
    assert world.version == v and len(world.logged) == 1
    assert world.holds == {}  # the late PICK did not put it back on the robot


def test_the_report_then_the_service_moves_the_object_once(world):
    u = _updater(world)
    u.action_done_callback(_done("PICK"))
    assert _service(u, "testbot_a", True).success  # already held: no bump
    u.action_done_callback(_done("PLACE", points=[T2249]))
    v = world.version
    assert _service(u, "testbot_a", False, position=T2249).success
    assert world.version == v and len(world.logged) == 1


def test_failures_other_actions_and_other_robots_are_ignored(world):
    u = _updater(world)
    u.action_done_callback(_done("PICK", success=False))
    u.action_done_callback(_done("FOLLOW"))
    u.action_done_callback(_done("PICK", robot="testbot_c"))  # not this updater's
    assert world.holds == {} and world.version == 4


def test_a_report_without_an_object_is_logged(world):
    u = _updater(world)
    u.action_done_callback(_done("PICK", obj=""))
    assert world.holds == {}
    assert any(k == "warn" and "names no object" in s for k, s in u.lines)
