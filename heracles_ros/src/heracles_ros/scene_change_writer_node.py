#!/usr/bin/env python3
"""Write observed scene changes into Neo4j, then announce them as applied.

Input  ~/scene_changes          SceneChangeMsg from a perception system
Output ~/scene_changes_applied  the changes as the map now has them, with the
                                map_version that contains them

Announcing only after the write lets a planner wait for a scene graph that is
at least that version before it replans, instead of replanning on the map from
before the change.

Every applied change is also logged (change_log.py), so the map can be rebuilt
from the prior scene graph (prior_dsg_path) without some of them:

  ~/reset           (std_srvs/Trigger) drop every change: Neo4j is the prior
                    scene graph again. Also clears the AWCD base station's
                    cached changes (awcd_clear_service), or they would come
                    straight back.
  ~/undo_last_edit  drop the latest map edit made by language (source sget/...)
  ~/clear_edits     drop every map edit made by language

Sources matching snapshot_sources (change detection: "awcd/") send their whole
current change set in every message instead of new changes only. Each change
carries the reporter's id (removed objects: the prior symbol; added objects:
a key such as "awcd:reconciled:12" in the symbol field). A change missing from
the next message was withdrawn, and the map is rebuilt without it -- a removed
object comes back. A message whose added objects have no key is from an older
sender and is applied as new changes only.

While the map is being reloaded or rebuilt by someone else (the console, a
seed), incoming changes are queued and applied once it is done, since an
object added meanwhile could take a symbol the load then reuses; undo and
reset are refused. One failing message or request is logged and answered
(success=False): it does not stop the node.
"""

import os
import threading
import traceback
from collections import deque

import rclpy
from geometry_msgs.msg import Point
from heracles.query_interface import Neo4jWrapper
from heracles_ros_interfaces.msg import SceneChange, SceneChangeMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from heracles_ros import change_log
from heracles_ros.map_state import (
    bump_map_version,
    get_map_version,
    get_prior_path,
    is_rebuilding,
    rebuild_state,
)
from heracles_ros.scene_change import (
    ADDED,
    KIND_NAMES,
    REMOVED,
    add_object,
    apply_changes,
    move_object,
    remove_object,
)

EDIT_SOURCE_PREFIX = "sget/"
RESET_SOURCE = "scene_graph/reset"
# Changes held while the map is reloaded; the oldest go first past this.
MAX_QUEUED = 200


def to_point(xyz):
    return (
        Point()
        if xyz is None
        else Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    )


def applied_msg(source, version, applied, stamp):
    out = SceneChangeMsg()
    out.header.stamp = stamp
    out.header.frame_id = "map"
    out.source = source
    out.map_version = int(version)
    for a in applied:
        c = SceneChange()
        c.kind = a["kind"]
        c.symbol = a["symbol"]
        c.semantic_label = a.get("label") or ""
        c.old_position = to_point(a["old"])
        c.new_position = to_point(a["new"])
        c.confidence = 1.0
        out.changes.append(c)
    return out


def snapshot_key(change) -> str:
    """The reporter's id for a change in a snapshot message ("" if it has none)."""
    if change.kind == REMOVED:
        return f"removed:{change.symbol.strip()}" if change.symbol.strip() else ""
    return change.symbol.strip()


def is_snapshot(msg, prefixes) -> bool:
    if not any(msg.source.startswith(p) for p in prefixes if p):
        return False
    return all(snapshot_key(c) for c in msg.changes)


class SceneChangeWriter(Node):
    def __init__(self):
        super().__init__("scene_change_writer")
        self.declare_parameter("heracles_ip", "")
        self.declare_parameter("heracles_port", -1)
        self.declare_parameter("heracles_neo4j_user", "neo4j")
        self.declare_parameter("heracles_neo4j_pass", "neo4j_pw")
        # Observations less sure than this are dropped.
        self.declare_parameter("min_confidence", 0.0)
        # The original map that rebuilds start from: the file Neo4j was seeded
        # or loaded from (map_state.get_prior_path), unless set here. A map that
        # was not loaded from a file is never replaced by one.
        self.declare_parameter("prior_dsg_path", "")
        self.declare_parameter("snapshot_sources", ["awcd/"])
        # An added object reported again this far from where the map has it moves.
        self.declare_parameter("snapshot_move_m", 0.3)
        self.declare_parameter("awcd_clear_service", "awcd_base_station/clear_robots")
        ip = self.get_parameter("heracles_ip").value
        port = self.get_parameter("heracles_port").value
        assert ip and port > 0, "Please set the database IP and port"
        self.db = Neo4jWrapper(
            f"neo4j://{ip}:{port}",
            (
                self.get_parameter("heracles_neo4j_user").value,
                self.get_parameter("heracles_neo4j_pass").value,
            ),
            atomic_queries=True,
            print_profiles=False,
        )
        self.db.connect()
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.prior_param = self.get_parameter("prior_dsg_path").value
        self._prior = None  # (path, scene graph), loaded on the first rebuild
        self.snapshot_sources = list(self.get_parameter("snapshot_sources").value)
        self.snapshot_move_m = float(self.get_parameter("snapshot_move_m").value)
        # Changes, rebuilds and resets one at a time.
        self._lock = threading.Lock()
        # Changes that arrived while the map was being reloaded.
        self._queued = deque()
        try:
            change_log.ensure_schema(self.db)
        except Exception as ex:
            self.get_logger().error(f"Could not set up the change log schema: {ex}")

        self.applied_pub = self.create_publisher(
            SceneChangeMsg, "~/scene_changes_applied", 10
        )
        self.create_subscription(SceneChangeMsg, "~/scene_changes", self.on_changes, 10)
        services = MutuallyExclusiveCallbackGroup()
        for name, callback in (
            ("reset", self.on_reset),
            ("undo_last_edit", self.on_undo_last_edit),
            ("clear_edits", self.on_clear_edits),
        ):
            self.create_service(
                Trigger,
                f"~/{name}",
                self._answering(name, callback),
                callback_group=services,
            )
        self.create_timer(2.0, self._drain_queued)
        self._awcd_clear = self._make_awcd_clear_client(
            self.get_parameter("awcd_clear_service").value
        )
        self.get_logger().info(
            "Writing scene changes from ~/scene_changes into Neo4j; original map: "
            + (
                self._prior_path()
                or "not seeded from a file yet, so changes cannot be undone"
            )
        )

    # ------------- incoming changes -------------

    def on_changes(self, msg: SceneChangeMsg):
        try:
            with self._lock:
                if self._queued or is_rebuilding(self.db):
                    self._queue(msg)
                    return
                self._apply(msg)
        except Exception as ex:
            self._log_error(f"{msg.source}: changes not applied", ex)

    def _apply(self, msg):
        if is_snapshot(msg, self.snapshot_sources):
            self._on_snapshot(msg)
        else:
            self._on_new_changes(msg)

    def _queue(self, msg):
        if len(self._queued) >= MAX_QUEUED:
            dropped = self._queued.popleft()
            self.get_logger().warn(
                f"Map still being reloaded: dropped the oldest queued changes "
                f"(from {dropped.source})"
            )
        self._queued.append(msg)
        self.get_logger().warn(
            f"{msg.source}: the map is being reloaded or rebuilt; "
            f"{len(msg.changes)} change(s) queued until it is done "
            f"({len(self._queued)} message(s) waiting)",
            throttle_duration_sec=5.0,
        )

    def _drain_queued(self):
        """Apply the changes queued during a reload, once it is over."""
        if not self._queued:
            return
        try:
            with self._lock:
                if is_rebuilding(self.db):
                    return
                self.get_logger().info(
                    f"Map reload done: applying {len(self._queued)} queued message(s)"
                )
                while self._queued:
                    msg = self._queued.popleft()
                    try:
                        self._apply(msg)
                    except Exception as ex:
                        self._log_error(f"{msg.source}: queued changes not applied", ex)
        except Exception as ex:
            self._log_error("Queued changes not applied", ex)

    def _log_error(self, what, ex):
        self.get_logger().error(f"{what}: {ex!r}\n{traceback.format_exc()}")

    def _wanted(self, msg):
        return [c for c in msg.changes if c.confidence >= self.min_confidence]

    def _on_new_changes(self, msg: SceneChangeMsg):
        wanted = [
            (
                c.kind,
                c.symbol.strip(),
                c.semantic_label.strip(),
                (c.new_position.x, c.new_position.y, c.new_position.z),
            )
            for c in self._wanted(msg)
        ]
        applied = apply_changes(self.db, wanted)
        change_log.record_applied(self.db, msg.source, applied)
        self._announce(msg.source, applied, len(msg.changes) - len(applied))

    def _on_snapshot(self, msg: SceneChangeMsg):
        """Make the map hold exactly this source's current changes."""
        incoming = {snapshot_key(c): c for c in self._wanted(msg)}
        logged = {}
        for e in change_log.entries(self.db, source=msg.source):
            logged.setdefault(e["key"], e)
        withdrawn = [e for k, e in logged.items() if k not in incoming]
        applied = []
        if withdrawn:
            if self._can_rebuild():
                change_log.deactivate(self.db, [e["seq"] for e in withdrawn])
                applied += self._rebuild()
            else:
                self.get_logger().warn(
                    f"{msg.source} withdrew {len(withdrawn)} change(s), but they stay "
                    "in the map: no prior scene graph to rebuild it from"
                )
        for key, c in incoming.items():
            xyz = (c.new_position.x, c.new_position.y, c.new_position.z)
            entry = logged.get(key)
            if entry is None:
                if c.kind == REMOVED:
                    done = remove_object(self.db, c.symbol.strip())
                elif c.kind == ADDED:
                    done = add_object(self.db, "", c.semantic_label.strip(), xyz)
                else:
                    done = move_object(self.db, c.symbol.strip(), xyz)
                if done is not None:
                    change_log.record_applied(self.db, msg.source, [done], key)
                    applied.append(done)
            elif entry["op"] == "add_object" and entry["active"]:
                old = entry["data"]["xyz"]
                if (
                    sum((a - b) ** 2 for a, b in zip(old, xyz))
                    > self.snapshot_move_m**2
                ):
                    done = move_object(self.db, entry["data"]["symbol"], xyz)
                    if done is not None:
                        change_log.update_data(
                            self.db, entry["seq"], {**entry["data"], "xyz": list(xyz)}
                        )
                        applied.append(done)
        self._announce(msg.source, applied, 0, quiet=not msg.changes and not withdrawn)

    def _announce(self, source, applied, skipped, quiet=False):
        if not applied:
            if not quiet:
                self.get_logger().info(
                    f"{source}: nothing to apply"
                    + (
                        f" ({skipped} change(s) skipped: unknown object, held by a "
                        "robot, or below min_confidence)"
                        if skipped
                        else ""
                    )
                )
            return
        version = bump_map_version(self.db)
        self.applied_pub.publish(
            applied_msg(source, version, applied, self.get_clock().now().to_msg())
        )
        self.get_logger().info(
            f"{source}: applied "
            + ", ".join(f"{KIND_NAMES[a['kind']]} {a['symbol']}" for a in applied)
            + f" (map_version={version}"
            + (f", {skipped} skipped)" if skipped else ")")
        )

    # ------------- rebuilds -------------

    def _prior_path(self) -> str:
        return self.prior_param or get_prior_path(self.db)

    def _can_rebuild(self) -> bool:
        return not self._prior_problem()

    def _prior_problem(self) -> str:
        """Why the map cannot be rebuilt now ("" if it can)."""
        path = self._prior_path()
        if not path:
            return (
                "The map was not seeded from a scene graph file (and prior_dsg_path "
                "is not set), so there is no original to rebuild it from."
            )
        if not os.path.isfile(path):
            return f"The prior scene graph file {path} does not exist (any more)."
        return ""

    def _rebuild(self):
        import spark_dsg

        path = self._prior_path()
        # Read now, not at startup: a later seed or load may have changed it.
        self.get_logger().info(f"Rebuilding the map from the prior scene graph {path}")
        if self._prior is None or self._prior[0] != path:
            self._prior = (path, spark_dsg.DynamicSceneGraph.load(path))
        return change_log.rebuild(self.db, self._prior[1], log=self._rebuild_log)

    def _rebuild_log(self, line: str):
        logger = self.get_logger()
        if line.startswith("ERROR"):
            logger.error(line)
        elif line.startswith("WARNING") or line.startswith("  "):
            logger.warn(line)
        else:
            logger.info(line)

    def _drop(self, source, entries, response, what):
        """Deactivate entries, rebuild, announce; fills a Trigger response."""
        problem = self._prior_problem()
        if not problem and rebuild_state(self.db)[0]:
            problem = "The map is being reloaded right now; try again when it is done."
        if problem:
            response.success = False
            response.message = problem
            self.get_logger().warn(f"{source}: {response.message}")
            return response
        change_log.deactivate(self.db, [e["seq"] for e in entries])
        restored = self._rebuild()
        version = get_map_version(self.db)
        if restored:
            self.applied_pub.publish(
                applied_msg(source, version, restored, self.get_clock().now().to_msg())
            )
        response.success = True
        response.message = (
            f"{what}; map rebuilt from the prior scene graph (map_version {version})."
        )
        self.get_logger().info(response.message)
        return response

    def _answering(self, name, callback):
        """A service callback that answers success=False instead of raising."""

        def answer(request, response):
            try:
                return callback(request, response)
            except Exception as ex:
                self._log_error(f"~/{name} failed", ex)
                response.success = False
                response.message = f"{name} failed: {ex}"
                return response

        return answer

    def on_reset(self, request, response):
        cleared = self._clear_awcd()
        with self._lock:
            entries = change_log.entries(self.db)
            self._drop(
                RESET_SOURCE, entries, response, f"Dropped all {len(entries)} change(s)"
            )
        if response.success and cleared:
            response.message += f" {cleared}"
        return response

    def on_undo_last_edit(self, request, response):
        with self._lock:
            edits = [
                e
                for e in change_log.entries(self.db)
                if e["source"].startswith(EDIT_SOURCE_PREFIX)
            ]
            if not edits:
                response.success = False
                response.message = "There is no map edit to undo."
                self.get_logger().warn(f"undo_last_edit: {response.message}")
                return response
            last = edits[-1]
            return self._drop(
                last["source"],
                [last],
                response,
                f"Undid {last['op']} {json_args(last['data'])}",
            )

    def on_clear_edits(self, request, response):
        with self._lock:
            edits = [
                e
                for e in change_log.entries(self.db)
                if e["source"].startswith(EDIT_SOURCE_PREFIX)
            ]
            return self._drop(
                EDIT_SOURCE_PREFIX + "clear",
                edits,
                response,
                f"Dropped {len(edits)} map edit(s)",
            )

    # ------------- AWCD base station -------------

    def _make_awcd_clear_client(self, service):
        if not service:
            return None
        try:
            # Optional: only where the AWCD base station's messages are built.
            from dcist_launch_system_msgs.srv import ClearRobots
        except ImportError:
            self.get_logger().info(
                "dcist_launch_system_msgs not found: ~/reset will not clear AWCD"
            )
            return None
        self._clear_type = ClearRobots
        return self.create_client(
            ClearRobots, service, callback_group=MutuallyExclusiveCallbackGroup()
        )

    def _clear_awcd(self) -> str:
        """Clear the AWCD base station's cached changes; what happened, for the reply."""
        if self._awcd_clear is None:
            return ""
        if not self._awcd_clear.wait_for_service(timeout_sec=1.0):
            return "(AWCD base station not running: nothing to clear there.)"
        done = threading.Event()
        future = self._awcd_clear.call_async(self._clear_type.Request())
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout=3.0):
            return "(AWCD base station did not answer the clear request.)"
        result = future.result()
        return f"(AWCD base station: {result.message or 'cleared'}.)"

    def destroy_node(self):
        self.db.close()
        super().destroy_node()


def json_args(data: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in data.items() if k != "label") or ""


def main(args=None):
    rclpy.init(args=args)
    node = SceneChangeWriter()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
