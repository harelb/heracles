#!/usr/bin/env python3
"""Write observed scene changes into Neo4j, then announce them as applied.

Input  ~/scene_changes          SceneChangeMsg from a perception system
Output ~/scene_changes_applied  the changes as the map now has them, with the
                                map_version that contains them

Announcing only after the write lets a planner wait for a scene graph that is
at least that version before it replans, instead of replanning on the map from
before the change.
"""

import rclpy
from geometry_msgs.msg import Point
from heracles.query_interface import Neo4jWrapper
from heracles_ros_interfaces.msg import SceneChange, SceneChangeMsg
from rclpy.node import Node

from heracles_ros.map_state import bump_map_version
from heracles_ros.scene_change import KIND_NAMES, apply_changes


def to_point(xyz):
    return Point() if xyz is None else Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


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
        c.old_position = to_point(a["old"])
        c.new_position = to_point(a["new"])
        c.confidence = 1.0
        out.changes.append(c)
    return out


class SceneChangeWriter(Node):
    def __init__(self):
        super().__init__("scene_change_writer")
        self.declare_parameter("heracles_ip", "")
        self.declare_parameter("heracles_port", -1)
        self.declare_parameter("heracles_neo4j_user", "neo4j")
        self.declare_parameter("heracles_neo4j_pass", "neo4j_pw")
        # Observations less sure than this are dropped.
        self.declare_parameter("min_confidence", 0.0)
        ip = self.get_parameter("heracles_ip").value
        port = self.get_parameter("heracles_port").value
        assert ip and port > 0, "Please set the database IP and port"
        self.db = Neo4jWrapper(
            f"neo4j://{ip}:{port}",
            (self.get_parameter("heracles_neo4j_user").value,
             self.get_parameter("heracles_neo4j_pass").value),
            atomic_queries=True,
            print_profiles=False,
        )
        self.db.connect()
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.applied_pub = self.create_publisher(SceneChangeMsg, "~/scene_changes_applied", 10)
        self.create_subscription(SceneChangeMsg, "~/scene_changes", self.on_changes, 10)
        self.get_logger().info("Writing scene changes from ~/scene_changes into Neo4j")

    def on_changes(self, msg: SceneChangeMsg):
        wanted = [
            (c.kind, c.symbol.strip(), c.semantic_label.strip(),
             (c.new_position.x, c.new_position.y, c.new_position.z))
            for c in msg.changes
            if c.confidence >= self.min_confidence
        ]
        applied = apply_changes(self.db, wanted)
        skipped = len(msg.changes) - len(applied)
        if not applied:
            self.get_logger().info(
                f"{msg.source}: nothing to apply ({skipped} change(s) skipped: unknown "
                "object, held by a robot, or below min_confidence)"
            )
            return
        version = bump_map_version(self.db)
        self.applied_pub.publish(
            applied_msg(msg.source, version, applied, self.get_clock().now().to_msg())
        )
        self.get_logger().info(
            f"{msg.source}: applied "
            + ", ".join(f"{KIND_NAMES[a['kind']]} {a['symbol']}" for a in applied)
            + f" (map_version={version}"
            + (f", {skipped} skipped)" if skipped else ")")
        )

    def destroy_node(self):
        self.db.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SceneChangeWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
