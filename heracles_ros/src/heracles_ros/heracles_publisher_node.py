#!/usr/bin/env python3

import logging
import os
import time

import numpy as np
import rclpy
import spark_dsg
from heracles.dsg_utils import summarize_dsg
from heracles.graph_interface import db_to_spark_dsg
from heracles.query_interface import Neo4jWrapper
from neo4j import GraphDatabase
from rclpy.node import Node

from heracles_ros.hydra_python_publisher import DsgPublisher
from heracles_ros.map_state import (
    METADATA_KEY,
    get_map_version,
    map_metadata,
    seed_db,
)

logger = logging.getLogger(__name__)


def center_w_h_to_bb(center, w, h):
    return np.array(
        [
            [center[0] - w, center[1] - h],
            [center[0] + w, center[1] - h],
            [center[0] + w, center[1] + h],
            [center[0] - w, center[1] + h],
        ]
    )


class HeraclesPublisher(Node):
    def __init__(self):
        super().__init__("heracles_publisher")

        # Declare and get parameters
        self.declare_parameter("heracles_ip", "")
        self.declare_parameter("heracles_port", -1)
        self.declare_parameter("heracles_neo4j_user", "neo4j")
        self.declare_parameter("heracles_neo4j_pass", "neo4j_pw")

        ip = self.get_parameter("heracles_ip").get_parameter_value().string_value
        port = self.get_parameter("heracles_port").get_parameter_value().integer_value
        self.get_logger().info(f"Port: {port}")

        assert ip != "", "Please set database IP"
        assert port > 0, "Please set database port"

        # IP / Port for database
        self.URI = f"neo4j://{ip}:{port}"
        self.get_logger().info(f"Connecting to {self.URI}")
        # Database name / password for database
        user = (
            self.get_parameter("heracles_neo4j_user").get_parameter_value().string_value
        )
        pw = (
            self.get_parameter("heracles_neo4j_pass").get_parameter_value().string_value
        )
        self.AUTH = (user, pw)
        self._wait_for_neo4j()

        # Load a scene graph into the database before publishing anything, so
        # Neo4j can be the only map: the JSON is just what fills it at startup.
        # Off by default -- it wipes the database, which must not happen on a
        # restart mid-run or where something else fills Neo4j.
        self.declare_parameter("seed_on_start", False)
        self.declare_parameter("seed_dsg_path", "")
        seed_path = self.get_parameter("seed_dsg_path").value
        if self.get_parameter("seed_on_start").value:
            if seed_path and os.path.isfile(seed_path):
                version = seed_db(self.URI, self.AUTH, seed_path)
                self.get_logger().info(
                    f"Seeded Neo4j from {seed_path} (map_version={version})"
                )
            else:
                # Without a map file, keep whatever Neo4j already holds rather
                # than dying and publishing no map at all.
                self.get_logger().error(
                    f"Not seeding Neo4j: no scene graph at '{seed_path}'. Is the "
                    "prior map set (run-adt4 -p)? Publishing what Neo4j already holds."
                )

        self.db = Neo4jWrapper(
            self.URI, self.AUTH, atomic_queries=True, print_profiles=False
        )
        self.db.connect()

        self.dsg_sender = DsgPublisher(self, "~/dsg_out", True)
        # Republish as soon as a writer bumps map_version, so nobody plans on
        # the graph from before a change for up to a full period. The slow
        # period still republishes changes that do not bump the version, such
        # as a held object following its robot.
        self.declare_parameter("version_poll_period_s", 0.5)
        self.declare_parameter("publish_period_s", 5.0)
        self._publish_period_s = self.get_parameter("publish_period_s").value
        self._published_version = None
        self._last_publish = None
        self.timer = self.create_timer(
            self.get_parameter("version_poll_period_s").value, self.poll_db
        )

    def _wait_for_neo4j(self):
        """Neo4j often starts together with this node: wait rather than die."""
        while True:
            try:
                with GraphDatabase.driver(self.URI, auth=self.AUTH) as driver:
                    driver.verify_connectivity()
                return
            except Exception as ex:
                self.get_logger().warn(
                    f"Waiting for Neo4j at {self.URI}: {ex}",
                    throttle_duration_sec=10.0,
                )
                time.sleep(2.0)

    def poll_db(self):
        # A Neo4j hiccup must not kill the node: restarting it would reseed and
        # wipe the map. Skip this tick; the driver reconnects on its own.
        try:
            self._poll_db()
        except Exception as ex:
            self.get_logger().error(
                f"Neo4j unavailable, map not published: {ex}",
                throttle_duration_sec=5.0,
            )

    def _poll_db(self):
        version = get_map_version(self.db)
        now = self.get_clock().now()
        stale = (
            self._last_publish is None
            or (now - self._last_publish).nanoseconds * 1e-9 >= self._publish_period_s
        )
        if version != self._published_version or stale:
            self.publish_dsg()

    def publish_dsg(self):
        # Read the metadata before the graph: a write landing in between makes
        # the graph newer than its version says, never older, and the next poll
        # republishes it under the new version.
        metadata = map_metadata(self.db)
        new_scene_graph = db_to_spark_dsg(self.db)
        new_scene_graph.metadata.add(metadata)
        summarize_dsg(new_scene_graph)

        for n in new_scene_graph.get_layer(spark_dsg.DsgLayers.MESH_PLACES).nodes:
            # Assign bounding_box if the attribute type supports it
            if hasattr(n.attributes, "bounding_box"):
                n.attributes.bounding_box = spark_dsg.BoundingBox(
                    (1, 1, 0.001), n.attributes.position
                )

            # Assign boundary polygon if the attribute type supports it.
            if hasattr(n.attributes, "boundary"):
                corners = center_w_h_to_bb(n.attributes.position, 1, 1)
                boundary = np.zeros((4, 3))
                boundary[:, :2] = corners
                boundary[:, 2] = n.attributes.position[2]
                n.attributes.boundary = boundary

        self.dsg_sender.publish(new_scene_graph, frame_id="map")
        state = metadata[METADATA_KEY]
        self._published_version = state["map_version"]
        self._last_publish = self.get_clock().now()
        self.get_logger().info(
            f"Published map (map_version={state['map_version']}, "
            f"holding={state['holding']})"
        )

    def destroy_node(self):
        self.db.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    # Create the node
    node = HeraclesPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up before shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
