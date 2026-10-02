# A batch of scene graph changes.
#
# Sent to the scene change writer by whoever observed them, and published again
# by the writer once they are in Neo4j ("applied"), with map_version set to the
# version of the map that contains them. A planner reacting to a change should
# wait for a scene graph at least that new.

std_msgs/Header header

# Who observed the changes, e.g. "change_detection". Changes a robot made
# itself (pick, place) are published with source "executor/<robot>".
string source

# Applied topic only: the map version that contains these changes. 0 on input.
uint64 map_version

SceneChange[] changes
