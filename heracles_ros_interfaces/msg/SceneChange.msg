# One change to an object in the scene graph.
#
# A perception system (e.g. change detection) reports these; the scene change
# writer applies them to Neo4j. Positions are in the map frame.

uint8 ADDED=0
uint8 REMOVED=1
uint8 MOVED=2
uint8 kind

# The object's nodeSymbol, e.g. "O16". For ADDED it may be left empty: the
# writer then assigns the next free object symbol.
string symbol

# Class, e.g. "chair". ADDED needs it; otherwise informational.
string semantic_label

# MOVED / REMOVED: where it was. On the applied topic the writer fills this
# from the map, whatever the sender put here.
geometry_msgs/Point old_position

# ADDED / MOVED: where it is now.
geometry_msgs/Point new_position

float32 confidence
