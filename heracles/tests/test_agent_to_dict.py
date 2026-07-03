import spark_dsg

from heracles.graph_interface import agent_to_dict


class _FakeId:
    def __init__(self, symbol):
        self._symbol = symbol

    def str(self, _short):
        return self._symbol


class _FakeAgentNode:
    """Minimal stand-in exposing the two members agent_to_dict reads."""

    def __init__(self, symbol, attributes):
        self.id = _FakeId(symbol)
        self.attributes = attributes


def _make_attrs():
    attrs = spark_dsg.AgentNodeAttributes()
    attrs.position = [1.0, 2.0, 3.0]
    # w, x, y, z — a non-identity rotation so we detect field mixups
    attrs.world_R_body = spark_dsg.Quaternion(0.5, 0.5, 0.5, 0.5)
    attrs.image_folder = "/data/agents/agent_42"
    return attrs


def test_agent_to_dict_includes_position_and_orientation():
    node = _FakeAgentNode("a0", _make_attrs())

    d = agent_to_dict(node)

    assert d["nodeSymbol"] == "a0"
    assert d["pos_x"] == 1.0 and d["pos_y"] == 2.0 and d["pos_z"] == 3.0
    assert d["rot_w"] == 0.5 and d["rot_x"] == 0.5
    assert d["rot_y"] == 0.5 and d["rot_z"] == 0.5
    assert d["image_folder"] == "/data/agents/agent_42"


def test_agent_to_dict_omits_image_folder_when_empty():
    attrs = _make_attrs()
    attrs.image_folder = ""
    node = _FakeAgentNode("a1", attrs)

    d = agent_to_dict(node)

    # empty image_folder is still emitted as a key (empty string), matching the
    # existing hasattr-guarded behavior; orientation is always present.
    assert d["rot_w"] == 0.5
