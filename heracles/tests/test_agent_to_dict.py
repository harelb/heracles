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
    return attrs


def test_agent_to_dict_includes_position_and_orientation():
    node = _FakeAgentNode("a0", _make_attrs())

    d = agent_to_dict(node)

    assert d["nodeSymbol"] == "a0"
    assert d["pos_x"] == 1.0 and d["pos_y"] == 2.0 and d["pos_z"] == 3.0
    assert d["rot_w"] == 0.5 and d["rot_x"] == 0.5
    assert d["rot_y"] == 0.5 and d["rot_z"] == 0.5
    assert "image_folder" not in d


def test_agent_to_dict_timestamp_from_attribute():
    import datetime

    attrs = _make_attrs()
    attrs.timestamp = datetime.timedelta(seconds=1.5)
    d = agent_to_dict(_FakeAgentNode("a2", attrs))
    assert d["timestamp_ns"] == 1_500_000_000


def test_agent_to_dict_timestamp_absent_when_unset():
    # Default binding timestamp is timedelta(0) == unset; the store's keyframe
    # frame supplies it in _collect_agents.
    d = agent_to_dict(_FakeAgentNode("a3", _make_attrs()))
    assert "timestamp_ns" not in d
