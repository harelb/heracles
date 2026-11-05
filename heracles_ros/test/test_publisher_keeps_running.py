"""A map that cannot be built is reported, and the publisher keeps running."""

from types import SimpleNamespace as NS

from heracles_ros import heracles_publisher_node as hp


class _Time:
    def __sub__(self, other):
        return NS(nanoseconds=0)


def _node(errors):
    clock = NS(now=_Time)
    node = NS(
        db=None,
        _published_version=3,
        _last_publish=None,
        _bad_version=None,
        _publish_period_s=5.0,
        get_clock=lambda: clock,
        get_logger=lambda: NS(error=errors.append),
    )

    def publish_dsg():
        raise ValueError("Node O21 has no attr_type property.")

    node.publish_dsg = publish_dsg
    return node


def test_an_unbuildable_map_is_reported_once_and_does_not_raise(monkeypatch):
    monkeypatch.setattr(hp, "get_map_version", lambda db: 4)
    monkeypatch.setattr(hp, "is_rebuilding", lambda db: False)
    errors = []
    node = _node(errors)
    hp.HeraclesPublisher._poll_db(node)
    hp.HeraclesPublisher._poll_db(node)  # same version: not reported again
    assert len(errors) == 1
    assert "map_version=4" in errors[0] and "keep map_version=3" in errors[0]
    assert "O21" in errors[0]
