"""Which file a rebuild starts from, without a database."""

from heracles_ros.map_state import get_prior_path


class _Db:
    """Answers get_prior_path's two queries from fixed rows."""

    def __init__(self, map_state=None, graph_source=None):
        self.map_state = map_state
        self.graph_source = graph_source

    def query(self, cypher):
        if "_MapState" in cypher:
            return [] if self.map_state is None else [{"p": self.map_state}]
        assert "_GraphMetadata" in cypher
        return [] if self.graph_source is None else [{"p": self.graph_source}]


def test_the_seeded_path_comes_first():
    db = _Db(map_state="/maps/seeded/dsg.json", graph_source="/maps/other/dsg.json")
    assert get_prior_path(db) == "/maps/seeded/dsg.json"


def test_a_map_loaded_without_seeding_uses_its_source_file():
    # A reload that wiped _MapState.prior_path (e.g. a console reset) still
    # recorded which file it loaded.
    db = _Db(map_state=None, graph_source="/maps/killian/dsg.json")
    assert get_prior_path(db) == "/maps/killian/dsg.json"
    db = _Db(map_state="", graph_source="/maps/killian/dsg.json")
    assert get_prior_path(db) == "/maps/killian/dsg.json"


def test_a_map_not_loaded_from_a_file_has_no_prior():
    assert get_prior_path(_Db()) == ""
