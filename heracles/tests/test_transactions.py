import pytest

from heracles.transactions import GraphUnitOfWork, RevisionConflict


class _Result:
    def __init__(self, value):
        self.value = value

    def single(self, strict=False):
        return self.value


class _Transaction:
    def __init__(self, revision=0):
        self.revision = revision
        self.queries = []
        self.committed = False
        self.rolled_back = False

    def run(self, query, **parameters):
        self.queries.append((query, parameters))
        if "RETURN s.revision AS revision" in query:
            return _Result({"revision": self.revision})
        if "RETURN count(n) AS changed" in query:
            return _Result({"changed": 1})
        return _Result({})

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class _Session:
    def __init__(self, transaction):
        self.transaction = transaction
        self.closed = False

    def begin_transaction(self):
        return self.transaction

    def close(self):
        self.closed = True


class _Driver:
    def __init__(self, revision=0):
        self.transaction = _Transaction(revision)
        self.created_session = None

    def session(self, database):
        self.created_session = _Session(self.transaction)
        return self.created_session


def test_commit_advances_exactly_one_revision_and_records_provenance():
    driver = _Driver(revision=7)
    with GraphUnitOfWork(
        driver,
        expected_revision=7,
        mutation_id="patch-1",
        provenance={"source": "oracle"},
    ) as tx:
        tx.run("CREATE (:SceneEntity {nodeSymbol:$id})", id="o1")
        assert tx.commit() == 8

    assert driver.transaction.committed
    assert not driver.transaction.rolled_back
    audit = driver.transaction.queries[-1]
    assert audit[1]["previous_revision"] == 7
    assert audit[1]["revision"] == 8
    assert '\"source\": \"oracle\"' in audit[1]["provenance_json"]


def test_context_without_explicit_commit_rolls_back():
    driver = _Driver()
    with GraphUnitOfWork(
        driver,
        expected_revision=0,
        mutation_id="patch-1",
        provenance={},
    ):
        pass
    assert driver.transaction.rolled_back
    assert not driver.transaction.committed


def test_stale_revision_fails_before_caller_query():
    driver = _Driver(revision=4)
    with pytest.raises(RevisionConflict):
        with GraphUnitOfWork(
            driver,
            expected_revision=3,
            mutation_id="patch-1",
            provenance={},
        ):
            raise AssertionError("unreachable")
    assert driver.transaction.rolled_back
    assert len(driver.transaction.queries) == 1


def test_exception_rolls_back():
    driver = _Driver()
    with pytest.raises(RuntimeError):
        with GraphUnitOfWork(
            driver,
            expected_revision=0,
            mutation_id="patch-1",
            provenance={},
        ) as tx:
            tx.run("CREATE (:SceneEntity)")
            raise RuntimeError("failure")
    assert driver.transaction.rolled_back
