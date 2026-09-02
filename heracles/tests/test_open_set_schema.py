from heracles.open_set_schema import (
    OPEN_SET_SCHEMA_VERSION,
    REQUIRED_CONSTRAINTS,
    REQUIRED_INDEXES,
    SchemaContractError,
    ensure_open_set_schema,
    inspect_open_set_schema,
)


class _Driver:
    def __init__(self):
        self.statements = []
        self.constraints = set()
        self.indexes = set()
        self.version = None

    def execute_query(self, query, **parameters):
        self.statements.append((query, parameters))
        if query.startswith("CREATE CONSTRAINT"):
            self.constraints.add(query.split()[2])
            return [], None, None
        if query.startswith("CREATE INDEX"):
            self.indexes.add(query.split()[2])
            return [], None, None
        if query.startswith("MERGE (s:_Schema"):
            self.version = parameters["version"]
            return [], None, None
        if query.startswith("SHOW CONSTRAINTS"):
            return [{"name": value} for value in sorted(self.constraints)], None, None
        if query.startswith("SHOW INDEXES"):
            return [{"name": value} for value in sorted(self.indexes)], None, None
        if query.startswith("MATCH (s:_Schema"):
            rows = [{"version": self.version}] if self.version is not None else []
            return rows, None, None
        raise AssertionError(query)


def test_schema_provisioning_is_idempotent_and_inspectable():
    driver = _Driver()

    first = ensure_open_set_schema(driver)
    second = ensure_open_set_schema(driver)

    assert first.valid and second.valid
    assert first.schema_version == OPEN_SET_SCHEMA_VERSION
    assert set(first.constraints) == REQUIRED_CONSTRAINTS
    assert set(first.indexes) == REQUIRED_INDEXES
    assert inspect_open_set_schema(driver).valid


def test_schema_provisioning_refuses_to_downgrade_a_newer_database():
    import pytest

    driver = _Driver()
    driver.version = OPEN_SET_SCHEMA_VERSION + 1

    with pytest.raises(SchemaContractError, match="newer than supported"):
        ensure_open_set_schema(driver)
