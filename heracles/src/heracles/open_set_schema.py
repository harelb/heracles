"""Versioned Neo4j schema and data-contract checks for open-set navigation.

This module deliberately contains no agent-framework dependencies.  Heracles
owns the persisted graph contract; callers may provision it at startup and
inspect it read-only during preflight checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


OPEN_SET_SCHEMA_KEY = "open_set_navigation"
OPEN_SET_SCHEMA_VERSION = 1

REQUIRED_CONSTRAINTS = frozenset(
    {
        "open_set_scene_entity_symbol_unique",
        "open_set_state_key_unique",
        "open_set_mutation_id_unique",
        "open_set_schema_key_unique",
    }
)
REQUIRED_INDEXES = frozenset(
    {
        "open_set_scene_entity_kind_class",
        "open_set_scene_entity_admission",
    }
)

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT open_set_scene_entity_symbol_unique IF NOT EXISTS "
    "FOR (n:SceneEntity) REQUIRE n.nodeSymbol IS UNIQUE",
    "CREATE CONSTRAINT open_set_state_key_unique IF NOT EXISTS "
    "FOR (n:_State) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT open_set_mutation_id_unique IF NOT EXISTS "
    "FOR (n:GraphMutation) REQUIRE n.mutation_id IS UNIQUE",
    "CREATE CONSTRAINT open_set_schema_key_unique IF NOT EXISTS "
    "FOR (n:_Schema) REQUIRE n.key IS UNIQUE",
    "CREATE INDEX open_set_scene_entity_kind_class IF NOT EXISTS "
    "FOR (n:SceneEntity) ON (n.entity_kind, n.class)",
    "CREATE INDEX open_set_scene_entity_admission IF NOT EXISTS "
    "FOR (n:SceneEntity) ON (n.admission_status)",
)

VALID_ENTITY_KINDS = frozenset(
    {
        "object",
        "place",
        "region",
        "room",
        "building",
        "agent",
        "keyframe",
        "traversability",
        "other",
    }
)
VALID_ADMISSION_STATUSES = frozenset(
    {"trusted_prior", "candidate", "admitted", "rejected"}
)


class SchemaContractError(RuntimeError):
    """The database schema or stored scene violates the application contract."""


@dataclass(frozen=True)
class SchemaReport:
    schema_version: int | None
    constraints: tuple[str, ...]
    indexes: tuple[str, ...]
    missing_constraints: tuple[str, ...]
    missing_indexes: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.schema_version == OPEN_SET_SCHEMA_VERSION
            and not self.missing_constraints
            and not self.missing_indexes
        )

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


@dataclass(frozen=True)
class SceneDataReport:
    total_entities: int
    total_edges: int
    violations: dict[str, int]
    inventory: tuple[dict[str, Any], ...]

    @property
    def valid(self) -> bool:
        return not any(self.violations.values())

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "valid": self.valid}


def ensure_open_set_schema(driver, *, database: str = "neo4j") -> SchemaReport:
    """Idempotently migrate an empty or compatible database to the current schema."""

    version_rows, _, _ = driver.execute_query(
        "MATCH (s:_Schema {key:$key}) RETURN s.version AS version",
        key=OPEN_SET_SCHEMA_KEY,
        database_=database,
    )
    installed = int(version_rows[0]["version"]) if version_rows else None
    if installed is not None and installed > OPEN_SET_SCHEMA_VERSION:
        raise SchemaContractError(
            f"database schema version {installed} is newer than supported "
            f"version {OPEN_SET_SCHEMA_VERSION}"
        )
    for statement in SCHEMA_STATEMENTS:
        driver.execute_query(statement, database_=database)
    driver.execute_query(
        "MERGE (s:_Schema {key:$key}) "
        "ON CREATE SET s.created_at=datetime() "
        "SET s.version=$version, s.updated_at=datetime()",
        key=OPEN_SET_SCHEMA_KEY,
        version=OPEN_SET_SCHEMA_VERSION,
        database_=database,
    )
    report = inspect_open_set_schema(driver, database=database)
    if not report.valid:
        raise SchemaContractError(f"failed to provision open-set schema: {report.as_dict()}")
    return report


def inspect_open_set_schema(driver, *, database: str = "neo4j") -> SchemaReport:
    """Read the installed contract without modifying the database."""

    constraint_rows, _, _ = driver.execute_query(
        "SHOW CONSTRAINTS YIELD name RETURN name ORDER BY name",
        database_=database,
    )
    index_rows, _, _ = driver.execute_query(
        "SHOW INDEXES YIELD name RETURN name ORDER BY name",
        database_=database,
    )
    version_rows, _, _ = driver.execute_query(
        "MATCH (s:_Schema {key:$key}) RETURN s.version AS version",
        key=OPEN_SET_SCHEMA_KEY,
        database_=database,
    )
    constraints = tuple(str(row["name"]) for row in constraint_rows)
    indexes = tuple(str(row["name"]) for row in index_rows)
    version = int(version_rows[0]["version"]) if version_rows else None
    return SchemaReport(
        schema_version=version,
        constraints=constraints,
        indexes=indexes,
        missing_constraints=tuple(sorted(REQUIRED_CONSTRAINTS - set(constraints))),
        missing_indexes=tuple(sorted(REQUIRED_INDEXES - set(indexes))),
    )


def inspect_scene_data(driver, *, database: str = "neo4j") -> SceneDataReport:
    """Check graph-wide invariants without changing graph state."""

    entity_rows, _, _ = driver.execute_query(
        """
        MATCH (n:SceneEntity)
        RETURN count(n) AS total,
          sum(CASE WHEN n.nodeSymbol IS NULL OR trim(toString(n.nodeSymbol)) = ''
              THEN 1 ELSE 0 END) AS missing_node_symbol,
          sum(CASE WHEN n.attr_type IS NULL OR trim(toString(n.attr_type)) = ''
              THEN 1 ELSE 0 END) AS missing_attr_type,
          sum(CASE WHEN n.entity_kind IS NULL OR NOT n.entity_kind IN $entity_kinds
              THEN 1 ELSE 0 END)
              AS invalid_entity_kind,
          sum(CASE WHEN n.admission_status IS NULL
              OR NOT n.admission_status IN $admission_statuses THEN 1 ELSE 0 END)
              AS invalid_admission_status,
          sum(CASE
              WHEN n.attr_type CONTAINS 'Object' AND n.entity_kind <> 'object' THEN 1
              WHEN n.attr_type CONTAINS 'Agent' AND n.entity_kind <> 'agent' THEN 1
              WHEN n.attr_type CONTAINS 'SubKeyframe' AND n.entity_kind <> 'keyframe' THEN 1
              WHEN n.attr_type CONTAINS 'TravNode' AND n.entity_kind <> 'traversability' THEN 1
              ELSE 0 END) AS attr_kind_mismatch
        """,
        entity_kinds=sorted(VALID_ENTITY_KINDS),
        admission_statuses=sorted(VALID_ADMISSION_STATUSES),
        database_=database,
    )
    duplicate_rows, _, _ = driver.execute_query(
        """
        MATCH (n:SceneEntity)
        WITH n.nodeSymbol AS symbol, count(*) AS occurrences
        WHERE symbol IS NOT NULL AND occurrences > 1
        RETURN count(*) AS duplicate_node_symbols
        """,
        database_=database,
    )
    edge_rows, _, _ = driver.execute_query(
        """
        MATCH (:SceneEntity)-[r:SCENE_EDGE]->(:SceneEntity)
        RETURN count(r) AS total,
          sum(CASE WHEN r.edge_id IS NULL OR trim(toString(r.edge_id)) = ''
              THEN 1 ELSE 0 END) AS missing_edge_id
        """,
        database_=database,
    )
    duplicate_edge_rows, _, _ = driver.execute_query(
        """
        MATCH (:SceneEntity)-[r:SCENE_EDGE]->(:SceneEntity)
        WITH r.edge_id AS edge_id, count(*) AS occurrences
        WHERE edge_id IS NOT NULL AND occurrences > 1
        RETURN count(*) AS duplicate_edge_ids
        """,
        database_=database,
    )
    revision_rows, _, _ = driver.execute_query(
        """
        MATCH (s:_State {key:'open_set_graph'})
        OPTIONAL MATCH (m:GraphMutation)
        RETURN s.revision AS state_revision,
          coalesce(max(m.revision), 0) AS maximum_mutation_revision,
          sum(CASE WHEN m IS NOT NULL AND m.revision <> m.previous_revision + 1
              THEN 1 ELSE 0 END) AS invalid_mutation_steps
        """,
        database_=database,
    )
    inventory_rows, _, _ = driver.execute_query(
        """
        MATCH (n:SceneEntity)
        WHERE n.retired_at_revision IS NULL
          AND n.admission_status IN ['trusted_prior', 'admitted']
        RETURN n.entity_kind AS entity_kind, coalesce(n.class, n.name, '') AS class,
          n.attr_type AS attr_type, count(*) AS count
        ORDER BY count DESC, entity_kind, class
        """,
        database_=database,
    )
    entity = dict(entity_rows[0]) if entity_rows else {"total": 0}
    edge = dict(edge_rows[0]) if edge_rows else {"total": 0}
    violations = {
        key: int(entity.get(key) or 0)
        for key in (
            "missing_node_symbol",
            "missing_attr_type",
            "invalid_entity_kind",
            "invalid_admission_status",
            "attr_kind_mismatch",
        )
    }
    violations.update(
        duplicate_node_symbols=int(duplicate_rows[0]["duplicate_node_symbols"] or 0),
        missing_edge_id=int(edge.get("missing_edge_id") or 0),
        duplicate_edge_ids=int(duplicate_edge_rows[0]["duplicate_edge_ids"] or 0),
    )
    if revision_rows:
        revision = dict(revision_rows[0])
        violations["invalid_mutation_steps"] = int(
            revision.get("invalid_mutation_steps") or 0
        )
        violations["revision_audit_mismatch"] = int(
            int(revision.get("state_revision") or 0)
            != int(revision.get("maximum_mutation_revision") or 0)
        )
    return SceneDataReport(
        total_entities=int(entity.get("total") or 0),
        total_edges=int(edge.get("total") or 0),
        violations=violations,
        inventory=tuple(dict(row) for row in inventory_rows),
    )


def assert_scene_data_contract(driver, *, database: str = "neo4j") -> SceneDataReport:
    report = inspect_scene_data(driver, database=database)
    if not report.valid:
        raise SchemaContractError(f"scene graph data contract failed: {report.as_dict()}")
    return report
