"""Revisioned, auditable Neo4j transactions for scene-graph mutations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class RevisionConflict(RuntimeError):
    """Raised before mutation when a caller planned against a stale graph."""


class GraphUnitOfWork:
    """Own one revision lock, mutation audit record, commit, and rollback."""

    STATE_KEY = "open_set_graph"

    def __init__(
        self,
        driver,
        *,
        database: str = "neo4j",
        expected_revision: int,
        mutation_id: str,
        provenance: Mapping[str, Any],
    ):
        self.driver = driver
        self.database = database
        self.expected_revision = int(expected_revision)
        self.mutation_id = str(mutation_id)
        self.provenance = dict(provenance)
        self._session = None
        self._transaction = None
        self._current_revision: int | None = None
        self._committed_revision: int | None = None

    def __enter__(self) -> "GraphUnitOfWork":
        self._session = self.driver.session(database=self.database)
        self._transaction = self._session.begin_transaction()
        record = self._transaction.run(
            "MERGE (s:_State {key:$key}) "
            "ON CREATE SET s.revision=0, s.created_at=datetime() "
            "SET s.revision=s.revision "
            "RETURN s.revision AS revision",
            key=self.STATE_KEY,
        ).single(strict=True)
        self._current_revision = int(record["revision"])
        if self._current_revision != self.expected_revision:
            self._transaction.rollback()
            self._transaction = None
            self._session.close()
            self._session = None
            raise RevisionConflict(
                f"expected graph revision {self.expected_revision}, found {self._current_revision}"
            )
        return self

    @property
    def current_revision(self) -> int:
        if self._current_revision is None:
            raise RuntimeError("unit of work has not been entered")
        return self._current_revision

    @property
    def next_revision(self) -> int:
        return self.current_revision + 1

    def run(self, query: str, **parameters):
        if self._transaction is None:
            raise RuntimeError("unit of work is not active")
        return self._transaction.run(query, **parameters)

    def retire_entity(self, node_symbol: str, *, reason: str) -> None:
        result = self.run(
            "MATCH (n:SceneEntity {nodeSymbol:$node_symbol}) "
            "WHERE n.retired_at_revision IS NULL "
            "SET n.retired_at_revision=$revision, n.retirement_reason=$reason "
            "RETURN count(n) AS changed",
            node_symbol=node_symbol,
            revision=self.next_revision,
            reason=reason,
        ).single(strict=True)
        if int(result["changed"]) != 1:
            raise KeyError(f"active scene entity does not exist: {node_symbol}")

    def supersede_entity(
        self,
        old_symbol: str,
        new_symbol: str,
        properties: Mapping[str, Any],
        *,
        reason: str,
    ) -> None:
        self.retire_entity(old_symbol, reason=reason)
        self.run(
            "CREATE (n:SceneEntity {nodeSymbol:$new_symbol}) "
            "SET n += $properties, n.supersedes=$old_symbol, n.created_at_revision=$revision",
            new_symbol=new_symbol,
            old_symbol=old_symbol,
            properties=dict(properties),
            revision=self.next_revision,
        )

    def commit(self) -> int:
        if self._transaction is None:
            if self._committed_revision is not None:
                return self._committed_revision
            raise RuntimeError("unit of work is not active")
        revision = self.next_revision
        self._transaction.run(
            "MATCH (s:_State {key:$key}) "
            "SET s.revision=$revision, s.updated_at=datetime() "
            "CREATE (m:GraphMutation {mutation_id:$mutation_id, "
            "previous_revision:$previous_revision, revision:$revision, "
            "provenance_json:$provenance_json, committed_at:datetime()}) "
            "CREATE (m)-[:ADVANCED]->(s)",
            key=self.STATE_KEY,
            revision=revision,
            previous_revision=self.current_revision,
            mutation_id=self.mutation_id,
            provenance_json=json.dumps(self.provenance, sort_keys=True, default=str),
        )
        self._transaction.commit()
        self._transaction = None
        self._committed_revision = revision
        return revision

    def rollback(self) -> None:
        if self._transaction is not None:
            self._transaction.rollback()
            self._transaction = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._transaction is not None:
            self._transaction.rollback()
            self._transaction = None
        if self._session is not None:
            self._session.close()
            self._session = None


def current_graph_revision(driver, *, database: str = "neo4j") -> int:
    records, _, _ = driver.execute_query(
        "MERGE (s:_State {key:$key}) ON CREATE SET s.revision=0 "
        "RETURN s.revision AS revision",
        key=GraphUnitOfWork.STATE_KEY,
        database_=database,
    )
    return int(records[0]["revision"])


def admitted_objects(
    driver,
    *,
    class_name: str | None = None,
    database: str = "neo4j",
) -> tuple[dict[str, Any], ...]:
    """Return active planning facts, excluding candidates and rejections."""
    records, _, _ = driver.execute_query(
        "MATCH (n) WHERE (n:Object OR n:SceneEntity) "
        "AND n.retired_at_revision IS NULL "
        "AND n.admission_status IN ['trusted_prior','admitted'] "
        "AND ($class_name IS NULL OR "
        "toLower(coalesce(n.class,n.name,''))=toLower($class_name)) "
        "RETURN n.nodeSymbol AS node_symbol, properties(n) AS properties",
        class_name=class_name,
        database_=database,
    )
    return tuple(
        {"node_symbol": record["node_symbol"], **dict(record["properties"])}
        for record in records
    )

