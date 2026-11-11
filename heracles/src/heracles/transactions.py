"""Revisioned, auditable Neo4j transactions for scene-graph mutations."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any


class RevisionConflict(RuntimeError):
    """Raised before mutation when a caller planned against a stale graph."""


class GraphUnitOfWork:
    """Own one revision lock, mutation audit record, commit, and rollback.

    ``prepare`` (optional) runs first in the transaction, with this unit of
    work, before the revision is read and locked. A caller that also locks
    other nodes uses it to take them in a fixed order; if it raises, the
    transaction is rolled back.
    """

    STATE_KEY = "open_set_graph"

    def __init__(
        self,
        driver,
        *,
        database: str = "neo4j",
        expected_revision: int,
        mutation_id: str,
        provenance: Mapping[str, Any],
        annotations: Mapping[str, Any] | None = None,
        prepare: Callable[["GraphUnitOfWork"], None] | None = None,
    ):
        self.driver = driver
        self.database = database
        self.expected_revision = int(expected_revision)
        self.mutation_id = str(mutation_id)
        self.provenance = dict(provenance)
        # Extra scalar properties for the GraphMutation audit node (e.g. a map
        # edit's kind, summary and inverse ops); settable until commit().
        self.annotations: dict[str, Any] = dict(annotations or {})
        self.prepare = prepare
        self._session = None
        self._transaction = None
        self._current_revision: int | None = None
        self._committed_revision: int | None = None

    def _abort(self) -> None:
        if self._transaction is not None:
            self._transaction.rollback()
            self._transaction = None
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "GraphUnitOfWork":
        self._session = self.driver.session(database=self.database)
        self._transaction = self._session.begin_transaction()
        try:
            if self.prepare is not None:
                self.prepare(self)
            record = self._transaction.run(
                "MERGE (s:_State {key:$key}) "
                "ON CREATE SET s.revision=0, s.created_at=datetime() "
                "SET s.revision=s.revision "
                "RETURN s.revision AS revision",
                key=self.STATE_KEY,
            ).single(strict=True)
        except BaseException:
            self._abort()
            raise
        self._current_revision = int(record["revision"])
        if self._current_revision != self.expected_revision:
            self._abort()
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
            "SET m += $annotations "
            "CREATE (m)-[:ADVANCED]->(s)",
            key=self.STATE_KEY,
            revision=revision,
            previous_revision=self.current_revision,
            mutation_id=self.mutation_id,
            provenance_json=json.dumps(self.provenance, sort_keys=True, default=str),
            annotations=self.annotations,
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
        self._abort()
