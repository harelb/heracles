"""Safe admitted-scene query compilation for model-facing clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALLOWED_FIELDS = {
    "nodeSymbol",
    "entity_kind",
    "class",
    "name",
    "layer",
    "position",
    "bounding_box",
    "confidence",
    "last_observed_at",
    "source",
    "admission_status",
}

ALLOWED_ENTITY_KINDS = {
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

ENTITY_KIND_EXPRESSION = """coalesce(n.entity_kind, CASE
    WHEN n.attr_type CONTAINS 'Object' OR n.nodeSymbol STARTS WITH 'O' THEN 'object'
    WHEN n.attr_type CONTAINS 'Agent' OR n.nodeSymbol STARTS WITH 'a' THEN 'agent'
    WHEN n.attr_type CONTAINS 'SubKeyframe' OR n.nodeSymbol STARTS WITH 's' THEN 'keyframe'
    WHEN n.attr_type CONTAINS 'TravNode' OR n.nodeSymbol STARTS WITH 't' THEN 'traversability'
    WHEN n.attr_type CONTAINS 'Room' OR n.nodeSymbol STARTS WITH 'R' THEN 'room'
    WHEN n.attr_type CONTAINS 'Building' OR n.nodeSymbol STARTS WITH 'B' THEN 'building'
    WHEN n.attr_type CONTAINS 'Region' THEN 'region'
    WHEN n.attr_type CONTAINS 'Place' OR n.nodeSymbol STARTS WITH 'p' THEN 'place'
    ELSE 'other' END)"""


def compile_admitted_query(spec: Mapping[str, Any]) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    """Compile a small query AST; raw Cypher is deliberately not accepted."""

    limit = min(max(int(spec.get("limit", 50)), 1), 100)
    select = tuple(spec.get("select") or ("nodeSymbol", "class", "position"))
    invalid = set(select) - ALLOWED_FIELDS
    if invalid:
        raise ValueError(f"unsupported scene query fields: {sorted(invalid)}")
    aggregation = str(spec.get("aggregation", "rows"))
    if aggregation not in {"rows", "count", "distinct_classes"}:
        raise ValueError(f"unsupported aggregation: {aggregation}")

    clauses = [
        "n.nodeSymbol IS NOT NULL",
        "n.retired_at_revision IS NULL",
        "n.admission_status IN ['trusted_prior', 'admitted']",
    ]
    parameters: dict[str, Any] = {"limit": limit + 1}
    classes = tuple(str(value) for value in spec.get("classes") or ())
    symbols = tuple(str(value) for value in spec.get("symbols") or ())
    layers = tuple(int(value) for value in spec.get("layers") or ())
    entity_kinds = tuple(str(value) for value in spec.get("entity_kinds") or ())
    invalid_entity_kinds = set(entity_kinds) - ALLOWED_ENTITY_KINDS
    if invalid_entity_kinds:
        raise ValueError(f"unsupported entity kinds: {sorted(invalid_entity_kinds)}")
    if entity_kinds:
        clauses.append(f"{ENTITY_KIND_EXPRESSION} IN $entity_kinds")
        parameters["entity_kinds"] = list(entity_kinds)
    if classes:
        clauses.append("toLower(coalesce(n.class,n.name,'')) IN $classes")
        parameters["classes"] = [value.casefold() for value in classes]
    if symbols:
        clauses.append("n.nodeSymbol IN $symbols")
        parameters["symbols"] = list(symbols)
    if layers:
        clauses.append("n.layer IN $layers")
        parameters["layers"] = list(layers)
    if spec.get("name_contains"):
        clauses.append("toLower(coalesce(n.name,n.class,'')) CONTAINS $name_contains")
        parameters["name_contains"] = str(spec["name_contains"]).casefold()
    near = spec.get("near_position")
    distance = spec.get("max_distance_m")
    if (near is None) != (distance is None):
        raise ValueError("near_position and max_distance_m must be supplied together")
    if near is not None:
        if len(near) != 3:
            raise ValueError("near_position must have exactly three coordinates")
        clauses.extend(
            (
                "n.position IS NOT NULL",
                "size(n.position) >= 3",
                "sqrt((n.position[0]-$x)^2 + (n.position[1]-$y)^2 + "
                "(n.position[2]-$z)^2) <= $max_distance_m",
            )
        )
        parameters.update(
            x=float(near[0]),
            y=float(near[1]),
            z=float(near[2]),
            max_distance_m=float(distance),
        )

    query = "MATCH (n:SceneEntity) WHERE " + " AND ".join(clauses)
    if aggregation == "count":
        return query + " RETURN count(DISTINCT n) AS count", parameters, ("count",)
    if aggregation == "distinct_classes":
        return (
            query
            + " WITH n ORDER BY n.nodeSymbol "
            + "RETURN coalesce(n.class,n.name,'') AS class, count(*) AS count, "
            + "collect(DISTINCT n.nodeSymbol)[..10] AS nodeSymbols "
            + "ORDER BY count DESC, class LIMIT $limit",
            parameters,
            ("class", "count", "nodeSymbols"),
        )

    projections = [
        f"{ENTITY_KIND_EXPRESSION} AS `entity_kind`"
        if field == "entity_kind"
        else f"n.`{field}` AS `{field}`"
        for field in select
    ]
    columns = list(select)
    if spec.get("include_neighbors"):
        query += (
            " OPTIONAL MATCH (n)-[neighbor_edge:SCENE_EDGE]-(neighbor:SceneEntity) "
            "WHERE neighbor.nodeSymbol IS NOT NULL "
            "AND neighbor_edge.retired_at_revision IS NULL "
            "AND neighbor.retired_at_revision IS NULL "
            "AND neighbor.admission_status IN ['trusted_prior', 'admitted']"
        )
        projections.append("collect(DISTINCT neighbor.nodeSymbol) AS neighbors")
        columns.append("neighbors")
    query += " RETURN " + ", ".join(projections) + " ORDER BY n.nodeSymbol LIMIT $limit"
    return query, parameters, tuple(columns)


def execute_admitted_query(
    driver,
    spec: Mapping[str, Any],
    *,
    database: str = "neo4j",
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Execute one compiled read in a consistent Neo4j transaction."""

    import neo4j

    cypher, parameters, columns = compile_admitted_query(spec)
    with driver.session(database=database, default_access_mode=neo4j.READ_ACCESS) as session:
        with session.begin_transaction(timeout=timeout_s) as tx:
            state = tx.run(
                "MATCH (s:_State {key:'open_set_graph'}) RETURN s.revision AS revision"
            ).single(strict=True)
            result = tx.run(cypher, **parameters)
            rows = [dict(record) for record in result]
            revision = int(state["revision"])
            tx.commit()
    limit = min(max(int(spec.get("limit", 50)), 1), 100)
    truncated = len(rows) > limit
    rows = rows[:limit]
    cited = tuple(
        dict.fromkeys(
            str(symbol)
            for row in rows
            for symbol in (
                [row["nodeSymbol"]]
                if row.get("nodeSymbol") is not None
                else row.get("nodeSymbols") or ()
            )
        )
    )
    return {
        "graph_revision": revision,
        "cypher": cypher,
        "parameters": parameters,
        "columns": columns,
        "rows": tuple(rows),
        "cited_entity_ids": cited,
        "truncated": truncated,
    }
