"""Safe admitted-scene query compilation for model-facing clients."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ALLOWED_FIELDS = {
    "nodeSymbol",
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

    query = "MATCH (n) WHERE " + " AND ".join(clauses)
    if aggregation == "count":
        return query + " RETURN count(DISTINCT n) AS count", parameters, ("count",)
    if aggregation == "distinct_classes":
        return (
            query
            + " RETURN coalesce(n.class,n.name,'') AS class, count(*) AS count "
            + "ORDER BY count DESC, class LIMIT $limit",
            parameters,
            ("class", "count"),
        )

    projections = [f"n.`{field}` AS `{field}`" for field in select]
    columns = list(select)
    if spec.get("include_neighbors"):
        query += (
            " OPTIONAL MATCH (n)-[:SCENE_EDGE]-(neighbor) "
            "WHERE neighbor.nodeSymbol IS NOT NULL "
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
        str(row["nodeSymbol"])
        for row in rows
        if row.get("nodeSymbol") is not None
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
