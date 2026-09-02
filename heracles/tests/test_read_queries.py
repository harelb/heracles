from heracles.read_queries import compile_admitted_query


def test_compiler_forces_admission_and_parameterizes_values():
    query, parameters, columns = compile_admitted_query(
        {
            "classes": ["coffee mug' MATCH (x) DELETE x //"],
            "select": ["nodeSymbol", "class"],
            "limit": 10,
        }
    )

    assert "DELETE" not in query
    assert query.startswith("MATCH (n:SceneEntity)")
    assert "admission_status IN ['trusted_prior', 'admitted']" in query
    assert "retired_at_revision IS NULL" in query
    assert parameters["classes"] == ["coffee mug' match (x) delete x //"]
    assert columns == ("nodeSymbol", "class")


def test_compiler_rejects_unlisted_properties():
    import pytest

    with pytest.raises(ValueError, match="unsupported scene query fields"):
        compile_admitted_query({"select": ["password"]})


def test_compiler_filters_entity_kinds_and_cites_inventory_symbols():
    query, parameters, columns = compile_admitted_query(
        {
            "entity_kinds": ["object"],
            "aggregation": "distinct_classes",
            "limit": 10,
        }
    )

    assert "entity_kind" in query
    assert "SubKeyframe" in query
    assert "collect(DISTINCT n.nodeSymbol)[..10] AS nodeSymbols" in query
    assert parameters["entity_kinds"] == ["object"]
    assert columns == ("class", "count", "nodeSymbols")


def test_compiler_rejects_unknown_entity_kinds_even_without_pydantic():
    import pytest

    with pytest.raises(ValueError, match="unsupported entity kinds"):
        compile_admitted_query({"entity_kinds": ["all_database_nodes"]})
