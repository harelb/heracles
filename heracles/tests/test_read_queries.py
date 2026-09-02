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
    assert "admission_status IN ['trusted_prior', 'admitted']" in query
    assert "retired_at_revision IS NULL" in query
    assert parameters["classes"] == ["coffee mug' match (x) delete x //"]
    assert columns == ("nodeSymbol", "class")


def test_compiler_rejects_unlisted_properties():
    import pytest

    with pytest.raises(ValueError, match="unsupported scene query fields"):
        compile_admitted_query({"select": ["password"]})
