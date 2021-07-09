import pytest
from .. import config
from ...utils.relational_algebra_set import pandas, dask_sql

@pytest.fixture
def clean_config_fixture():
    backend = config["RAS"]["backend"]
    try:
        yield
    finally:
        config.set_query_backend(backend)

def test_config_set_query_backend(clean_config_fixture):
    with pytest.raises(ValueError):
        config.set_query_backend("hello")

    config.set_query_backend("dask")
    assert config["RAS"]["backend"] == "dask"
    from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
    assert RelationalAlgebraFrozenSet is dask_sql.RelationalAlgebraFrozenSet

    config.set_query_backend("pandas")
    assert config["RAS"]["backend"] == "pandas"
    from ...utils.relational_algebra_set import RelationalAlgebraFrozenSet
    assert RelationalAlgebraFrozenSet is pandas.RelationalAlgebraFrozenSet