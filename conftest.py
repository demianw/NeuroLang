"""Pytest configuration file.

NOTE: Do not remove the unused dask_sql import !

The dask_sql library uses jpype to start a JVM and access Java objects
from python. This JVM needs to be started before we import neurolang,
otherwise it can cause major side-effects which are hard to track. See
https://github.com/jpype-project/jpype/issues/933 for reference.
"""

import warnings
from typing import Any, Generator

import pytest
from neurolang.probabilistic import containment, dalvi_suciu_lift

try:
    import importlib.util
    dask_sql_spec = importlib.util.find_spec("dask_sql")
    if dask_sql_spec is not None:
        from neurolang.utils.relational_algebra_set.dask_helpers import (
            DaskContextManager,
        )
        HAS_DASK: bool = True
    else:
        HAS_DASK: bool = False
except ImportError:
    HAS_DASK: bool = False
from neurolang import config


def pytest_addoption(parser: Any) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Modify test collection to skip slow tests unless --runslow is given."""
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_sessionstart(session: pytest.Session) -> None:
    """Ensure pytest session is properly configured before running tests.

    The dask-sql library uses the jpype library which starts a JVM and allows
    us to use Java classes from Python. But the JVM will trigger a
    segmentation fault when starting and when interrupting threads and Pythons
    fault handler can intercept these operations and interpret these as
    real faults. So we need to disable faulthandlers which pytest starts
    otherwise we get segmentation faults when running the tests.
    See (https://jpype.readthedocs.io/en/latest/userguide.html#errors-reported-by-python-fault-handler)
    """
    try:
        import faulthandler

        faulthandler.enable()
        faulthandler.disable()
    except ImportError:
        warnings.warn("Faulthandler lib not available.")


@pytest.fixture(autouse=config["RAS"].get("backend", "pandas") == "dask")
def clear_dask_context_after_test_module() -> Generator[int, None, None]:
    """Clear Dask context after each test module to prevent clustering issues."""
    yield 0

    DaskContextManager._context = None


@pytest.fixture(autouse=True)
def clear_probabilistic_caches() -> Generator[None, None, None]:
    """Clear probabilistic resolution caches before each test.

    This avoids stale state across test boundaries.
    """
    dalvi_suciu_lift.clear_cache()
    containment.clear_cache()
    yield
    dalvi_suciu_lift.clear_cache()
    containment.clear_cache()
