"""Can the built engine of the browser REPL be serialised, and how big is it?

    python pickle_engine.py [DATA_DIR]

Builds the engine (neurosynth at 2 mm without the heavy analyses), pickles it,
loads it back and checks that a query gives the same rows.
"""
import io
import pickle
import sys
import tempfile
import time
from pathlib import Path

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "neurolang_data"
QUERY = "ans(s) :- mentions(s, 'pain')"

with tempfile.TemporaryDirectory() as tmp:
    start = time.perf_counter()
    nl = engine_registry.build_engine(
        "neurosynth", DATA_DIR, 2.0, yaml_path=web.lightweight_engines_yaml(tmp)
    )
    print(f"built in {time.perf_counter() - start:.1f} s")
expected = sorted(nl.execute_datalog_program(QUERY).as_pandas_dataframe().iloc[:, 0])

try:
    start = time.perf_counter()
    data = pickle.dumps(nl, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"pickled: {len(data) / 1048576:.0f} MB in {time.perf_counter() - start:.1f} s")
except Exception as error:
    print("pickle fails:", type(error).__name__, str(error)[:300])
    # which attribute of the engine fails?
    for name, value in vars(nl).items():
        try:
            pickle.dumps(value)
        except Exception as inner:
            print(f"  attribute {name!r} ({type(value).__name__}): {type(inner).__name__}: {str(inner)[:160]}")
    sys.exit(1)

start = time.perf_counter()
loaded = pickle.loads(data)
print(f"loaded in {time.perf_counter() - start:.1f} s")
rows = sorted(loaded.execute_datalog_program(QUERY).as_pandas_dataframe().iloc[:, 0])
print("same rows:", rows == expected, len(rows))
