"""The sizes of the three ``*_activated`` relations, natively, and the queries
to count them in the browser: ``python check_activated.py QUERIES_OUT.json``
writes the queries for ``browser.js queries`` and prints the native counts.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web

NAMES = ("schaefer_activated", "destrieux_activated", "tian_activated")
Path(sys.argv[1]).write_text(json.dumps([
    {"name": name, "datalog": f"ans(l, s) :- {name}(l, s)"} for name in NAMES
]))
with tempfile.TemporaryDirectory() as tmp:
    nl = engine_registry.build_engine(
        "neurosynth", Path.home() / "neurolang_data", 2.0,
        yaml_path=web.lightweight_engines_yaml(tmp),
    )
repl = web.WebRepl(nl, "neurosynth", 2)
for name in NAMES:
    [table] = repl.submit(f"ans(l, s) :- {name}(l, s)").tables
    print(name, "native rows:", table["total"])
