"""Time the benchmark's queries natively, through the code of the REPL page.

The engine is the browser's (neurosynth at 2 mm without the heavy analyses),
the REPL is ``WebRepl`` that the page's worker drives, and the time is the
one the REPL prints, so that the numbers compare with those of Pyodide.

    python native.py QUERIES.json OUT.json [REPEATS]
"""
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web

queries = json.loads(Path(sys.argv[1]).read_text())
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 2

t0 = time.perf_counter()
with tempfile.TemporaryDirectory() as tmp:
    nl = engine_registry.build_engine(
        "neurosynth", Path.home() / "neurolang_data", 2.0,
        yaml_path=web.lightweight_engines_yaml(tmp),
    )
build = time.perf_counter() - t0
repl = web.WebRepl(nl, "neurosynth", 2)
results = {"engine build s": round(build, 1), "queries": []}
print("engine built in", results["engine build s"], "s", flush=True)
for query in queries:
    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        reply = repl.submit(query["datalog"])
        wall = time.perf_counter() - start
        done = re.search(r"Query completed in ([\d.]+) s", reply.html)
        rows = [t["total"] for t in reply.tables]
        runs.append({
            "s": float(done.group(1)) if done else None, "wall s": round(wall, 2),
            "failed": reply.failed, "rows": rows,
        })
        print(query["name"], runs[-1], flush=True)
    results["queries"].append({"name": query["name"], "runs": runs})
Path(sys.argv[2]).write_text(json.dumps(results, indent=1))
