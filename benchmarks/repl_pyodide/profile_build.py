"""Where the time of building the browser's engine goes, phase by phase.

    python profile_build.py [--cprofile] [DATA_DIR]

The engine is the page's: neurosynth at 2 mm without the heavy analyses.
Each ``_phase_*`` of ``engine_registry.build_engine`` is timed; with
``--cprofile`` the 25 most expensive functions overall are listed too.
"""
import cProfile
import pstats
import sys
import tempfile
import time
from pathlib import Path

DATA_DIR = ([Path(a) for a in sys.argv[1:] if not a.startswith("--")] or [
    Path.home() / "neurolang_data"
])[0]

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web

timings = []


def timed(name, function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings.append((name, time.perf_counter() - start))
    return wrapper


for name in dir(engine_registry):
    if name.startswith("_phase_"):
        setattr(engine_registry, name, timed(name, getattr(engine_registry, name)))

with tempfile.TemporaryDirectory() as tmp:
    yaml_path = web.lightweight_engines_yaml(tmp)
    start = time.perf_counter()
    profiler = cProfile.Profile() if "--cprofile" in sys.argv else None
    if profiler:
        profiler.enable()
    engine_registry.build_engine(
        "neurosynth", DATA_DIR, 2.0, yaml_path=yaml_path
    )
    if profiler:
        profiler.disable()
    total = time.perf_counter() - start

print(f"\nbuild_engine: {total:.1f} s")
for name, seconds in sorted(timings, key=lambda t: -t[1]):
    print(f"  {seconds:6.1f} s  {100 * seconds / total:4.0f}%  {name}")
if profiler:
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
