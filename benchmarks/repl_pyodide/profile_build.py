"""Where the time of building the browser's engine goes, phase by phase.

    python profile_build.py [--cprofile] [--nilearn-first] [DATA_DIR]

The engine is the page's: neurosynth at 2 mm without the heavy analyses.
Each ``_phase_*`` of ``engine_registry.build_engine`` is timed; with
``--cprofile`` the 25 most expensive functions overall are listed too.
"""
import cProfile
import importlib
import pstats
import sys
import tempfile
import time
from pathlib import Path

IMPORTS = []


def timed_import(name):
    """Import *name* and note how long it took (what it imported included)."""
    start = time.perf_counter()
    module = importlib.import_module(name)
    IMPORTS.append((name, time.perf_counter() - start))
    return module


# The libraries first, so that what NeuroLang imports is not charged to them.
for library in ("numpy", "pandas"):
    timed_import(library)
if "--nilearn-first" in sys.argv:  # what importing nilearn costs, apart from the build
    for library in ("scipy.stats", "sklearn.base", "nilearn.datasets", "nilearn.image"):
        timed_import(library)
engine_registry = timed_import("neurolang_apps.utils.engine_registry")
web = timed_import("neurolang_apps.utils.tui.web")

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

print("\nimports (s, what they import included):")
for name, seconds in IMPORTS:
    print(f"  {seconds:6.2f} s  {name}")
print("loaded after the build:", ", ".join(
    f"{name}{'' if name in sys.modules else ' (no)'}" for name in (
        "nilearn", "sklearn", "scipy", "matplotlib", "rdflib", "lark",
        "rich", "prompt_toolkit", "nibabel", "pyarrow", "requests", "httpx",
    )
))
print(f"\nbuild_engine: {total:.1f} s")
for name, seconds in sorted(timings, key=lambda t: -t[1]):
    print(f"  {seconds:6.1f} s  {100 * seconds / total:4.0f}%  {name}")
if profiler:
    pstats.Stats(profiler).sort_stats("cumulative").print_stats(25)
