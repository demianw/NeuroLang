"""A function-level profile of one phase of building the engine.

    python profile_phase.py PHASE [DATA_DIR]      e.g. python_init, or the
                                                  mask: resolve_mask

Builds the engine once to warm the files, then profiles the phase alone.
"""
import cProfile
import pstats
import sys
import tempfile
from pathlib import Path

from neurolang_apps.utils import engine_registry
from neurolang_apps.utils.tui import web

phase = sys.argv[1]
data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "neurolang_data"
profiler = cProfile.Profile()
target = (
    "_resolve_mask" if phase == "resolve_mask" else f"_phase_{phase}"
)
original = getattr(engine_registry, target)


def profiled(*args, **kwargs):
    profiler.enable()
    try:
        return original(*args, **kwargs)
    finally:
        profiler.disable()


setattr(engine_registry, target, profiled)
with tempfile.TemporaryDirectory() as tmp:
    engine_registry.build_engine(
        "neurosynth", data_dir, 2.0, yaml_path=web.lightweight_engines_yaml(tmp)
    )
pstats.Stats(profiler).sort_stats("cumulative").print_stats(22)
