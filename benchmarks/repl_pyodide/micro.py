"""A small CPU benchmark, the same natively and in Pyodide.

It times pure Python, numpy and pandas, the three things the engine is made
of, so that the slowdown of a browser can be told from NeuroLang's. It runs
natively as ``python micro.py`` and in Pyodide through ``browser.js micro``.
"""
import json
import time


def best(function, repeats=3):
    """The shortest of *repeats* runs of *function*, in seconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times)


def run() -> dict:
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    matrix = rng.random((1200, 1200))
    frame = pd.DataFrame({
        "k": rng.integers(0, 50_000, 2_000_000), "v": rng.random(2_000_000),
    })
    left = pd.DataFrame({"k": rng.permutation(1_000_000), "a": 1.0})
    right = pd.DataFrame({"k": rng.permutation(1_000_000), "b": 2.0})
    strings = pd.Series([f"term{i % 3000}" for i in range(500_000)])
    return {
        "python loop, 3M iterations": best(lambda: sum(i * i for i in range(3_000_000))),
        "numpy 1200x1200 matmul": best(lambda: matrix @ matrix),
        "numpy sort of 3M floats": best(lambda: np.sort(rng.random(3_000_000))),
        "pandas groupby sum, 2M rows": best(lambda: frame.groupby("k")["v"].sum()),
        "pandas merge, 1M x 1M": best(lambda: left.merge(right, on="k")),
        "pandas str.startswith, 500k": best(lambda: strings.str.startswith("term1")),
        "versions": f"numpy {np.__version__}, pandas {pd.__version__}",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
