"""Read and write a table of the size of the engine's peaks (447 000 rows of
x, y, z and a study) as parquet, npz and csv, natively and in Pyodide.

It runs natively as ``python formats.py`` and in Pyodide through
``browser.js formats``.
"""
import io
import json
import time


def best(function, repeats=3):
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
    n = 447_000
    frame = pd.DataFrame({
        "x": rng.integers(-70, 70, n).astype(float),
        "y": rng.integers(-100, 70, n).astype(float),
        "z": rng.integers(-50, 80, n).astype(float),
        "study": rng.integers(1_000_000, 40_000_000, n),
    })
    result = {}
    npz = io.BytesIO()
    np.savez_compressed(npz, **{c: frame[c].to_numpy() for c in frame})
    result["npz KB"] = round(len(npz.getvalue()) / 1024)
    result["npz read s"] = best(lambda: pd.DataFrame(dict(np.load(io.BytesIO(npz.getvalue())))))
    csv = frame.to_csv(index=False).encode()
    result["csv KB"] = round(len(csv) / 1024)
    result["csv read s"] = best(lambda: pd.read_csv(io.BytesIO(csv)))
    try:
        start = time.perf_counter()
        import pyarrow  # noqa: F401
        result["import pyarrow s"] = round(time.perf_counter() - start, 2)
        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        result["parquet KB"] = round(len(buffer.getvalue()) / 1024)
        result["parquet read s"] = best(lambda: pd.read_parquet(io.BytesIO(buffer.getvalue())))
        result["pyarrow"] = pyarrow.__version__
    except Exception as error:  # noqa: BLE001
        result["parquet"] = f"{type(error).__name__}: {error}"[:200]
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
