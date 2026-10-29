# The REPL's queries in Pyodide: timings

Measured on 2026-10-29 on one machine: MacBook Pro, Intel Core i9-9880H (8
cores, 2.3 GHz base), 64 GB, macOS 26.7, nothing else running. The code is
that of branch `feat/repl-live-steps-highlight-downloads` (3a8972e2). The
engine is the browser's (Neurosynth at 2 mm, without the voxel maps), the
queries are example-bank queries (`queries.py`), and the time is the one the
REPL prints ("Query completed in … s"), the same code natively (`WebRepl`) and
in the page. Run: `./run_all.sh micro-native | micro BROWSER | native | queries BROWSER`.

| query | native | Chromium 149 | × | WebKit 26.4 | × | Firefox 150 | × |
|---|---|---|---|---|---|---|---|
| select (516 rows) | 1.3 | 1.9 | 1.4 | 2.1 | 1.6 | 13.7 | 10.1 |
| P(term), all 3228 terms | 1.1 | 1.7 | 1.6 | 1.9 | 1.8 | 12.5 | 11.8 |
| P(subcortical region) | 2.1 | 3.2 | 1.5 | 3.4 | 1.6 | 23.6 | 11.3 |
| Bayes factor of a term | 3.0 | 4.5 | 1.5 | 4.9 | 1.7 | 32.6 | 10.9 |
| BF of all terms, a gyrus | 2.8 | 4.3 | 1.6 | 4.7 | 1.7 | 31.8 | 11.4 |
| BF of all terms, an MNI point | 2.0 | 3.2 | 1.6 | 3.6 | 1.8 | 23.4 | 11.8 |
| co-activation, 4950 Schaefer pairs | 2.5 | 3.8 | 1.5 | 4.2 | 1.7 | 29.0 | 11.4 |
| co-activation network, subcortex | 2.3 | 3.6 | 1.5 | 3.9 | 1.7 | 27.0 | 11.5 |
| co-activation more than chance | 2.1 | 3.1 | 1.5 | 3.4 | 1.7 | 24.0 | 11.6 |
| 95% interval of a BF (500 replicates) | 14.9 | 16.7 | 1.1 | 17.8 | 1.2 | 97.7 | 6.6 |
| 95% interval of a difference | 10.2 | 11.0 | 1.1 | 11.5 | 1.1 | 62.4 | 6.1 |

Seconds. Native and Chromium are the mean of two runs (cold and warm agree
within 3%), WebKit and Firefox one run. Starting the page (Pyodide, packages,
the engine built in the tab): Chromium 134 s, WebKit 171 s, Firefox 620 s;
building the same engine natively takes 64 s. The page downloads 31 MB (the
data zip is 29.5 MB) besides Pyodide and its packages.

## CPU micro-benchmark (`micro.py`), seconds

| | native | Chromium | × | WebKit | × | Firefox | × |
|---|---|---|---|---|---|---|---|
| Python loop, 3M iterations | 0.24 | 0.51 | 2.1 | 0.49 | 2.1 | 2.88 | 12.0 |
| numpy 1200×1200 matmul | 0.017 | 2.36 | 137 | 2.54 | 147 | 29.7 | 1715 |
| numpy sort of 3M floats | 0.26 | 0.37 | 1.4 | 0.35 | 1.3 | 1.27 | 4.9 |
| pandas groupby sum, 2M rows | 0.060 | 0.073 | 1.2 | 0.069 | 1.1 | 0.41 | 6.8 |
| pandas merge, 1M × 1M | 0.107 | 0.059 | 0.5 | 0.055 | 0.5 | 0.28 | 2.6 |
| pandas str.startswith, 500k | 0.006 | 0.120 | 22 | 0.111 | 20 | 0.62 | 111 |

Versions differ: native numpy 1.26.4 and pandas 3.0.6, Pyodide numpy 2.4.6 and
pandas 3.0.2.

## Where the startup goes (`./run_all.sh profile-native | profile BROWSER`)

Building the engine, phase by phase (`profile_build.py`), natively and in
Pyodide (Chromium), seconds:

| phase of `build_engine` | native | Pyodide |
|---|---|---|
| `_phase_atlases` (3 atlases, `atlas_activation`) | 47.5 | 98.9 |
| `_phase_ontologies` (Cognitive Atlas OWL) | 3.1 | 4.3 |
| `_phase_python_init` | 1.9 | 2.6 |
| the rest | 3.0 | 6.3 |
| total | 55.5 | 112.1 |

The page also takes, in Pyodide on a local server: 1.9 s to load Pyodide, 3.6 s
the numpy/scipy/pandas/scikit-learn packages, 1.7 s the PyPI packages, 0.5 s the
wheels and the data. `atlas_activation` finds, for each of the 447 000 peaks, the
voxels of the atlas within 3 mm in a cube of 343 candidates, for each atlas, at
every start: 15 s per atlas natively, and its result never changes.

## With the atlas tables kept (branch `feat/repl-atlas-cache`)

`atlas_activation`'s pairs are kept in the data directory, 1.6 MB of `.npz` for
the three atlases (`profile_chromium_cached.json`). Seconds:

| | native | Pyodide, Chromium |
|---|---|---|
| `build_engine`, tables found | 58.1 | 112.1 |
| `build_engine`, tables read | 9.0 | 14.7 |
| `_phase_atlases`, found / read | 50.3 / 1.4 | 98.9 / 1.5 |
| the page ready (Pyodide, packages, engine) | | 127 → 30 |

The file written by CPython (numpy 1.26) was read as it is by Pyodide (numpy
2.4.6). Natively and in the page the three `*_activated` relations have the same
number of rows (298 632, 312 244, 40 675) and the 11 queries above give the same
number of rows (`activated_chromium.json`).

Keeping the whole engine instead, measured (`pickle_engine.py`, `browser.js
snapshot`): the built engine cannot be pickled (`PicklingError`: the
parameterised `Symbol[...]` classes of the expressions cannot be found by name),
and the memory of Pyodide after the build is 781 MB, 268 MB compressed
(`snapshot_heap.json`), more than GitHub accepts in a file of the pages' repository
and eight times the page's whole download; `_makeSnapshot` also stopped on an
import of `requests` (`js.crossOriginIsolated`: Illegal invocation).

## With the mask, the voxels and the atlas fetches kept too

Profiling imports showed what the build asked nilearn for: in Pyodide
`scipy.stats` takes 3.5 s to import, `sklearn.base` 0.4 s and `nilearn.datasets`
2.3 s (natively 1.2 s and 1 s), to find files already on disk and resample the
MNI template (`profile_phase.py`, `import_chains.py`, `--nilearn-first`). Keeping
the resampled mask, the brain and cortex voxels and where the atlases are
(`profile_chromium_cached_all.json`), 3.6 MB in all, a build never imports
nilearn or scikit-learn. Seconds:

| | native | Pyodide, Chromium |
|---|---|---|
| `build_engine`, nothing kept | 58.1 | 112.1 |
| atlas pairs kept (PR #1022, first version) | 9.0 | 14.8 |
| everything kept | 5.8 | 8.0 |
| the page ready, Chromium / WebKit | | 127 → 30 → 23 / 171 → 26 |

All 11 queries give the same number of rows in Pyodide (WebKit) as natively
(`queries_webkit_cached_all.json`).

## File formats in Pyodide (`formats.py`, `browser.js formats`)

A table of the size of the engine's peaks (447 000 rows: x, y, z, study), seconds
to read it; pyarrow 22.0.0 in Pyodide, 18.1.0 natively (`formats_chromium.json`):

| | size | native | Pyodide |
|---|---|---|---|
| npz (compressed) | 3.9 MB | 0.027 | 0.044 |
| parquet | 4.1 MB | 0.010 | 0.064 |
| csv | 10.8 MB | 0.137 | 0.164 |

Loading pyarrow in Pyodide takes 1.8 s and downloads 9.5 MB, which npz does not
need.

## Caveats

- One machine. Nothing here was measured on another CPU, a phone or a different
  memory size.
- Playwright's Chromium, WebKit and Firefox, headless. Chromium and WebKit are
  the engines of Chrome and Safari; Playwright's Firefox is a patched build and
  may not run Pyodide as the Firefox people install does.
- Memory was not measured.
