"""Who imports what, and how long it takes: the chains to the heavy libraries.

    python import_chains.py MODULE [MODULE...]

Runs ``python -X importtime`` importing the modules, and prints the most
expensive imports with the chain of modules that led to each of ``nilearn``,
``sklearn``, ``scipy``, ``matplotlib``, ``rdflib``, ``lark`` and ``nibabel``.
"""
import re
import subprocess
import sys

HEAVY = ("nilearn", "sklearn", "scipy", "matplotlib", "rdflib", "lark", "nibabel")
log = subprocess.run(
    [sys.executable, "-X", "importtime", "-c", "import " + ", ".join(sys.argv[1:])],
    capture_output=True, text=True,
).stderr
rows = []
for line in log.splitlines():
    match = re.match(r"import time:\s+(\d+) \|\s+(\d+) \| (\s*)(\S+)", line)
    if match:
        self_us, cumulative_us, indent, name = match.groups()
        rows.append((len(indent) // 2, name, int(cumulative_us) / 1e6))
# The log lists a module after the modules it imports, indented one more.
parents = {}
stack = []  # (depth, index) of the modules waiting for their parent
for index, (depth, name, _) in enumerate(rows):
    while stack and stack[-1][0] > depth:
        parents[stack.pop()[1]] = index
    stack.append((depth, index))
firsts = {}
for index, (depth, name, seconds) in enumerate(rows):
    top = name.split(".")[0]
    if top in HEAVY and depth > 0 and top not in firsts:
        firsts[top] = index
print(f"{'cumulative s':>12}  top-level imports")
for depth, name, seconds in sorted((r for r in rows if r[0] == 1 and r[2] > 0.1), key=lambda r: -r[2])[:12]:
    print(f"{seconds:12.2f}  {name}")
print()
for top, index in sorted(firsts.items(), key=lambda t: -rows[t[1]][2]):
    chain, i = [], index
    while i is not None:
        chain.append(rows[i][1])
        i = parents.get(i)
    print(f"{top} ({rows[index][2]:.2f} s): " + " <- ".join(chain[:6]))
