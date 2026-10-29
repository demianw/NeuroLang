"""The queries of the benchmark: example-bank queries of the browser engine.

They are those of ``llm/examples/coordinate_meta_analysis.yaml`` that the
REPL in the browser can run (it leaves out the voxel maps, which need
gigabytes), from a plain selection to probabilities of every term and
bootstrap intervals. ``python queries.py OUT.json`` writes them for the
native and the browser runners.
"""
import json
import sys
from pathlib import Path

import yaml

BANK = (
    Path(__file__).resolve().parents[2]
    / "packages/neurolang_apps/neurolang_apps/utils/llm/examples"
    / "coordinate_meta_analysis.yaml"
)

# (name, start of the question): the order is that of the cost, about.
QUERIES = [
    ("select", "Which studies mention pain?"),
    ("prob-all-terms", "Which terms are mentioned by the studies, and how"),
    ("prob-subcortex", "Which subcortical regions are activated by studies about fear"),
    ("bf-term", "What is the Bayes factor of fear for the activation of the left amygdala?"),
    ("bf-all-terms-gyrus", "Decode the left middle temporal gyrus"),
    ("bf-all-terms-point", "Which terms are associated with the MNI point (40, -60, 10)"),
    ("coactive-pairs", "Which pairs of Schaefer regions are co-activated by studies about reading?"),
    ("coactive-network", "What is the co-activation network of the subcortical regions"),
    ("coactive-chance", "Are the left and right amygdala co-activated more than chance?"),
    ("ci-bf", "What is the Bayes factor of fear for the activation of the left amygdala, with its 95%"),
    ("ci-difference", "Is the left amygdala activated more often by studies about fear than"),
]


def load() -> list:
    bank = yaml.safe_load(BANK.read_text())["examples"]
    chosen = []
    for name, start in QUERIES:
        [example] = [e for e in bank if e["question"].startswith(start)]
        chosen.append({
            "name": name, "question": example["question"],
            "datalog": example["datalog"],
        })
    return chosen


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(load(), indent=1))
    print(f"{len(QUERIES)} queries in {sys.argv[1]}")
