# Probabilistic Tract-Querier Integration

This module provides probabilistic operations for tract-querier integration in NeuroLangPDL.

## Features

1. **Probabilistic Tract Queries**: Add probabilistic tract-querier queries with associated probabilities
2. **Probabilistic Tract Data**: Add tract data with probabilistic weights
3. **Integrated Solving**: Solve probabilistic tract queries using NeuroLang's probabilistic solvers
4. **Compatibility**: Works with existing tract-querier frontend and data preparation modules

## Usage

```python
from neurolang.probabilistic.probabilistic_frontend import NeurolangPDL
from neurolang.probabilistic.tract_querier.probabilistic_tract_querier import ProbabilisticTractQuerierMixin

# Create a combined class
class NeurolangPDLWithTractQuerier(NeurolangPDL, ProbabilisticTractQuerierMixin):
    def __init__(self):
        NeurolangPDL.__init__(self)
        ProbabilisticTractQuerierMixin.__init__(self)

# Initialize
nl = NeurolangPDLWithTractQuerier()

# Add probabilistic tract queries
frontal_query = nl.add_probabilistic_tract_query(
    "tracts.left.frontal_lobe", 
    0.85, 
    "frontal_tract_query"
)

# Add probabilistic tract data
tracts_data = [
    (1, "tract_a", "region_1"),
    (2, "tract_b", "region_2"),
    (3, "tract_c", "region_1")
]
probabilities = [0.9, 0.75, 0.85]

tract_facts = nl.add_tract_data_probabilistic_facts(
    tracts_data, 
    probabilities, 
    "probabilistic_tracts"
)

# Solve probabilistic queries
solutions = nl.solve_probabilistic_tract_queries()
```

## API

### ProbabilisticTractQuerierMixin

#### `add_probabilistic_tract_query(query_string, probability, name=None)`
Add a probabilistic tract-querier query with associated probability

#### `execute_probabilistic_tract_query(query_string, data_context)`
Execute a probabilistic tract-querier query and compute probabilistic solution

#### `add_tract_data_probabilistic_facts(tracts_data, probabilities, name)`
Add probabilistic tract data as facts

#### `solve_probabilistic_tract_queries()`
Solve all probabilistic tract queries in the current program