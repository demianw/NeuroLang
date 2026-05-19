"""
Example usage of probabilistic tract-querier operations
"""
from neurolang.probabilistic.probabilistic_frontend import NeurolangPDL
from neurolang.probabilistic.tract_querier.probabilistic_tract_querier import ProbabilisticTractQuerierMixin


# Create a combined class with both probabilistic and tract-querier capabilities
class NeurolangPDLWithTractQuerier(NeurolangPDL, ProbabilisticTractQuerierMixin):
    def __init__(self):
        NeurolangPDL.__init__(self)
        ProbabilisticTractQuerierMixin.__init__(self)


# Example usage
if __name__ == "__main__":
    # Initialize the combined frontend
    nl = NeurolangPDLWithTractQuerier()
    
    # Initialize tract-querier with data (in practice, you'd provide real file paths)
    # For demonstration, we'll just show how it would work
    try:
        success = nl.tract_querier_frontend.initialize(
            atlas_filename="path/to/atlas.nii.gz",
            tracts_filename="path/to/tracts.trk"
        )
        if success:
            print("Tract-querier initialized successfully")
        else:
            print("Failed to initialize tract-querier")
    except Exception as e:
        print(f"Error initializing tract-querier: {e}")
    
    # Add probabilistic tract queries
    frontal_query = nl.add_probabilistic_tract_query(
        "tracts.left.frontal_lobe", 
        0.85, 
        "frontal_tract_query"
    )
    
    parietal_query = nl.add_probabilistic_tract_query(
        "tracts.left.parietal_lobe", 
        0.75, 
        "parietal_tract_query"
    )
    
    print(f"Added probabilistic queries: {frontal_query.name}, {parietal_query.name}")
    
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
    
    print(f"Added probabilistic tract facts: {tract_facts.name}")
    
    # Execute a query (in practice, this would use real tract-querier)
    if nl.tract_querier_frontend.is_initialized:
        data_context = {
            'tracts': [],
            'regions': [],
            'tract_traversals': [],
            'endpoints': []
        }
        
        result = nl.execute_probabilistic_tract_query(
            "tracts.left.frontal_lobe",
            data_context
        )
        print(f"Query execution result: {result}")
    
    # Solve probabilistic queries
    try:
        solutions = nl.solve_probabilistic_tract_queries()
        print(f"Found {len(solutions)} probabilistic tract query solutions")
    except Exception as e:
        print(f"Error solving probabilistic tract queries: {e}")