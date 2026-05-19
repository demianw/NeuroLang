"""
Integration tests for probabilistic tract-querier operations
"""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from neurolang.probabilistic.tract_querier.probabilistic_tract_querier import (
    ProbabilisticTractQuerierMixin
)
from neurolang.probabilistic.probabilistic_frontend import NeurolangPDL


class MockTractQuerierFrontend:
    """Mock tract-querier frontend for testing"""
    
    def __init__(self):
        self.is_initialized = True
        
    def execute_query(self, query_string):
        return {
            "query": query_string,
            "result": "mock_result",
            "count": 5
        }


class TestProbabilisticTractQuerierMixin(unittest.TestCase):
    """Test probabilistic tract-querier operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a mock version of NeurolangPDL with our mixin
        class TestNeurolangPDL(NeurolangPDL, ProbabilisticTractQuerierMixin):
            def __init__(self):
                NeurolangPDL.__init__(self)
                ProbabilisticTractQuerierMixin.__init__(self)
                
        self.nl = TestNeurolangPDL()
        
        # Mock tract-querier frontend
        self.nl.tract_querier_frontend = MockTractQuerierFrontend()
        
    def test_add_probabilistic_tract_query(self):
        """Test adding probabilistic tract queries"""
        # Test valid probability
        query_symbol = self.nl.add_probabilistic_tract_query(
            "tracts.left.frontal_lobe", 0.85, "frontal_tract_query"
        )
        
        self.assertIsNotNone(query_symbol)
        self.assertEqual(query_symbol.name, "frontal_tract_query")
        
        # Test invalid probability
        with self.assertRaises(ValueError):
            self.nl.add_probabilistic_tract_query(
                "tracts.left.frontal_lobe", 1.5, "invalid_query"
            )
            
    def test_add_tract_data_probabilistic_facts(self):
        """Test adding probabilistic tract data"""
        # Mock tract data
        tracts_data = [
            (1, "tract_a", "region_1"),
            (2, "tract_b", "region_2"),
            (3, "tract_c", "region_1")
        ]
        probabilities = [0.9, 0.75, 0.85]
        
        # Add probabilistic facts
        fact_symbol = self.nl.add_tract_data_probabilistic_facts(
            tracts_data, probabilities, "probabilistic_tracts"
        )
        
        self.assertIsNotNone(fact_symbol)
        self.assertEqual(fact_symbol.name, "probabilistic_tracts")
        
        # Test mismatched lengths
        with self.assertRaises(ValueError):
            self.nl.add_tract_data_probabilistic_facts(
                tracts_data, [0.5, 0.6], "mismatched_facts"
            )
            
    @patch('neurolang.probabilistic.tract_querier.probabilistic_tract_querier.compute_probabilistic_solution')
    def test_execute_probabilistic_tract_query(self, mock_compute):
        """Test executing probabilistic tract queries"""
        # Mock the compute function to return a simple solution
        mock_compute.return_value = {
            'query_symbol': Mock()
        }
        
        # Prepare mock data context
        data_context = {
            'tracts': [],
            'regions': [],
            'tract_traversals': [],
            'endpoints': []
        }
        
        # Execute query
        result = self.nl.execute_probabilistic_tract_query(
            "tracts.left.frontal_lobe", 
            data_context
        )
        
        # Check result structure
        self.assertIn("tract_query_result", result)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        
    def test_solve_probabilistic_tract_queries(self):
        """Test solving probabilistic tract queries"""
        # Add a probabilistic query
        self.nl.add_probabilistic_tract_query(
            "tracts.left.frontal_lobe", 0.85, "test_tract_query"
        )
        
        # Solve queries (this would normally compute actual results)
        with patch.object(self.nl, 'solve_all', return_value={
            'test_tract_query': Mock(),
            'other_predicate': Mock()
        }):
            results = self.nl.solve_probabilistic_tract_queries()
            
        # Check that tract-related results are returned
        self.assertIn('test_tract_query', results)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()