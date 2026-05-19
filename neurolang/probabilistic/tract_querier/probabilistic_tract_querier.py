"""
Probabilistic tract-querier integration for NeuroLangPDL
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from ..probabilistic_frontend import NeurolangPDL
from ..query_resolution import compute_probabilistic_solution
from ..dichotomy_theorem_based_solver import solve_marg_query as lifted_solve_marg_query
from ..weighted_model_counting import solve_marg_query as wmc_solve_marg_query
from ...frontend.tract_querier.frontend import TractQuerierFrontend, WMQLToTractQuerierAdapter


class ProbabilisticTractQuerierMixin:
    """
    Mixin to extend NeurolangPDL with probabilistic tract-querier operations
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tract_querier_frontend = TractQuerierFrontend()
        self.wmql_adapter = WMQLToTractQuerierAdapter()
        
    def add_probabilistic_tract_query(
        self,
        query_string: str,
        probability: float,
        name: Optional[str] = None
    ):
        """
        Add a probabilistic tract-querier query with associated probability
        
        Parameters
        ----------
        query_string : str
            Tract-querier query string
        probability : float
            Probability associated with this query (between 0 and 1)
        name : Optional[str]
            Optional name for this probabilistic query
            
        Returns
        -------
        Symbol representing the probabilistic query
        """
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1")
            
        # Create a symbol for this query
        if name is None:
            from uuid import uuid1
            name = f"tract_query_{str(uuid1()).replace('-', '_')}"
            
        # Add as a probabilistic fact
        return self.add_probabilistic_facts_from_tuples(
            [(probability, query_string)], 
            name=name
        )
        
    def execute_probabilistic_tract_query(
        self,
        query_string: str,
        data_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a probabilistic tract-querier query
        
        Parameters
        ----------
        query_string : str
            Tract-querier query string
        data_context : Dict[str, Any]
            Data context for tract-querier execution
            
        Returns
        -------
        Dict containing query results and probabilities
        """
        # Initialize tract-querier frontend
        if not self.tract_querier_frontend.is_initialized:
            raise RuntimeError("Tract-querier frontend not initialized")
            
        # Execute the query through tract-querier
        result = self.tract_querier_frontend.execute_query(query_string)
        
        # Compute probabilistic solution if applicable
        try:
            # Get current program and solve it
            solution = self.solve_all()
            return {
                "tract_query_result": result,
                "probabilistic_solution": solution,
                "status": "success"
            }
        except Exception as e:
            logging.error(f"Error computing probabilistic solution: {e}")
            return {
                "tract_query_result": result,
                "probabilistic_solution": None,
                "status": "partial_success",
                "error": str(e)
            }
            
    def add_tract_data_probabilistic_facts(
        self,
        tracts_data,
        probabilities: List[float],
        name: str = "probabilistic_tracts"
    ):
        """
        Add probabilistic tract data as facts
        
        Parameters
        ----------
        tracts_data : List
            List of tract data tuples
        probabilities : List[float]
            List of probabilities for each tract (must match length of tracts_data)
        name : str
            Name for the probabilistic predicate
            
        Returns
        -------
        Symbol representing the probabilistic facts
        """
        if len(tracts_data) != len(probabilities):
            raise ValueError("Length of tracts_data must match length of probabilities")
            
        # Create probabilistic tuples
        probabilistic_tuples = [
            (prob, *tract_data) 
            for prob, tract_data in zip(probabilities, tracts_data)
        ]
        
        return self.add_probabilistic_facts_from_tuples(
            probabilistic_tuples,
            name=name
        )
        
    def solve_probabilistic_tract_queries(self):
        """
        Solve all probabilistic tract queries in the current program
        
        Returns
        -------
        Dict with solutions for all probabilistic tract queries
        """
        # Get current program
        program = self.current_program
        
        # Solve using existing probabilistic solvers
        solution = self.solve_all()
        
        # Extract tract-specific results
        tract_results = {}
        for symbol_name, result in solution.items():
            if "tract" in symbol_name.lower() or "query" in symbol_name.lower():
                tract_results[symbol_name] = result
                
        return tract_results