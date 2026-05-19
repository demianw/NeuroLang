"""
Main interface for tract-querier integration in NeuroLang
"""
import logging
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Add the parent directory to sys.path to resolve relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import tract-querier components
try:
    import tract_querier
    TRACT_QUERIER_AVAILABLE = True
except ImportError:
    logging.warning("tract-querier not available, some functionality will be disabled")
    TRACT_QUERIER_AVAILABLE = False

from .data_prepare import prepare_tractquerier_data
from .query_translator import TractQuerierTranslator
from .compatibility import TractQuerierCompatibilityInterface

class TractQuerierFrontend:
    """Main frontend for tract-querier integration"""
    
    def __init__(self):
        self.data_context = None
        self.translator = None
        self.compatibility_interface = TractQuerierCompatibilityInterface()
        self.is_initialized = False
        
    def initialize(self, 
                   atlas_filename: str,
                   tracts_filename: str,
                   tract_symbol_name: str = 'tracts',
                   region_symbol_name: str = 'regions',
                   tract_traversals_symbol_name: str = 'tract_traversals',
                   endpoints_symbol_name: str = 'endpoints') -> bool:
        """
        Initialize the tract-querier frontend with data
        
        :param atlas_filename: Path to atlas image file
        :param tracts_filename: Path to tractography file
        :param tract_symbol_name: Symbol name for tract data
        :param region_symbol_name: Symbol name for region data
        :param tract_traversals_symbol_name: Symbol name for tract traversals
        :param endpoints_symbol_name: Symbol name for endpoints data
        :return: True if initialization successful
        """
        if not TRACT_QUERIER_AVAILABLE:
            logging.error("tract-querier not available, cannot initialize")
            return False
            
        try:
            # Prepare data
            self.data_context = prepare_tractquerier_data(
                atlas_filename=atlas_filename,
                tracts_filename=tracts_filename,
                tract_symbol_name=tract_symbol_name,
                region_symbol_name=region_symbol_name,
                tract_traversals_symbol_name=tract_traversals_symbol_name,
                endpoints_symbol_name=endpoints_symbol_name
            )
            
            # Initialize translator
            self.translator = TractQuerierTranslator(self.data_context)
            self.is_initialized = True
            return True
        except Exception as e:
            logging.error(f"Error initializing tract-querier frontend: {e}")
            return False
            
    def execute_query(self, query_string: str) -> Any:
        """
        Execute a query using tract-querier
        
        :param query_string: Query string to execute
        :return: Query results
        """
        if not self.is_initialized:
            raise RuntimeError("Frontend not initialized. Call initialize() first.")
            
        if not TRACT_QUERIER_AVAILABLE:
            raise ImportError("tract-querier is required for query execution")
            
        if self.translator is None:
            raise RuntimeError("Translator not initialized")
            
        return self.translator.execute_query(query_string)
        
    def translate_query(self, query_string: str) -> str:
        """
        Translate a query to tract-querier format without executing
        
        :param query_string: Query string to translate
        :return: Translated query string
        """
        if not self.is_initialized:
            raise RuntimeError("Frontend not initialized. Call initialize() first.")
            
        if not TRACT_QUERIER_AVAILABLE:
            raise ImportError("tract-querier is required for query translation")
            
        if self.translator is None:
            raise RuntimeError("Translator not initialized")
            
        return self.translator.translate_query(query_string)
        
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about loaded data
        
        :return: Dictionary with data information
        """
        if not self.is_initialized or self.data_context is None:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "tract_count": len(self.data_context['tracts']),
            "region_count": len(self.data_context['regions']),
            "tract_traversal_count": len(self.data_context['tract_traversals']),
            "endpoint_count": len(self.data_context['endpoints']),
            "symbols": {
                "tracts": self.data_context['tract_symbol_name'],
                "regions": self.data_context['region_symbol_name'],
                "tract_traversals": self.data_context['tract_traversals_symbol_name'],
                "endpoints": self.data_context['endpoints_symbol_name']
            }
        }

# Backward compatibility layer
class WMQLToTractQuerierAdapter:
    """
    Adapter to maintain backward compatibility with WMQL interface
    """
    
    def __init__(self):
        self.tract_querier_frontend = TractQuerierFrontend()
        self.wmql_compatibility_mode = True
        
    def prepare_datalog_ir_program(self, 
                                   datalog,
                                   atlas_filename: str,
                                   tracts_filename: str,
                                   tracts_symbol_name: str = 'tracts',
                                   regions_symbol_name: str = 'regions',
                                   tract_traversals_symbol_name: str = 'tract_traversals',
                                   endpoints_in_symbol_name: str = 'endpoints_in'):
        """
        Compatibility function that mimics the original WMQL prepare function
        """
        # Initialize tract-querier frontend
        success = self.tract_querier_frontend.initialize(
            atlas_filename=atlas_filename,
            tracts_filename=tracts_filename,
            tract_symbol_name=tracts_symbol_name,
            region_symbol_name=regions_symbol_name,
            tract_traversals_symbol_name=tract_traversals_symbol_name,
            endpoints_symbol_name=endpoints_in_symbol_name
        )
        
        if not success:
            raise RuntimeError("Failed to initialize tract-querier frontend")
            
        # Get data context
        data_context = self.tract_querier_frontend.data_context
        
        # Add data to datalog program (backward compatibility)
        if hasattr(datalog, 'add_extensional_predicate_from_tuples') and data_context is not None:
            # Add tracts data
            datalog.add_extensional_predicate_from_tuples(
                data_context['tract_symbol_name'],
                data_context['tracts']
            )
            
            # Add regions data
            datalog.add_extensional_predicate_from_tuples(
                data_context['region_symbol_name'],
                data_context['regions']
            )
            
            # Add tract traversals data
            datalog.add_extensional_predicate_from_tuples(
                data_context['tract_traversals_symbol_name'],
                data_context['tract_traversals']
            )
            
            # Add endpoints data
            datalog.add_extensional_predicate_from_tuples(
                data_context['endpoints_symbol_name'],
                data_context['endpoints']
            )
            
        return datalog

# Global instance for easy access
tract_querier_frontend = TractQuerierFrontend()
wmql_adapter = WMQLToTractQuerierAdapter()