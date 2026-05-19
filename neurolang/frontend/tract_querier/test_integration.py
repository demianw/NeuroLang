"""
Test script for tract-querier integration
"""
import os
import sys

# Add the NeuroLang directory to the path
neurolang_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, neurolang_path)

def test_tract_querier_import():
    """Test that we can import the tract-querier modules"""
    try:
        # Test importing individual modules
        from neurolang.frontend.tract_querier import frontend
        from neurolang.frontend.tract_querier import data_prepare
        from neurolang.frontend.tract_querier import query_translator
        from neurolang.frontend.tract_querier import compatibility
        print("SUCCESS: All tract-querier modules imported successfully")
        return True
    except ImportError as e:
        print(f"FAILED: Could not import tract-querier modules: {e}")
        return False

def test_frontend_creation():
    """Test that we can create frontend instances"""
    try:
        from neurolang.frontend.tract_querier.frontend import TractQuerierFrontend, WMQLToTractQuerierAdapter
        frontend = TractQuerierFrontend()
        adapter = WMQLToTractQuerierAdapter()
        print("SUCCESS: Frontend instances created successfully")
        return True
    except Exception as e:
        print(f"FAILED: Could not create frontend instances: {e}")
        return False

def test_data_prepare_import():
    """Test that we can import the data preparation module"""
    try:
        from neurolang.frontend.tract_querier.data_prepare import prepare_tractquerier_data
        print("SUCCESS: Data preparation module imported successfully")
        return True
    except ImportError as e:
        print(f"FAILED: Could not import data preparation module: {e}")
        return False

def test_query_translator_import():
    """Test that we can import the query translator module"""
    try:
        from neurolang.frontend.tract_querier.query_translator import TractQuerierTranslator
        print("SUCCESS: Query translator module imported successfully")
        return True
    except ImportError as e:
        print(f"FAILED: Could not import query translator module: {e}")
        return False

def test_compatibility_import():
    """Test that we can import the compatibility module"""
    try:
        from neurolang.frontend.tract_querier.compatibility import TractQuerierCompatibilityInterface
        print("SUCCESS: Compatibility module imported successfully")
        return True
    except ImportError as e:
        print(f"FAILED: Could not import compatibility module: {e}")
        return False

if __name__ == "__main__":
    print("Testing tract-querier integration...")
    
    tests = [
        test_tract_querier_import,
        test_frontend_creation,
        test_data_prepare_import,
        test_query_translator_import,
        test_compatibility_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! tract-querier integration is ready.")
    else:
        print("Some tests failed. Please check the implementation.")