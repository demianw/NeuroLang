#!/usr/bin/env python3
"""
Simple test script to verify tract-querier integration
"""

def test_imports():
    """Test that we can import the modules"""
    print("Testing imports...")
    
    try:
        import sys
        import os
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        neurolang_dir = os.path.join(current_dir, "..")
        sys.path.insert(0, neurolang_dir)
        
        print(f"Added to path: {neurolang_dir}")
        
        # Try importing the modules
        from neurolang.frontend.tract_querier import frontend
        print("✓ Successfully imported frontend module")
        
        from neurolang.frontend.tract_querier import data_prepare
        print("✓ Successfully imported data_prepare module")
        
        from neurolang.frontend.tract_querier import query_translator
        print("✓ Successfully imported query_translator module")
        
        from neurolang.frontend.tract_querier import compatibility
        print("✓ Successfully imported compatibility module")
        
        # Try creating instances
        frontend_instance = frontend.TractQuerierFrontend()
        print("✓ Successfully created TractQuerierFrontend instance")
        
        adapter_instance = frontend.WMQLToTractQuerierAdapter()
        print("✓ Successfully created WMQLToTractQuerierAdapter instance")
        
        print("\nAll tests passed! tract-querier integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    if not success:
        exit(1)