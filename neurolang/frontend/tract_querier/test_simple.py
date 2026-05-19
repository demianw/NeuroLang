#!/usr/bin/env python3
"""
Simple test script to verify tract-querier integration
"""

def test_imports():
    """Test that we can import the modules"""
    print("Testing imports...")
    
    try:
        # Try importing the modules directly
        import frontend
        print("✓ Successfully imported frontend module")
        
        import data_prepare
        print("✓ Successfully imported data_prepare module")
        
        import query_translator
        print("✓ Successfully imported query_translator module")
        
        import compatibility
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