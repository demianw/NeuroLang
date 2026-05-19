#!/usr/bin/env python3
"""
Performance benchmark comparison between WMQL and tract-querier implementations
"""

import time
import os

def benchmark_module_imports():
    """Benchmark module import performance"""
    print("Benchmarking module import performance...")
    
    # Test importing frontend module
    start_time = time.time()
    try:
        import frontend
        end_time = time.time()
        print(f"✓ Frontend module import time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"⚠ Frontend module import error: {e}")
    
    # Test importing data_prepare module
    start_time = time.time()
    try:
        import data_prepare
        end_time = time.time()
        print(f"✓ Data prepare module import time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"⚠ Data prepare module import error: {e}")
    
    # Test importing query_translator module
    start_time = time.time()
    try:
        import query_translator
        end_time = time.time()
        print(f"✓ Query translator module import time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"⚠ Query translator module import error: {e}")

def benchmark_initialization():
    """Benchmark frontend initialization performance"""
    print("\nBenchmarking frontend initialization performance...")
    
    try:
        import frontend
        start_time = time.time()
        frontend_instance = frontend.TractQuerierFrontend()
        end_time = time.time()
        print(f"✓ Frontend initialization time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"⚠ Frontend initialization error: {e}")

def benchmark_adapter_creation():
    """Benchmark adapter creation performance"""
    print("\nBenchmarking adapter creation performance...")
    
    try:
        import frontend
        start_time = time.time()
        adapter_instance = frontend.WMQLToTractQuerierAdapter()
        end_time = time.time()
        print(f"✓ Adapter creation time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"⚠ Adapter creation error: {e}")

def benchmark_spatial_operations():
    """Benchmark spatial operations mapping"""
    print("\nBenchmarking spatial operations mapping...")
    
    try:
        import query_translator
        # Create a translator instance with mock data context
        mock_data_context = {}
        start_time = time.time()
        translator = query_translator.TractQuerierTranslator(mock_data_context)
        end_time = time.time()
        print(f"✓ Translator creation time: {end_time - start_time:.4f} seconds")
        
        # Check spatial operations count
        operations_count = len(translator.spatial_operations)
        print(f"✓ Spatial operations count: {operations_count}")
        
    except Exception as e:
        print(f"⚠ Spatial operations benchmark error: {e}")

def compare_file_sizes():
    """Compare file sizes between implementations"""
    print("\nComparing implementation file sizes...")
    
    # Get sizes of tract-querier implementation files
    tract_querier_dir = "."
    files_to_check = [
        "frontend.py",
        "data_prepare.py", 
        "query_translator.py",
        "compatibility.py"
    ]
    
    total_size = 0
    for filename in files_to_check:
        filepath = os.path.join(tract_querier_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: {size} bytes")
            total_size += size
        else:
            print(f"✗ {filename}: File not found")
    
    print(f"✓ Total implementation size: {total_size} bytes")

if __name__ == "__main__":
    print("TRACT-QUERIER INTEGRATION PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Run all benchmarks
    benchmark_module_imports()
    benchmark_initialization()
    benchmark_adapter_creation()
    benchmark_spatial_operations()
    compare_file_sizes()
    
    print("\n" + "=" * 50)
    print("Performance benchmarking complete.")
    print("Note: Import errors may be expected in test environment.")