# tract-querier Integration Documentation

## Overview

This documentation describes the integration of tract-querier into NeuroLang, replacing the existing WMQL implementation with a more robust and efficient system for white matter query processing.

## Architecture

The tract-querier integration consists of several key components:

1. **Data Preparation** (`data_prepare.py`): Loads and prepares tractography and atlas data for tract-querier processing
2. **Query Translation** (`query_translator.py`): Translates WMQL-style queries to tract-querier format
3. **Compatibility Layer** (`compatibility.py`): Maintains backward compatibility with existing WMQL interface
4. **Frontend Interface** (`frontend.py`): Main interface for tract-querier integration

## Installation

To use the tract-querier integration, you need to install the tract-querier package:

```bash
pip install tract-querier>=0.3.0
```

The dependency has already been added to `setup.cfg`.

## Usage

### Basic Usage

```python
from neurolang.frontend.tract_querier.frontend import tract_querier_frontend

# Initialize the frontend
success = tract_querier_frontend.initialize(
    atlas_filename="path/to/atlas.nii.gz",
    tracts_filename="path/to/tracts.trk"
)

if success:
    # Translate a query
    translated_query = tract_querier_frontend.translate_query("anterior_of(region1)")
    
    # Execute a query
    results = tract_querier_frontend.execute_query("anterior_of(region1)")
```

### Backward Compatibility

For existing WMQL users, the adapter maintains compatibility:

```python
from neurolang.frontend.tract_querier.frontend import wmql_adapter

# This mimics the original WMQL prepare function
datalog_program = wmql_adapter.prepare_datalog_ir_program(
    datalog=datalog_instance,
    atlas_filename="path/to/atlas.nii.gz",
    tracts_filename="path/to/tracts.trk"
)
```

## Migration from WMQL

### Key Changes

1. **Query Translation**: WMQL queries are automatically translated to tract-querier format
2. **Data Processing**: Uses tract-querier's optimized algorithms for spatial operations
3. **Performance**: Improved performance through tract-querier's optimized implementation

### Breaking Changes

1. **Assignment Semantics**: Subtle differences in `|=` behavior may occur
2. **Spatial Precision**: Different algorithms may produce slightly different results
3. **Performance Characteristics**: tract-querier performance profile differs from WMQL
4. **Error Messages**: New error reporting formats

### Backward Compatibility Measures

1. **Compatibility Mode**: Option to use legacy WMQL for critical workflows
2. **Migration Tools**: Automated script conversion utilities
3. **Detailed Documentation**: Comprehensive migration guide
4. **Deprecation Warnings**: Clear notifications of legacy component usage

## API Reference

### TractQuerierFrontend

Main class for tract-querier integration.

#### Methods

- `initialize()`: Initialize the frontend with data files
- `execute_query()`: Execute a query using tract-querier
- `translate_query()`: Translate a query to tract-querier format
- `get_data_info()`: Get information about loaded data

### WMQLToTractQuerierAdapter

Adapter class for backward compatibility.

#### Methods

- `prepare_datalog_ir_program()`: Prepare datalog program (mimics original WMQL function)

## Testing

Run the test script to verify the integration:

```bash
python neurolang/frontend/tract_querier/test_integration.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure tract-querier is installed
2. **Data Loading Issues**: Verify file paths and formats
3. **Query Translation Errors**: Check query syntax against tract-querier documentation

### Support

For issues with the tract-querier integration, please:
1. Check the tract-querier documentation
2. Verify your data files are in the correct format
3. Contact the development team with detailed error information