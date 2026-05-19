# Migration Guide: WMQL to tract-querier Integration

## Overview

This guide explains how to migrate from the existing WMQL implementation to the new tract-querier integration in NeuroLang. The tract-querier implementation provides better performance, more accurate spatial operations, and improved compatibility with modern tractography analysis workflows.

## Key Components Replaced

### 1. Core Translation Layer
**Previous:** `translator.py` with Datalog-based translation  
**Replacement:** tract-querier query engine with direct spatial operations  
**Files Affected:** neurolang/frontend/wmql/translator.py

### 2. Data Preparation Layer
**Previous:** `prepare.py` initializing Datalog predicates  
**Replacement:** tract-querier data interface and initialization  
**Files Affected:** neurolang/frontend/wmql/prepare.py

### 3. Grammar Definition
**Previous:** `wmql.ebnf` for TatSu parser  
**Replacement:** tract-querier query language with compatibility parser  
**Files Affected:** neurolang/frontend/wmql/wmql.ebnf

## Migration Steps

### Phase 1: Environment Setup
1. Install tract-querier dependency:
   ```bash
   pip install tract-querier>=0.3.0
   ```

2. Update setup.cfg to include tract-querier as a dependency:
   ```ini
   install_requires =
     # ... existing dependencies ...
     tract-querier>=0.3.0
   ```

### Phase 2: Code Migration

#### Data Preparation Migration
**Previous WMQL approach:**
```python
# In prepare.py
from ...solver_datalog_naive import Symbol

def prepare_datalog_ir_program(
    datalog, atlas_filename, tracts_filename,
    tracts_symbol_name='tracts',
    regions_symbol_name='regions',
    tract_traversals_symbol_name='tract_traversals',
    endpoints_in_symbol_name='endpoints_in',
):
    # WMQL-specific data preparation
```

**New tract-querier approach:**
```python
# In tract_querier/data_prepare.py
from neurolang.frontend.tract_querier.data_prepare import prepare_tractquerier_data

def prepare_datalog_ir_program(
    datalog, atlas_filename, tracts_filename,
    tracts_symbol_name='tracts',
    regions_symbol_name='regions',
    tract_traversals_symbol_name='tract_traversals',
    endpoints_in_symbol_name='endpoints_in',
):
    # Use tract-querier compatibility adapter
    from neurolang.frontend.tract_querier.frontend import wmql_adapter
    return wmql_adapter.prepare_datalog_ir_program(
        datalog, atlas_filename, tracts_filename,
        tracts_symbol_name, regions_symbol_name,
        tract_traversals_symbol_name, endpoints_in_symbol_name
    )
```

#### Query Translation Migration
**Previous WMQL approach:**
```python
# Using TatSu parser and custom translation
from tatsu.model import NodeWalker, ModelBuilderSemantics

class WMQLDatalogSemantics(NodeWalker):
    def walk_function_evaluation(self, fe, **kwargs):
        # Custom translation logic for each function
```

**New tract-querier approach:**
```python
# Using tract-querier's built-in query engine
from neurolang.frontend.tract_querier.frontend import tract_querier_frontend

# Initialize once with data
tract_querier_frontend.initialize(
    atlas_filename="path/to/atlas.nii.gz",
    tracts_filename="path/to/tracts.trk"
)

# Translate queries
translated_query = tract_querier_frontend.translate_query("anterior_of(region1)")

# Execute queries
results = tract_querier_frontend.execute_query("anterior_of(region1)")
```

### Phase 3: API Compatibility

#### Backward Compatible Interface
The new implementation includes a compatibility adapter that mimics the original WMQL interface:

```python
# For existing code that uses the old prepare function
from neurolang.frontend.tract_querier.frontend import wmql_adapter

# This works exactly like the original
datalog_program = wmql_adapter.prepare_datalog_ir_program(
    datalog=datalog_instance,
    atlas_filename="path/to/atlas.nii.gz",
    tracts_filename="path/to/tracts.trk"
)
```

#### New Enhanced Interface
For new code, use the direct tract-querier interface:

```python
from neurolang.frontend.tract_querier.frontend import tract_querier_frontend

# Initialize with data
tract_querier_frontend.initialize(
    atlas_filename="path/to/atlas.nii.gz",
    tracts_filename="path/to/tracts.trk"
)

# Execute queries directly
results = tract_querier_frontend.execute_query("anterior_of(region1)")
```

## Breaking Changes and Considerations

### 1. Query Translation Approach
- **Before:** WMQL queries translated to Datalog expressions
- **After:** Direct execution through tract-querier engine
- **Impact:** Subtle semantic differences possible in complex queries

### 2. Data Structure Representation
- **Before:** ExplicitVBR and PointSet representations
- **After:** tract-querier native data structures
- **Impact:** Potential precision differences in region boundaries

### 3. Spatial Operation Implementation
- **Before:** Custom implementations in WMQL translator
- **After:** tract-querier optimized spatial algorithms
- **Impact:** Performance improvements but potentially different results

### 4. Dependency Stack
- **Before:** Depends on TatSu parser generator
- **After:** Direct integration with tract-querier
- **Impact:** Different error messages and debugging experience

## Backward Compatibility Measures

### 1. Syntax Compatibility
- WMQL syntax is maintained through translation layer
- Existing query scripts should work without modification
- New tract-querier features available as extensions

### 2. API Compatibility
- Existing NeuroLang API remains unchanged
- Result formats converted to maintain compatibility
- Deprecation warnings for legacy functions

### 3. File Format Support
- Existing trackvis/NIfTI support maintained
- Additional tract-querier native formats supported
- Automatic format conversion where necessary

## Testing Migration

### Functional Equivalence
All existing WMQL queries must produce equivalent results:
- Spatial operations must match within acceptable tolerances
- Performance must meet or exceed current capabilities

### Regression Testing
Existing script libraries must be validated:
- Output formats must remain compatible with downstream tools
- Error conditions must be properly handled and reported

### Performance Validation
Query execution times must be benchmarked:
- Memory usage patterns must be analyzed
- Scalability with large tractography datasets must be verified

## Rollback Plan

### Emergency Rollback
If issues are encountered during migration:

1. Revert to the original WMQL implementation branch
2. Use the compatibility adapter temporarily while debugging
3. Report issues to the development team with detailed error information

### Parallel Testing
During transition period:
- Run both WMQL and tract-querier versions in parallel
- Validate results match within acceptable tolerances
- Report any discrepancies to development team

## Conclusion

The tract-querier integration provides significant improvements over the previous WMQL implementation while maintaining backward compatibility. The migration process is designed to be smooth with minimal disruption to existing workflows.