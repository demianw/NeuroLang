"""
Data preparation for tract-querier integration
"""
import nibabel as nib
import numpy as np
import logging
import sys
import os

# Add the parent directory to sys.path to resolve relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    # Try importing NeuroLang modules
    from neurolang.regions import ExplicitVBR, PointSet
    from neurolang.solver_datalog_naive import Symbol
    NEUROLANG_IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"NeuroLang modules not available: {e}")
    NEUROLANG_IMPORTS_AVAILABLE = False

# Try to import tract-querier components
try:
    from tract_querier import tractography, tract_label_indices
    TRACT_QUERIER_AVAILABLE = True
except ImportError:
    logging.warning("tract-querier not available, some functionality will be disabled")
    TRACT_QUERIER_AVAILABLE = False

def prepare_tractquerier_data(
    atlas_filename,
    tracts_filename,
    tract_symbol_name='tracts',
    region_symbol_name='regions',
    tract_traversals_symbol_name='tract_traversals',
    endpoints_symbol_name='endpoints',
):
    """
    Given tractography data and an atlas, prepare the data structures needed for
    tract-querier integration.
    
    :param atlas_filename str: filename for the atlas image.
    :param tracts_filename str: filename for the tracts.
    :param tract_symbol_name str: symbol name for tract data
    :param region_symbol_name str: symbol name for region data
    :param tract_traversals_symbol_name str: symbol name for tract traversals
    :param endpoints_symbol_name str: symbol name for endpoints data
    """
    
    if not TRACT_QUERIER_AVAILABLE:
        raise ImportError("tract-querier is required for this functionality")
    
    # Load tractography data
    tracts_img = nib.streamlines.load(tracts_filename)
    tracts_streamlines = tracts_img.streamlines
    
    # Load atlas data
    atlas_map = nib.load(atlas_filename)
    atlas_data = atlas_map.get_fdata()
    atlas_affine = atlas_map.affine
    
    # Create tractography object
    tr = tractography.Tractography(tracts=[s for s in tracts_streamlines])
    
    # Create spatial indexing
    tli = tract_label_indices.TractographySpatialIndexing(
        tr.tracts(), atlas_data, atlas_affine, 0., 2.
    )
    
    # Prepare tract traversals data
    tract_traversals_data = []
    tracts_data = []
    eye_4 = np.eye(4)
    
    for tract_idx, labels in tli.crossing_tracts_labels.items():
        # Get streamline points
        points = tli.tractography[tract_idx]
        if len(points) == 0:
            continue
        tract_region = PointSet(points, eye_4)
        tracts_data.append((tract_idx, tract_region))
        for label in labels:
            tract_traversals_data.append((tract_idx, label))
    
    # Prepare region data
    regions_data = []
    for label in tli.crossing_labels_tracts:
        region = ExplicitVBR(
            np.transpose((atlas_data == label).nonzero()),
            atlas_affine
        )
        regions_data.append((label, region))
    
    # Prepare endpoints data
    endpoints_data = []
    for ending in tli.ending_tracts_labels:
        for tract_idx, label in ending.items():
            endpoints_data.append((tract_idx, label))
    
    return {
        'tracts': tracts_data,
        'regions': regions_data,
        'tract_traversals': tract_traversals_data,
        'endpoints': endpoints_data,
        'tract_symbol_name': tract_symbol_name,
        'region_symbol_name': region_symbol_name,
        'tract_traversals_symbol_name': tract_traversals_symbol_name,
        'endpoints_symbol_name': endpoints_symbol_name
    }