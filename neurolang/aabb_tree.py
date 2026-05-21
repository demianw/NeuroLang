import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.aabb_tree instead of neurolang.aabb_tree",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import aabb_tree as _aabb_tree
sys.modules[__name__] = _aabb_tree
