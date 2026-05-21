import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.perior_tree instead of neurolang.perior_tree",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import perior_tree as _perior_tree
sys.modules[__name__] = _perior_tree
