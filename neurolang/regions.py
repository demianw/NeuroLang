import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.regions instead of neurolang.regions",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import regions as _regions
sys.modules[__name__] = _regions
