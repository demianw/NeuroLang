import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.region_solver instead of neurolang.region_solver",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import region_solver as _region_solver
sys.modules[__name__] = _region_solver
