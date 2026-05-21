import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.interval_algebra instead of neurolang.interval_algebra",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import interval_algebra as _interval_algebra
sys.modules[__name__] = _interval_algebra
