import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.neurosynth_utils instead of neurolang.frontend.neurosynth_utils",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import neurosynth_utils as _neurosynth_utils
sys.modules[__name__] = _neurosynth_utils
