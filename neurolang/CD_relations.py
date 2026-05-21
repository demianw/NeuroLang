import warnings
warnings.warn(
    "Import from neurolang.neuroimaging.CD_relations instead of neurolang.CD_relations",
    DeprecationWarning, stacklevel=2
)

import sys
from neurolang.neuroimaging import CD_relations as _CD_relations
sys.modules[__name__] = _CD_relations
