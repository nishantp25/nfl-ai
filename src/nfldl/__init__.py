"""nfldl -- a manifest-driven, leakage-aware NFL raw data layer.

Design rules, in priority order:

1. Nothing is downloaded that is not declared in ``configs/data_sources.yaml``.
2. Coverage is verified against the provider, never assumed from a constant.
3. Derived features are computed here, never downloaded from a third party.
4. Every stored row can answer "where did this come from, and when did we
   learn it".
"""

from .config import get_settings
from .manifest import get_manifest

__version__ = "1.0.0"
__all__ = ["get_settings", "get_manifest"]
