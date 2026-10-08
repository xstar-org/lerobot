"""Compatibility import for the shared vlabench native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import vlabench as _native

if TYPE_CHECKING:
    from lerobot.sims.native.vlabench import *  # noqa: F403

sys.modules[__name__] = _native
