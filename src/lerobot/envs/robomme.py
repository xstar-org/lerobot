"""Compatibility import for the shared robomme native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import robomme as _native

if TYPE_CHECKING:
    from lerobot.sims.native.robomme import *  # noqa: F403

sys.modules[__name__] = _native
