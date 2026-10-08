"""Compatibility import for the shared libero native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import libero as _native

if TYPE_CHECKING:
    from lerobot.sims.native.libero import *  # noqa: F403

sys.modules[__name__] = _native
