"""Compatibility import for the shared robotwin native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import robotwin as _native

if TYPE_CHECKING:
    from lerobot.sims.native.robotwin import *  # noqa: F403

sys.modules[__name__] = _native
