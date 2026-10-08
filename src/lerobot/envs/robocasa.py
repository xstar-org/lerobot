"""Compatibility import for the shared robocasa native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import robocasa as _native

if TYPE_CHECKING:
    from lerobot.sims.native.robocasa import *  # noqa: F403

sys.modules[__name__] = _native
