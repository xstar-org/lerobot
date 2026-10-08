"""Compatibility import for the shared metaworld native wrapper."""

import sys
from typing import TYPE_CHECKING

from lerobot.sims.native import metaworld as _native

if TYPE_CHECKING:
    from lerobot.sims.native.metaworld import *  # noqa: F403

sys.modules[__name__] = _native
