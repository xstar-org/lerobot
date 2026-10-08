"""Simulator addressing over the shared version-one envelope.

Operation names live in query keys / CONTROL bodies. No inference message tags change.
"""

from lerobot.transport.wire.protocol import validate_segment


def deployment_prefix(deployment: str) -> str:
    """Return a validated simulator deployment key."""
    return f"lerobot/sim/v1/sims/{validate_segment(deployment)}"


def instance_prefix(deployment: str, instance: str) -> str:
    """Return a simulator boot key under its deployment."""
    return f"{deployment_prefix(deployment)}/instances/{validate_segment(instance)}"


def session_prefix(deployment: str, instance: str, session: str) -> str:
    """Return an admitted world-session key."""
    return f"{instance_prefix(deployment, instance)}/sessions/{validate_segment(session)}"
