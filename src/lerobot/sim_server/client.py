"""Explicit, non-retrying simulator RPC client."""

import uuid
from threading import RLock

from lerobot.transport.wire.codec import decode_message, encode_message
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import PresenceToken, ZenohConfig, ZenohTransport

from .contracts import SimDescriptor, StepResult, validate_result
from .protocol import deployment_prefix, instance_prefix, session_prefix


class SimClient:
    """Own one explicitly negotiated simulator session without replaying mutations."""

    def __init__(self, endpoint: str, deployment: str = "default", timeout_s: float = 120):
        """Initialize owned resources without advancing simulation time."""
        self.transport = ZenohTransport(ZenohConfig(connect_endpoints=[endpoint]))
        self.deployment = deployment
        self.timeout_s = timeout_s
        self.instance = ""
        self.session = ""
        self.generation = 0
        self.token: PresenceToken | None = None
        self.descriptor: SimDescriptor | None = None
        self.lock = RLock()
        self.failed = False
        self.num_envs = 0

    def describe(self) -> SimDescriptor:
        """Discover the simulator boot and canonical schema without opening a world session."""
        with self.lock:
            self.transport.open()
            body = self._query(deployment_prefix(self.deployment) + "/describe", MessageType.DESCRIBE, {})
            self.instance = body["instance"]
            self.descriptor = SimDescriptor.from_dict(body["descriptor"])
            return self.descriptor

    @property
    def prefix(self) -> str:
        """Return the address of the currently owned world session."""
        if not self.session:
            raise RuntimeError("Simulator session is not open")
        return session_prefix(self.deployment, self.instance, self.session)

    def open(
        self,
        num_envs: int,
        clock: str,
        task_group: str | None = None,
        task_id: int = 0,
        seeds: list[int | None] | None = None,
    ) -> StepResult:
        """Admit an exclusive environment batch with the requested clock."""
        with self.lock:
            if self.session:
                raise RuntimeError("Simulator session already open")
            if self.descriptor is None:
                self.describe()
            body = self._query(
                instance_prefix(self.deployment, self.instance) + "/open",
                MessageType.OPEN,
                {
                    "num_envs": num_envs,
                    "clock": clock,
                    "task_group": task_group,
                    "task_id": task_id,
                    "seeds": seeds,
                },
            )
            assert self.descriptor is not None
            opened_descriptor = SimDescriptor.from_dict(body["descriptor"])
            if (
                opened_descriptor.features != self.descriptor.features
                or opened_descriptor.action_feature != self.descriptor.action_feature
                or opened_descriptor.semantics != self.descriptor.semantics
            ):
                self.transport.close()
                raise ProtocolError(
                    ErrorCode.INCOMPATIBLE, "Selected task differs from the negotiated simulator schema"
                )
            self.descriptor = opened_descriptor
            self.session = body["session"]
            self.generation = body["generation"]
            self.token = self.transport.declare_token(self.prefix + "/client_alive")
            self.num_envs = num_envs
            result = StepResult.from_dict(body["result"])
            validate_result(result, self.descriptor, num_envs)
            return result

    def request(self, operation: str, **body) -> dict:
        """Issue one correlated world operation and fail on an ambiguous result."""
        with self.lock:
            if self.failed:
                raise RuntimeError("Simulator session failed; close and reopen before continuing")
            key = self.prefix + (f"/{operation}" if operation in {"reset", "step"} else "/control")
            if operation not in {"reset", "step"}:
                body["operation"] = operation
            try:
                result = self._query(key, MessageType.CONTROL, body)
                if "result" in result:
                    assert self.descriptor is not None
                    validate_result(StepResult.from_dict(result["result"]), self.descriptor, self.num_envs)
                if "generation" in result:
                    self.generation = result["generation"]
                return result
            except Exception:
                # A timed-out mutation may have executed. Never continue on an ambiguous world.
                self.failed = True
                raise

    def _query(self, key, kind, body):
        message = Envelope(kind, self.instance, self.session, self.generation, uuid.uuid4().hex, body)
        replies = self.transport.query(key, encode_message(message), self.timeout_s, max_replies=1)
        if len(replies) != 1:
            raise TimeoutError(f"No simulator reply: {key}")
        reply = decode_message(replies[0])
        if (
            reply.request_id != message.request_id
            or reply.instance_id != message.instance_id
            or reply.session_id != message.session_id
            or reply.generation != message.generation
        ):
            raise ProtocolError(ErrorCode.STALE, "Simulator reply identity does not match request")
        if reply.message_type == MessageType.ERROR:
            raise ProtocolError(ErrorCode(reply.body["code"]), reply.body["message"])
        return reply.body

    def close(self):
        """Release owned simulator resources and transport declarations."""
        with self.lock:
            try:
                if self.session and not self.failed:
                    self.request("close")
            finally:
                self.transport.close()
                self.token = None
                self.session = ""
                self.instance = ""
                self.descriptor = None
                self.failed = False
