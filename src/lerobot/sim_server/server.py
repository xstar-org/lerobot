"""Bounded simulator serving. All physics and world mutations run on the serving worker."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty
from threading import Event

import numpy as np

from lerobot.sims.adapters import hold_action
from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.codec import CodecLimits, decode_message, encode_message
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport

from .contracts import SimDescriptor, validate_result
from .protocol import deployment_prefix, instance_prefix, session_prefix

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Bound simulator resources and select explicit Zenoh endpoints."""

    sim: BackendConfig = field(default_factory=BackendConfig)
    deployment: str = "default"
    zenoh: ZenohConfig = field(default_factory=lambda: ZenohConfig(listen_endpoints=["tcp/0.0.0.0:7448"]))
    max_envs: int = 32
    max_sessions: int = 1
    request_timeout_s: float = 120
    presence_grace_s: float = 10
    command_timeout_s: float = 1


@dataclass
class _Session:
    backend: Backend
    clock: str
    prefix: str
    presence: object
    token: object
    actions: object
    target: np.ndarray
    generation: int = 0
    requests: set[str] = field(default_factory=set)
    next_tick: float = 0
    last_command: float = 0
    opened_at: float = field(default_factory=time.monotonic)
    seen_client: bool = False
    paused: bool = False
    fault: str | None = None


class SimServer:
    """Serve world operations and physics ticks from one owner thread."""

    def __init__(self, config: ServerConfig):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        if config.max_envs < 1 or config.max_sessions < 1 or config.command_timeout_s <= 0:
            raise ValueError("Server resource bounds must be positive")
        self.instance = uuid.uuid4().hex
        self.transport = ZenohTransport(config.zenoh)
        self.sessions: dict[str, _Session] = {}
        self.ready = Event()
        self.stop = Event()
        self.descriptor: SimDescriptor

    def serve(self) -> None:
        """Pump bounded requests and realtime worlds until stopped."""
        try:
            # Includes descriptor probing; native renderer setup stays on the owner thread.
            probe = Backend(self.config.sim)
            try:
                self.descriptor = probe.descriptor()
            finally:
                probe.close()
            self.transport.open()
            root = deployment_prefix(self.config.deployment)
            instance = instance_prefix(self.config.deployment, self.instance)
            describe = self.transport.declare_queryable(
                root + "/describe", reply_timeout=self.config.request_timeout_s
            )
            requests = self.transport.declare_queryable(
                instance + "/**", reply_timeout=self.config.request_timeout_s
            )
            self.transport.declare_token(instance + "/alive")
            self.ready.set()
            while not self.stop.is_set():
                for channel in (describe, requests):
                    for _ in range(8):
                        try:
                            pending = channel.get()
                        except Empty:
                            break
                        self._reply(pending)
                self._tick()
                self.stop.wait(0.002)
        finally:
            for session_id in list(self.sessions):
                try:
                    self._close_session(session_id)
                except Exception:
                    logger.exception("Failed to close simulator session %s", session_id)
            self.transport.close()

    def _reply(self, pending) -> None:
        message = None
        try:
            message = decode_message(pending.payload)
            body = self._dispatch(pending.key, message)
            reply = message.reply(MessageType.ACK if message.session_id else MessageType.ACCEPTED, body)
        except Exception as exc:
            code = exc.code if isinstance(exc, ProtocolError) else ErrorCode.EXECUTION
            reply = (message or Envelope(MessageType.DESCRIBE)).error(code, str(exc))
            logger.debug("Simulator request failed", exc_info=True)
        pending.reply(encode_message(reply))

    def _dispatch(self, key: str, message: Envelope) -> dict:
        if key == deployment_prefix(self.config.deployment) + "/describe":
            return {"instance": self.instance, "descriptor": self.descriptor.to_dict()}
        if message.instance_id != self.instance:
            raise ProtocolError(ErrorCode.STALE, "Simulator boot changed")
        if key == instance_prefix(self.config.deployment, self.instance) + "/open":
            return self._open(message.body)
        session = self.sessions.get(message.session_id)
        if session is None or not key.startswith(session.prefix + "/"):
            raise ProtocolError(ErrorCode.STALE, "Unknown simulator session")
        if message.generation != session.generation:
            raise ProtocolError(ErrorCode.STALE, "World generation changed")
        if message.request_id in session.requests:
            raise ProtocolError(ErrorCode.STALE, "Duplicate request; mutations are never replayed")
        if len(session.requests) >= 32768:
            raise ProtocolError(ErrorCode.UNSUPPORTED, "Session request budget exhausted; reopen the session")
        session.requests.add(message.request_id)
        operation = key.rsplit("/", 1)[-1]
        body = message.body
        if operation == "control":
            operation = body.get("operation", "")
        if operation == "close":
            self._close_session(message.session_id)
            return {}
        if session.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, session.fault)
        if operation == "reset":
            try:
                result = session.backend.reset(body.get("seeds"), body.get("env_ids"))
            except Exception as exc:
                session.fault = f"Simulator reset failed: {exc}"
                raise
            session.generation += 1
            session.requests.clear()
            session.target = self._initial_target(session.backend, result)
            session.last_command = time.monotonic()
            return {"generation": session.generation, "result": result.to_dict()}
        if operation == "step":
            if session.clock != "lockstep" or body.get("n_ticks", 1) != 1:
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Step requires lockstep with n_ticks=1")
            try:
                result = session.backend.step(body["actions"])
            except Exception as exc:
                session.fault = f"Simulator step failed: {exc}"
                raise
            validate_result(result, self.descriptor, session.backend.num_envs)
            return {"result": result.to_dict()}
        if operation in {"apply", "hold"}:
            if session.clock != "realtime":
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Commands require realtime mode")
            if operation == "apply":
                actions = np.asarray(body["actions"])
                if actions.shape != session.target.shape or not np.isfinite(actions).all():
                    raise ProtocolError(ErrorCode.INCOMPATIBLE, "Command shape or values invalid")
                session.target = actions.astype(np.float32, copy=True)
            else:
                session.target = hold_action(
                    session.target,
                    self.descriptor.control,
                    self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                )
            session.last_command = time.monotonic()
            return {"applied": session.target}
        if operation == "snapshot":
            return {"result": session.backend.snapshot().to_dict()}
        if operation == "render":
            return {"frames": session.backend.render()}
        if operation in {"pause", "resume"}:
            session.paused = operation == "pause"
            session.next_tick = time.monotonic() + 1 / self.config.sim.fps
            return {}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def _initial_target(self, backend, result):
        dim = self.descriptor.action_feature.shape[0]
        target = np.zeros((backend.num_envs, dim), dtype=np.float32)
        if self.descriptor.control in {"position", "eef_pose"}:
            state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if state is None or not set(self.descriptor.action_feature.names).issubset(state.names):
                raise ValueError("Position control requires observed initial positions")
            target = result.obs[state.name][
                :, [state.names.index(n) for n in self.descriptor.action_feature.names]
            ].copy()
        return target

    def _open(self, body: dict) -> dict:
        count = body.get("num_envs", 1)
        clock = body.get("clock", "lockstep")
        if (
            type(count) is not int
            or not 1 <= count <= self.config.max_envs
            or clock not in self.descriptor.clocks
        ):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Unsupported batch size or clock")
        batch_bytes = count * sum(
            int(np.prod(f.shape)) * np.dtype(f.dtype).itemsize for f in self.descriptor.features
        )
        if batch_bytes + 65536 > min(CodecLimits().max_payload_bytes, self.config.zenoh.max_payload_bytes):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Observation batch exceeds transport payload budget")
        if len(self.sessions) >= self.config.max_sessions:
            raise ProtocolError(ErrorCode.BUSY, "Simulator capacity is owned by another session")
        backend = Backend(self.config.sim, count, body.get("task_group"), body.get("task_id", 0))
        try:
            result = backend.reset(body.get("seeds"))
            validate_result(result, backend.descriptor(), count)
            target = self._initial_target(backend, result)
            session_id = uuid.uuid4().hex
            prefix = session_prefix(self.config.deployment, self.instance, session_id)
            session = _Session(
                backend,
                clock,
                prefix,
                self.transport.subscribe_liveliness(prefix + "/client_alive"),
                self.transport.declare_token(prefix + "/alive"),
                self.transport.subscribe(prefix + "/act", capacity=8),
                target,
            )
            session.next_tick = time.monotonic() + 1 / self.config.sim.fps
            session.last_command = time.monotonic()
            self.sessions[session_id] = session
            return {
                "session": session_id,
                "result": result.to_dict(),
                "generation": 0,
                "descriptor": backend.descriptor().to_dict(),
            }
        except BaseException:
            backend.close()
            raise

    def _tick(self):
        now = time.monotonic()
        for session_id, session in list(self.sessions.items()):
            lost = session.presence.dropped > 0
            while True:
                try:
                    event = session.presence.get()
                    if event.alive:
                        session.seen_client = True
                    elif session.seen_client:
                        lost = True
                except Empty:
                    break
            if lost or (not session.seen_client and now - session.opened_at > self.config.presence_grace_s):
                self._close_session(session_id)
                continue
            while True:
                try:
                    message = decode_message(session.actions.get())
                    if message.instance_id != self.instance or message.session_id != session_id:
                        continue
                    self._dispatch(
                        session.prefix + "/control",
                        Envelope(
                            MessageType.CONTROL,
                            self.instance,
                            session_id,
                            message.generation,
                            message.request_id,
                            {"operation": "apply", **message.body},
                        ),
                    )
                except Empty:
                    break
                except Exception:
                    logger.debug("Dropped invalid realtime command", exc_info=True)
            if (
                session.clock != "realtime"
                or session.paused
                or session.fault is not None
                or now < session.next_tick
            ):
                continue
            try:
                if now - session.last_command > self.config.command_timeout_s:
                    session.target = hold_action(
                        session.target,
                        self.descriptor.control,
                        self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                    )
                result = session.backend.step(session.target)
                validate_result(result, self.descriptor, session.backend.num_envs)
                if self.descriptor.control == "delta":
                    session.target = hold_action(
                        session.target,
                        "delta",
                        self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                    )
                self.transport.publish(
                    session.prefix + "/obs",
                    encode_message(
                        Envelope(
                            MessageType.OBSERVATION,
                            self.instance,
                            session_id,
                            session.generation,
                            uuid.uuid4().hex,
                            {"result": result.to_dict()},
                        )
                    ),
                )
                session.next_tick = now + 1 / self.config.sim.fps
            except Exception as exc:
                session.fault = f"Simulator execution failed: {exc}"
                logger.exception("Simulator session failed")

    def _close_session(self, session_id):
        session = self.sessions.pop(session_id)
        try:
            session.backend.close()
        finally:
            session.actions.close()
            session.presence.close()
            session.token.undeclare()
