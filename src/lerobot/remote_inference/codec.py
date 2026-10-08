"""Torch adaptation over the shared bounded NumPy codec (wire-compatible)."""

from typing import Any

import torch

from lerobot.transport.wire.codec import (
    CodecLimits,
    RGBImage,
    decode_message,
    encode_message as _encode_message,
    peek_envelope,
)
from lerobot.transport.wire.protocol import Envelope, ErrorCode, ProtocolError

__all__ = ["CodecLimits", "RGBImage", "decode_message", "encode_message", "peek_envelope"]


def _tensor_to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        try:
            return value.detach().cpu().numpy()
        except (TypeError, RuntimeError) as exc:
            raise ProtocolError(ErrorCode.MALFORMED, "Unsupported tensor dtype") from exc
    return value


def encode_message(envelope: Envelope, limits: CodecLimits = CodecLimits()) -> bytes:
    """Encode finite values, adapting torch tensors within the shared resource budget."""
    return _encode_message(envelope, limits, tensor_converter=_tensor_to_numpy)
