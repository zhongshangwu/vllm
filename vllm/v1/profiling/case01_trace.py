# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Case01 profiling logs for scheduler / cudagraph / ACL graph analysis.

Enable at runtime:
    export VLLM_CASE01_PROFILE=1
    export VLLM_LOGGING_LEVEL=INFO

Optional sampling (log every N-th profile event, default 1 = all):
    export VLLM_CASE01_PROFILE_EVERY=10

Grep server logs:
    grep CASE01_PROFILE
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED: bool | None = None
_EVERY: int | None = None
_COUNTER: int = 0


def _load_config() -> None:
    global _ENABLED, _EVERY
    if _ENABLED is not None:
        return
    raw = os.environ.get("VLLM_CASE01_PROFILE", "0")
    _ENABLED = raw not in ("0", "", "false", "False", "FALSE")
    _EVERY = max(1, int(os.environ.get("VLLM_CASE01_PROFILE_EVERY", "1")))


def case01_enabled() -> bool:
    _load_config()
    return bool(_ENABLED)


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    return str(value).replace(" ", "_")


def case01_log(tag: str, **fields: Any) -> None:
    """Emit one grep-friendly profile line when VLLM_CASE01_PROFILE=1."""
    if not case01_enabled():
        return

    global _COUNTER
    _COUNTER += 1
    _load_config()
    assert _EVERY is not None
    if (_COUNTER - 1) % _EVERY != 0:
        return

    parts = " ".join(f"{k}={_format_value(v)}" for k, v in fields.items())
    logger.info("CASE01_PROFILE event=%s seq=%s %s", tag, _COUNTER, parts)


def case01_classify_scheduler_step(per_req_tokens: list[int]) -> str:
    """Heuristic step kind for log interpretation."""
    if not per_req_tokens:
        return "empty"
    if all(t == 1 for t in per_req_tokens):
        return "uniform_decode"
    if len(per_req_tokens) == 1 and per_req_tokens[0] > 1:
        return "prefill"
    return "mixed"
