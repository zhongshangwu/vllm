# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Educational inference trace logs for mastering LLM serving internals.

Enable at runtime::

    export VLLM_INFERENCE_TRACE=1
    export VLLM_LOGGING_LEVEL=INFO

Optional controls::

    export VLLM_INFERENCE_TRACE_EVERY=1      # log every N-th event (default 1)
    export VLLM_INFERENCE_TRACE_MAX_IDS=32     # max token ids shown per field
    export VLLM_INFERENCE_TRACE_REQ=abc123     # only log matching request_id

Grep server logs::

    grep INFERENCE_TRACE /path/to/vllm.log

Phases (in typical request order)::

    tokenize → request_arrive → schedule → kv_alloc → prepare_inputs
    → attention_meta → embed → forward → sample → detokenize → output
    → kv_free

Ascend-specific phases: ``ascend_attn_state``, ``aclgraph`` (see acl_graph.py).

See ``docs/inference_trace_guide.md`` in the llm-inference repo for a
walkthrough that maps each log line to prefill/decode/KV-cache concepts.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED: bool | None = None
_EVERY: int | None = None
_MAX_IDS: int | None = None
_REQ_FILTER: str | None = None
_COUNTER: int = 0


def _load_config() -> None:
    global _ENABLED, _EVERY, _MAX_IDS, _REQ_FILTER
    if _ENABLED is not None:
        return
    raw = os.environ.get("VLLM_INFERENCE_TRACE", "0")
    _ENABLED = raw not in ("0", "", "false", "False", "FALSE")
    _EVERY = max(1, int(os.environ.get("VLLM_INFERENCE_TRACE_EVERY", "1")))
    _MAX_IDS = max(0, int(os.environ.get("VLLM_INFERENCE_TRACE_MAX_IDS", "32")))
    _REQ_FILTER = os.environ.get("VLLM_INFERENCE_TRACE_REQ") or None


def inference_trace_enabled() -> bool:
    _load_config()
    return bool(_ENABLED)


def _format_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        items = (f"{k}:{_format_value(v)}" for k, v in value.items())
        return "{" + ",".join(items) + "}"
    text = str(value)
    return text.replace(" ", "_").replace("\n", "\\n")


def _truncate_ids(ids: list[int] | None) -> str:
    if ids is None:
        return "none"
    _load_config()
    assert _MAX_IDS is not None
    if _MAX_IDS == 0:
        return f"len={len(ids)}"
    if len(ids) <= _MAX_IDS:
        return _format_value(ids)
    head = ids[: _MAX_IDS // 2]
    tail = ids[-(_MAX_IDS // 2) :]
    return _format_value(head) + "..." + _format_value(tail) + f"(len={len(ids)})"


def classify_step_kind(per_req_tokens: list[int]) -> str:
    """Heuristic: uniform_decode | prefill | mixed | empty."""
    if not per_req_tokens:
        return "empty"
    if all(t == 1 for t in per_req_tokens):
        return "uniform_decode"
    if len(per_req_tokens) == 1 and per_req_tokens[0] > 1:
        return "prefill"
    return "mixed"


def classify_request_phase(
    num_computed_tokens: int,
    num_scheduled_tokens: int,
) -> str:
    """Per-request phase within one engine step."""
    if num_computed_tokens == 0:
        return "prefill_start"
    if num_scheduled_tokens == 1:
        return "decode"
    if num_scheduled_tokens > 1:
        return "chunked_prefill"
    return "unknown"


def inference_trace(
    phase: str,
    *,
    request_id: str | None = None,
    **fields: Any,
) -> None:
    """Emit one grep-friendly educational trace line."""
    if not inference_trace_enabled():
        return

    _load_config()
    assert _EVERY is not None
    assert _REQ_FILTER is not None or True

    if request_id is not None and _REQ_FILTER is not None:
        if _REQ_FILTER not in request_id:
            return

    global _COUNTER
    _COUNTER += 1
    if (_COUNTER - 1) % _EVERY != 0:
        return

    # Normalize common field names for readability.
    if "prompt_token_ids" in fields:
        fields["prompt_token_ids"] = _truncate_ids(fields["prompt_token_ids"])
    if "input_ids" in fields and isinstance(fields["input_ids"], list):
        fields["input_ids"] = _truncate_ids(fields["input_ids"])
    if "new_token_ids" in fields:
        fields["new_token_ids"] = _truncate_ids(fields["new_token_ids"])
    if "sampled_token_ids" in fields:
        val = fields["sampled_token_ids"]
        if isinstance(val, list) and val and isinstance(val[0], list):
            fields["sampled_token_ids"] = _format_value(val)
        elif isinstance(val, list):
            fields["sampled_token_ids"] = _truncate_ids(val)

    parts = " ".join(f"{k}={_format_value(v)}" for k, v in fields.items())
    req_part = f" request_id={request_id}" if request_id else ""
    logger.info(
        "INFERENCE_TRACE phase=%s step=%s%s %s",
        phase,
        _COUNTER,
        req_part,
        parts,
    )
