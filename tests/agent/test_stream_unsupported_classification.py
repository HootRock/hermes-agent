"""Streaming capability errors must not match upstream/CLI text (#144896)."""

from types import SimpleNamespace

import pytest

from agent.anthropic_adapter import _is_stream_unavailable_error
from agent.chat_completion_helpers import _StreamingCall
from agent.turn_api_call import _should_stream


ERRORS = [
    ("Incomplete upstream response (first attempt: status 400, upstream said: adaptive thinking is not supported on this model)", False),
    ("Incomplete upstream response (first attempt: status 400, upstream said: For 'integer' type, properties maximum, minimum are not supported)", False),
    ("Native history replay not supported: ...\nnative stderr: Error: --input-format stream-json ...", False),
    ("Incomplete upstream response: prompt is too long", False),
    ("stream_options is not supported", False),
    ("non-streaming requests are not supported on this endpoint", False),
    ("x-event-stream is not supported", False),
    ("text/event-stream-json is not supported", False),
    ("text/event-stream is not supported", True),
    ("Content-Type text/event-stream is not supported", True),
    ("event-stream is not supported", True),
    ("TEXT/EVENT-STREAM is not supported", True),
    ("text/event-stream; charset=utf-8 is not supported", True),
    ("streaming is not supported", True),
    ("stream not supported for this model", True),
    ("Stream mode is not supported", True),
]


@pytest.mark.parametrize("message,unsupported", ERRORS)
def test_stream_capability_classification(message, unsupported):
    assert _is_stream_unavailable_error(RuntimeError(message)) is unsupported


@pytest.mark.parametrize("message,unsupported,provider", [
    *(entry + ("custom",) for entry in ERRORS),
    ("Unexpected event order, got content_block_start before message_start", True, "custom"),
    ("Unexpected event order, got content_block_start before message_start", False, "bedrock"),
    ("Unexpected event order from upstream: adaptive thinking is not supported", False, "bedrock"),
])
def test_request_error_preserves_session_streaming(message, unsupported, provider):
    diagnostics = []
    agent = SimpleNamespace(
        api_mode="anthropic_messages", provider=provider, model="example",
        base_url="", _has_stream_consumers=lambda: True,
        _disable_streaming=False,
        _safe_print=lambda message, **kwargs: diagnostics.append(message),
    )
    call = _StreamingCall(agent, {}, None)
    call._maybe_disable_streaming(RuntimeError(message))
    assert agent._disable_streaming is unsupported
    assert bool(diagnostics) is unsupported
    assert _should_stream(agent) is not unsupported
