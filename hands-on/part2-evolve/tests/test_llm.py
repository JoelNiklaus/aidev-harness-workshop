"""The model call: one route, fixed by the experiment (GLM-5.3-Flash on the HF router, effort low)."""

import json

import httpx
import pytest

from aidev_evolve import llm as llm_mod


def reply_json(content="ok", tool_calls=None, usage=None):
    msg = {"role": "assistant", "content": content, "reasoning_content": "long thoughts"}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": usage or {"prompt_tokens": 50, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 20}}}


def make(handler, sleeps=None):
    return llm_mod.RouterLLM("hf_dummy", route="zai-org/GLM-5.3-Flash:zai-org", effort="low",
                             client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_request_pins_route_and_effort_and_returns_clean_message_and_usage():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json("hello"))

    msg, usage = make(handler)([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "bash"}}], timeout=30, temperature=0.1)
    assert seen["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert seen["auth"] == "Bearer hf_dummy"
    assert seen["body"]["model"] == "zai-org/GLM-5.3-Flash:zai-org"
    assert seen["body"]["reasoning_effort"] == "low"
    assert seen["body"]["temperature"] == 0.1 and seen["body"]["tools"][0]["function"]["name"] == "bash"
    assert msg == {"role": "assistant", "content": "hello"}
    assert usage == {"prompt_tokens": 50, "completion_tokens": 5, "cached_tokens": 20}


def test_no_tools_key_when_the_agent_passes_none():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply_json())

    make(handler)([{"role": "user", "content": "hi"}], None, timeout=30)
    assert "tools" not in seen["body"]


def test_tool_calls_are_kept():
    calls = [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
    msg, _ = make(lambda r: httpx.Response(200, json=reply_json(None, calls)))([], None, timeout=30)
    assert msg["tool_calls"] == calls and msg["content"] is None


def test_retries_rate_limits_and_server_errors_then_succeeds():
    codes = iter([429, 503, 200])
    sleeps = []

    def handler(request):
        c = next(codes)
        return httpx.Response(c, json=reply_json() if c == 200 else {"error": "busy"})

    msg, _ = make(handler, sleeps)([], None, timeout=60)
    assert msg["content"] == "ok" and len(sleeps) == 2


def test_gives_up_with_timeout_error_when_retries_would_pass_the_deadline():
    sleeps = []
    with pytest.raises(TimeoutError):
        make(lambda r: httpx.Response(503, json={}), sleeps)([], None, timeout=3)


def test_a_bad_request_is_an_error_the_agent_sees():
    with pytest.raises(llm_mod.LLMError, match="400"):
        make(lambda r: httpx.Response(400, json={"error": "messages[3]: tool_call_id missing"}))([], None, timeout=30)
