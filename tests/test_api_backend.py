"""API 后端（OpenAI 兼容）：注入假 client，不打真网络。"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.llm.api_backend import APIBackend
from app.llm.factory import get_backend


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content=None, delta=None):
        self.message = _Msg(content)
        self.delta = _Msg(delta)


class _Resp:
    def __init__(self, choices):
        self.choices = choices


class _FakeCompletions:
    def create(self, *, stream=False, **kw):
        if stream:
            return iter([_Resp([_Choice(delta=d)]) for d in ["你", "好", "呀"]])
        return _Resp([_Choice(content="<think>略</think>老板，没问题！")])


class _FakeClient:
    """模拟 openai 客户端：create() 按 stream 返回整段或分片。"""

    def __init__(self):
        self.chat = type("C", (), {"completions": _FakeCompletions()})()


def test_api_generate_strips_think():
    be = APIBackend("m", "http://x", "k", client=_FakeClient())
    out = be.generate("sys", [{"role": "user", "content": "hi"}])
    assert out == "老板，没问题！"          # <think> 被去掉


def test_api_stream_yields_chunks():
    be = APIBackend("m", "http://x", "k", client=_FakeClient())
    chunks = list(be.stream("sys", [{"role": "user", "content": "hi"}]))
    assert "".join(chunks) == "你好呀"


def test_factory_api_requires_creds():
    with pytest.raises(ValueError):
        get_backend(Settings(backend="api", api_base_url="", api_key=""))


def test_disable_thinking_passes_extra_body():
    seen = {}

    class _Cap:
        def create(self, **kw):
            seen.update(kw)
            return _Resp([_Choice(content="ok")])

    client = type("C", (), {"chat": type("X", (), {"completions": _Cap()})()})()
    APIBackend("m", "u", "k", disable_thinking=True, client=client).generate(
        "s", [{"role": "user", "content": "hi"}])
    assert seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
