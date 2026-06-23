"""内部消费者后端：api 模式且配 internal_api_key 时用独立 token，否则复用主后端。"""

from __future__ import annotations

from app.config import Settings
from app.llm.factory import get_internal_backend


def test_reuses_main_when_not_api():
    s = Settings(backend="scripted")
    main = object()
    assert get_internal_backend(s, main) is main          # 非 api → 不分账


def test_reuses_main_when_api_without_internal_key():
    s = Settings(backend="api", api_base_url="http://gw/v1", api_key="main", internal_api_key="")
    main = object()
    assert get_internal_backend(s, main) is main          # api 但没配内部 token → 复用


def test_separate_backend_when_internal_key_set():
    s = Settings(backend="api", api_base_url="http://gw/v1", api_key="main",
                 internal_api_key="internal", model_path="m")
    main = object()
    b = get_internal_backend(s, main)
    assert b is not main                                   # 独立后端
    assert b.label == "api:m"
