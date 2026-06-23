"""网关路由策略：按模型、内部优先外部兜底、百分比分流。"""

from __future__ import annotations

from gateway.router import Router


class _B:  # 假后端，route 只用 targets 的 key
    pass


def _names(route):
    return [n for n, _ in route]


def _router(**kw):
    return Router(targets={"pool": _B(), "external": _B()}, **kw)


def test_explicit_model_routes_to_external():
    r = _router(table={"deepseek-chat": "external"}, default="pool", fallback="external")
    assert _names(r.route("deepseek-chat"))[0] == "external"


def test_default_internal_first_with_fallback():
    r = _router(default="pool", fallback="external")
    assert _names(r.route("x")) == ["pool", "external"]


def test_no_fallback_single_target():
    r = Router(targets={"pool": _B()}, default="pool", fallback=None)
    assert _names(r.route("x")) == ["pool"]


def test_unknown_target_in_table_falls_to_default():
    r = _router(table={"m": "nope"}, default="pool", fallback="external")
    assert _names(r.route("m"))[0] == "pool"


def test_split_half_routes_external_first():
    r = _router(default="pool", fallback="external", split=0.5)
    ext = sum(1 for _ in range(100) if _names(r.route("x"))[0] == "external")
    assert ext == 50


def test_split_full_all_external_first():
    r = _router(default="pool", fallback="external", split=1.0)
    assert all(_names(r.route("x"))[0] == "external" for _ in range(20))
