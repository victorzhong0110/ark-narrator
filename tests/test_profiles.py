"""部署档位预设：注入默认 / 不覆盖显式值 / 未知档 / load_settings 优先级。"""

from __future__ import annotations

import os

from app.profiles import PROFILES, apply_profile


def test_apply_profile_sets_defaults():
    env: dict[str, str] = {}
    assert apply_profile("budget", env=env) is True
    assert env["ARK_BACKEND"] == "mlx"
    assert env["ARK_SCENE_TAGGER"] == "heuristic"
    assert env["ARK_CLOUD_AUDIT"] == "true"


def test_apply_profile_does_not_override_explicit():
    env = {"ARK_BACKEND": "scripted"}      # 显式值应保留
    apply_profile("flagship", env=env)
    assert env["ARK_BACKEND"] == "scripted"
    assert env["ARK_CLOUD_AUDIT_PROVIDER"] == "aliyun"   # 没设的才填


def test_apply_profile_unknown_returns_false():
    env: dict[str, str] = {}
    assert apply_profile("does-not-exist", env=env) is False
    assert env == {}


def test_all_profiles_have_backend():
    for name, preset in PROFILES.items():
        assert "ARK_BACKEND" in preset, name


def test_load_settings_uses_profile(monkeypatch):
    # 清掉档位涉及的键，设 ARK_PROFILE，验证 load_settings 应用预设
    for k in PROFILES["budget"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ARK_PROFILE", "budget")
    # apply_profile 用 setdefault 写 os.environ；monkeypatch 会在测试后回滚
    from app.config import load_settings

    s = load_settings()
    assert s.backend == "mlx"
    assert s.scene_tagger == "heuristic"
    assert s.cloud_audit_enabled is True
    # 清理 apply_profile 经 setdefault 落入的键，防止泄漏到其它测试
    for k in PROFILES["budget"]:
        os.environ.pop(k, None)


def test_explicit_env_beats_profile(monkeypatch):
    for k in PROFILES["budget"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ARK_PROFILE", "budget")
    monkeypatch.setenv("ARK_BACKEND", "scripted")   # 显式覆盖
    from app.config import load_settings

    assert load_settings().backend == "scripted"
    for k in PROFILES["budget"]:
        if k != "ARK_BACKEND":
            os.environ.pop(k, None)
