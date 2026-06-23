"""网关配置（独立于控制面 app）。全走 GW_* 环境变量。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _bool(k: str, d: bool) -> bool:
    v = os.getenv(k)
    return d if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GatewaySettings:
    # backend：pool（内部机队，Redis 服务发现）| scripted（无需节点，测试/演示）
    backend: str = field(default_factory=lambda: os.getenv("GW_BACKEND", "pool"))
    model: str = field(default_factory=lambda: os.getenv("GW_MODEL", "mlx-community/Qwen3-8B-4bit"))
    redis_url: str = field(default_factory=lambda: os.getenv("GW_REDIS_URL", "redis://localhost:6379/0"))
    node_group: str = field(default_factory=lambda: os.getenv("GW_NODE_GROUP", "models"))
    node_allowlist: str = field(default_factory=lambda: os.getenv("GW_NODE_ALLOWLIST", ""))
    disable_thinking: bool = field(default_factory=lambda: _bool("GW_DISABLE_THINKING", True))
    tokens_file: Path = field(
        default_factory=lambda: ROOT / os.getenv("GW_TOKENS_FILE", "gateway/tokens.yaml")
    )


def load_gateway_settings() -> GatewaySettings:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return GatewaySettings()


def build_backend(s: GatewaySettings):
    """按配置造网关的推理后端（pool 复用控制面的 PooledAPIBackend；scripted 测试用）。"""
    if s.backend == "scripted":
        from app.llm.scripted_backend import ScriptedBackend
        return ScriptedBackend(lambda system, msgs: f"（网关 scripted）收到：{msgs[-1]['content'] if msgs else ''}")
    if s.backend == "pool":
        from app.llm.pool_backend import PooledAPIBackend
        from app.store.redis import RedisStore
        allow = [p.strip() for p in s.node_allowlist.split(",") if p.strip()]
        return PooledAPIBackend(RedisStore(s.redis_url), s.model, group=s.node_group,
                                disable_thinking=s.disable_thinking, allowlist=allow)
    raise ValueError(f"未知 GW_BACKEND：{s.backend}")
