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
    # ---- 路由（G2）：外部 API + 路由表/策略 ----
    # external_base_url：留空=无外部；"mock"=测试用脚本上游；否则 OpenAI 兼容端点(MiniMax/DeepSeek…)
    external_base_url: str = field(default_factory=lambda: os.getenv("GW_EXTERNAL_BASE_URL", ""))
    external_key: str = field(default_factory=lambda: os.getenv("GW_EXTERNAL_KEY", "EMPTY"))
    external_model: str = field(default_factory=lambda: os.getenv("GW_EXTERNAL_MODEL", ""))
    route_default: str = field(default_factory=lambda: os.getenv("GW_ROUTE_DEFAULT", "pool"))
    route_fallback: str = field(default_factory=lambda: os.getenv("GW_ROUTE_FALLBACK", "external"))
    external_split: float = field(default_factory=lambda: float(os.getenv("GW_EXTERNAL_SPLIT", "0")))
    # 路由表："model=target,model2=target2"，如 "deepseek-chat=external,ark-local=pool"
    route_table: str = field(default_factory=lambda: os.getenv("GW_ROUTE_TABLE", ""))


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


def _build_external(s: GatewaySettings):
    """外部 API 目标：""=无；"mock"=脚本上游(测试)；否则 OpenAI 兼容端点。"""
    if not s.external_base_url:
        return None
    if s.external_base_url == "mock":
        from app.llm.scripted_backend import ScriptedBackend
        return ScriptedBackend(
            lambda system, msgs: f"（外部mock）{msgs[-1]['content'] if msgs else ''}")
    from app.llm.api_backend import APIBackend
    return APIBackend(s.external_model or s.model, s.external_base_url, s.external_key,
                      disable_thinking=False)   # 外部托管模型自带思考策略，不强关


def _parse_table(spec: str) -> dict[str, str]:
    table: dict[str, str] = {}
    for pair in spec.split(","):
        if "=" in pair:
            model, target = pair.split("=", 1)
            if model.strip() and target.strip():
                table[model.strip()] = target.strip()
    return table


def build_router(s: GatewaySettings):
    """组装路由器：pool（内部）+ external（外部，可选）两个目标 + 路由表/策略。"""
    from gateway.router import Router
    targets: dict[str, object] = {"pool": build_backend(s)}
    external = _build_external(s)
    if external is not None:
        targets["external"] = external
    fallback = s.route_fallback if s.route_fallback in targets else None
    return Router(targets=targets, table=_parse_table(s.route_table),
                  default=s.route_default if s.route_default in targets else "pool",
                  fallback=fallback, split=max(0.0, min(1.0, s.external_split)))
