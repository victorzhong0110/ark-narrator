"""运行配置。

全部走环境变量 / .env，提供合理默认值，便于「本地优先、云端审核可开关」。
配置对象不可变（frozen dataclass）——加载一次，全程只读。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 项目根目录（app/config.py 的上两级）
ROOT = Path(__file__).resolve().parent.parent


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """整套服务的只读配置。"""

    # ---- 后端 ----
    # backend: "mlx"（本地真模型）| "api"（OpenAI 兼容云端）| "scripted"（无需模型，测试/演示）
    backend: str = field(default_factory=lambda: os.getenv("ARK_BACKEND", "mlx"))
    model_path: str = field(
        default_factory=lambda: os.getenv("ARK_MODEL_PATH", "mlx-community/Qwen3-8B-4bit")
    )
    # 适配器目录：可选。base+卡+RAG 路线默认不需要微调权重。
    adapter_dir: str | None = field(
        default_factory=lambda: os.getenv("ARK_ADAPTER_DIR") or None
    )
    # api 后端（backend=api 时用；OpenAI 兼容：MiniMax/DeepSeek/通义等）
    api_base_url: str = field(default_factory=lambda: os.getenv("ARK_API_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("ARK_API_KEY", ""))
    # pool 后端（backend=pool）：自注册节点池的组名（Mac 节点心跳注册到这个组）
    node_group: str = field(default_factory=lambda: os.getenv("ARK_NODE_GROUP", "models"))
    # 关闭思考链（自托管 Qwen 节点：经 chat_template_kwargs 传给 mlx_lm.server/vLLM）。
    # 云端通用 OpenAI 端点可能不支持，故默认关；pool 后端默认开。
    disable_thinking: bool = field(
        default_factory=lambda: _env_bool("ARK_DISABLE_THINKING", False)
    )

    # ---- 生成参数 ----
    temperature: float = field(default_factory=lambda: _env_float("ARK_TEMPERATURE", 0.7))
    max_tokens: int = field(default_factory=lambda: _env_int("ARK_MAX_TOKENS", 320))
    max_history_turns: int = field(
        default_factory=lambda: _env_int("ARK_MAX_HISTORY_TURNS", 8)
    )

    # ---- 入口护栏 ----
    max_input_chars: int = field(
        default_factory=lambda: _env_int("ARK_MAX_INPUT_CHARS", 800)
    )
    rate_limit_per_min: int = field(
        default_factory=lambda: _env_int("ARK_RATE_LIMIT_PER_MIN", 30)
    )
    session_risk_threshold: int = field(
        default_factory=lambda: _env_int("ARK_SESSION_RISK_THRESHOLD", 5)
    )

    # ---- 出口护栏：云端审核（默认关，留开关）----
    cloud_audit_enabled: bool = field(
        default_factory=lambda: _env_bool("ARK_CLOUD_AUDIT", False)
    )
    # provider: mock（无凭证可跑，dev/CI）| http（POST 到你的审核网关）| aliyun | tencent
    cloud_audit_provider: str = field(
        default_factory=lambda: os.getenv("ARK_CLOUD_AUDIT_PROVIDER", "mock")
    )
    cloud_audit_timeout: float = field(
        default_factory=lambda: _env_float("ARK_CLOUD_AUDIT_TIMEOUT", 2.0)
    )
    cloud_audit_retries: int = field(
        default_factory=lambda: _env_int("ARK_CLOUD_AUDIT_RETRIES", 1)
    )
    cloud_audit_cache_ttl: float = field(
        default_factory=lambda: _env_float("ARK_CLOUD_AUDIT_CACHE_TTL", 300.0)
    )
    # 云端审核不可用时是否拦截（fail-closed）。默认 False→放行本地结果（本地规则仍兜底）；
    # 按合规要求可改 True（云端挂掉就拦，宁可错杀）。
    cloud_audit_fail_closed: bool = field(
        default_factory=lambda: _env_bool("ARK_CLOUD_AUDIT_FAIL_CLOSED", False)
    )

    # ---- RAG ----
    rag_enabled: bool = field(default_factory=lambda: _env_bool("ARK_RAG", True))
    rag_top_k: int = field(default_factory=lambda: _env_int("ARK_RAG_TOP_K", 3))

    # ---- 场景情绪判定：llm（读语义，准）| heuristic（关键词，零成本兜底）----
    scene_tagger: str = field(
        default_factory=lambda: os.getenv("ARK_SCENE_TAGGER", "llm")
    )

    # ---- 状态存储：memory（单机）| sqlite（单机持久）| redis（多 worker 共享）----
    store: str = field(default_factory=lambda: os.getenv("ARK_STORE", "memory"))
    # 长期记忆：每 N 轮 用户×角色 对话滚动摘要一次；上限字数
    memory_every: int = field(default_factory=lambda: _env_int("ARK_MEMORY_EVERY", 6))
    memory_max_chars: int = field(
        default_factory=lambda: _env_int("ARK_MEMORY_MAX_CHARS", 600)
    )

    # ---- 路径 ----
    characters_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("ARK_CHARACTERS_DIR", "data/characters")
    )
    lore_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("ARK_LORE_DIR", "data/lore")
    )
    audit_log_path: Path = field(
        default_factory=lambda: ROOT / os.getenv("ARK_AUDIT_LOG", "logs/audit.jsonl")
    )
    # 世界观/IP 配置（换 IP 只改这份；缺省=明日方舟）
    world_config: Path = field(
        default_factory=lambda: ROOT / os.getenv("ARK_WORLD_CONFIG", "data/world.yaml")
    )

    # ---- 生产服务（C）----
    # API 鉴权 key：留空=不鉴权；设了则 /chat /stream 需带 X-API-Key
    api_auth_key: str = field(default_factory=lambda: os.getenv("ARK_API_AUTH_KEY", ""))
    # CORS 允许来源（逗号分隔；默认 * 仅便于本地，上线务必收紧）
    cors_origins: str = field(default_factory=lambda: os.getenv("ARK_CORS_ORIGINS", "*"))
    # 单实例并发上限（单模型不能无限并发；超过即 429）
    max_concurrency: int = field(default_factory=lambda: _env_int("ARK_MAX_CONCURRENCY", 8))
    # 单次请求超时（秒）；超时返回 504，避免卡死请求堆积
    request_timeout: float = field(
        default_factory=lambda: _env_float("ARK_REQUEST_TIMEOUT", 60.0)
    )

    # ---- 合规：AI 标识 ----
    ai_label: str = field(
        default_factory=lambda: os.getenv(
            "ARK_AI_LABEL", "本回复由 AI 生成 · 角色与世界观版权归鹰角网络所有"
        )
    )


def load_settings() -> Settings:
    """加载配置。优先级：shell env > .env > 部署档位(ARK_PROFILE) > 硬默认。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")          # 不覆盖已有 shell env
    except ImportError:
        pass
    from app.profiles import apply_profile   # 在 .env 之后注入档位默认（setdefault）

    apply_profile(os.getenv("ARK_PROFILE", ""))
    return Settings()
