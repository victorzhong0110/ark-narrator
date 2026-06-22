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
    # backend: "mlx"（本地真模型）| "scripted"（无需模型，测试/演示管线用）
    backend: str = field(default_factory=lambda: os.getenv("ARK_BACKEND", "mlx"))
    model_path: str = field(
        default_factory=lambda: os.getenv("ARK_MODEL_PATH", "mlx-community/Qwen3-8B-4bit")
    )
    # 适配器目录：可选。base+卡+RAG 路线默认不需要微调权重。
    adapter_dir: str | None = field(
        default_factory=lambda: os.getenv("ARK_ADAPTER_DIR") or None
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
    cloud_audit_provider: str = field(
        default_factory=lambda: os.getenv("ARK_CLOUD_AUDIT_PROVIDER", "aliyun")
    )
    # 云端审核不可用时是否放行（fail-open）。安全产品默认 fail-closed=False→放行本地结果，
    # 但对最高敏感类别仍由本地规则兜底。可按合规要求改为 True（云端挂掉就拦）。
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

    # ---- 合规：AI 标识 ----
    ai_label: str = field(
        default_factory=lambda: os.getenv(
            "ARK_AI_LABEL", "本回复由 AI 生成 · 角色与世界观版权归鹰角网络所有"
        )
    )


def load_settings() -> Settings:
    """加载配置（如有 .env 先读入环境）。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    return Settings()
