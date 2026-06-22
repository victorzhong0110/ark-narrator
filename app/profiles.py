"""部署档位预设（按公司资金/硬件分档）。

一个档位 = 一组配置默认值。`ARK_PROFILE=budget|standard|flagship` 一键切换整套栈
（后端 / 模型 / 审核 / 场景判定 / 生成参数），免得逐个调十几个环境变量。

优先级：显式 shell env > .env 文件 > 档位预设 > 代码硬默认。
（预设用 setdefault 注入，所以只填「你没设过的」键。）

各档位的取舍与成本见 docs/DEPLOYMENT_TIERS.md。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

PROFILES: dict[str, dict[str, str]] = {
    # 轻量自托管：一台 GPU 机/Apple Silicon，本地 8B，场景判定走零成本关键词版
    "budget": {
        "ARK_BACKEND": "mlx",
        "ARK_MODEL_PATH": "mlx-community/Qwen3-8B-4bit",
        "ARK_SCENE_TAGGER": "heuristic",       # 省一次 LLM 调用
        "ARK_CLOUD_AUDIT": "true",
        "ARK_CLOUD_AUDIT_PROVIDER": "mock",     # 上线换 http/aliyun
        "ARK_STORE": "sqlite",                  # 单机持久
        "ARK_MAX_TOKENS": "240",
        "ARK_RAG_TOP_K": "3",
    },
    # 标准混合：自托管中端 GPU 或 API 生成 + 商用云审网关 + LLM 场景判定
    "standard": {
        "ARK_BACKEND": "api",
        "ARK_SCENE_TAGGER": "llm",
        "ARK_CLOUD_AUDIT": "true",
        "ARK_CLOUD_AUDIT_PROVIDER": "http",
        "ARK_STORE": "redis",                   # 多 worker 共享
        "ARK_MAX_TOKENS": "320",
        "ARK_RAG_TOP_K": "3",
    },
    # 云旗舰：前沿大模型 API + 厂商云审 + 更大上下文/记忆，追求最佳人设保真
    "flagship": {
        "ARK_BACKEND": "api",
        "ARK_SCENE_TAGGER": "llm",
        "ARK_CLOUD_AUDIT": "true",
        "ARK_CLOUD_AUDIT_PROVIDER": "aliyun",
        "ARK_STORE": "redis",
        "ARK_MAX_TOKENS": "400",
        "ARK_RAG_TOP_K": "4",
    },
}


def apply_profile(name: str, env: dict | None = None) -> bool:
    """把档位预设作为默认值注入环境（setdefault，不覆盖已设值）。返回是否命中档位。"""
    target = env if env is not None else os.environ
    preset = PROFILES.get(name.lower()) if name else None
    if not preset:
        if name:
            logger.warning("未知部署档位：%s（可选 %s）", name, "/".join(PROFILES))
        return False
    for k, v in preset.items():
        target.setdefault(k, v)
    logger.info("已应用部署档位：%s", name)
    return True
