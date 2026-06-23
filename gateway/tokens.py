"""网关 token：多租户的调用凭证 + 每 token 策略（限流/配额/可用目标）。

tokens.yaml 形如：
  tokens:
    - token: ark-control-plane
      name: arknarrator
      rate_limit: 600        # 每分钟请求数上限，0/缺省=不限
      daily_quota: 500000    # 每日请求数上限，0/缺省=不限
      allow_targets: [pool]  # 允许的上游(pool/external)，缺省/空=全部
    - token: eval-token
      name: eval
      rate_limit: 120
也可用 GW_TOKENS=token1,token2（无策略=不限）。未配置则开放（仅 dev，会告警）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _policy(entry: dict, name: str) -> dict:
    """归一化一条 token 的策略，缺省即不限。"""
    targets = entry.get("allow_targets") or []
    return {
        "name": name,
        "rate_limit": int(entry.get("rate_limit", 0) or 0),
        "daily_quota": int(entry.get("daily_quota", 0) or 0),
        "allow_targets": [str(t).strip() for t in targets if str(t).strip()],
    }


def anon_policy() -> dict:
    """开放模式（未配 token）的放行策略：不限、全目标。"""
    return {"name": "anon", "rate_limit": 0, "daily_quota": 0, "allow_targets": []}


def load_tokens(path: Path) -> dict[str, dict]:
    tokens: dict[str, dict] = {}
    if path.exists():
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for entry in data.get("tokens", []):
                tok = str(entry.get("token", "")).strip()
                if tok:
                    tokens[tok] = _policy(entry, str(entry.get("name", tok)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析 tokens 文件失败：%s", exc)
    for tok in (t.strip() for t in os.getenv("GW_TOKENS", "").split(",")):
        if tok:
            tokens.setdefault(tok, _policy({}, tok))
    return tokens
