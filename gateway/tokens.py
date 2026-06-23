"""网关 token：多租户的调用凭证。G1 用文件/env；配额在 G3。

tokens.yaml 形如：
  tokens:
    - token: ark-control-plane
      name: arknarrator
    - token: eval-token
      name: eval
也可用 GW_TOKENS=token1,token2（无名）。未配置则开放（仅 dev，会告警）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def load_tokens(path: Path) -> dict[str, dict]:
    tokens: dict[str, dict] = {}
    if path.exists():
        try:
            import yaml
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for entry in data.get("tokens", []):
                tok = str(entry.get("token", "")).strip()
                if tok:
                    tokens[tok] = {"name": str(entry.get("name", tok))}
        except Exception as exc:  # noqa: BLE001
            logger.warning("解析 tokens 文件失败：%s", exc)
    for tok in (t.strip() for t in os.getenv("GW_TOKENS", "").split(",")):
        if tok:
            tokens.setdefault(tok, {"name": tok})
    return tokens
