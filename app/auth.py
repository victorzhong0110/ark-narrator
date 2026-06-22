"""鉴权：静态 API key + JWT（HS256，零依赖）。

jwt 模式下，玩家身份从「鉴权服务签发的 JWT」里取（验签后的 player_id 比 body 字段可信）。
HS256 用标准库实现，不引第三方包。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def sign_jwt(payload: dict, secret: str) -> str:
    """签发 HS256 JWT（主要给测试/示例用；生产由游戏鉴权服务签发）。"""
    header = {"alg": "HS256", "typ": "JWT"}
    seg = (_b64url_encode(json.dumps(header, separators=(",", ":")).encode())
           + "." + _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()))
    sig = hmac.new(secret.encode(), seg.encode(), hashlib.sha256).digest()
    return seg + "." + _b64url_encode(sig)


def verify_jwt(token: str, secret: str, leeway: float = 0.0) -> dict | None:
    """验签 + 校验 exp。通过返回 claims，否则 None。"""
    try:
        h, p, sig = token.split(".")
    except ValueError:
        return None
    expected = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(expected, _b64url_decode(sig)):
            return None
        payload = json.loads(_b64url_decode(p))
    except Exception:  # noqa: BLE001
        return None
    exp = payload.get("exp")
    if exp is not None and time.time() > float(exp) + leeway:
        return None
    return payload
