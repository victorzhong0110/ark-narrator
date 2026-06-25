"""推理网关服务（OpenAI 兼容）。控制面用 ARK_BACKEND=api 指向这里。

端点：POST /v1/chat/completions（含 stream）、GET /v1/models、GET /healthz、GET /metrics。
鉴权：Authorization: Bearer <token>（配了 token 才强制）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app.metrics import Metrics
from gateway.config import build_router, load_gateway_settings
from gateway.quota import QuotaManager
from gateway.tokens import anon_policy, load_tokens

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = load_gateway_settings()
METRICS = Metrics()
_router = None
_quota: QuotaManager | None = None
_tokens: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _quota, _tokens
    _tokens = load_tokens(settings.tokens_file)
    if not _tokens:
        # fail-closed：没配 token 默认拒绝启动；要开匿名开放模式必须显式 GW_ALLOW_ANON=true（仅 dev）
        if not settings.allow_anon:
            raise RuntimeError(
                "网关未配置任何 token → 拒绝启动（裸模型不可匿名暴露）。"
                "配 GW_TOKENS/tokens.yaml；仅本地调试可设 GW_ALLOW_ANON=true 开匿名模式")
        logger.warning("⚠ GW_ALLOW_ANON=true：网关匿名开放模式（仅 dev，绝不可用于生产）")
    _router = build_router(settings)
    _quota = QuotaManager(settings.redis_url if settings.backend == "pool" else None)
    logger.info("推理网关就绪：targets=%s default=%s fallback=%s split=%.2f tokens=%d",
                list(_router.targets), _router.default, _router.fallback, _router.split, len(_tokens))
    yield


app = FastAPI(title="ArkNarrator 推理网关", version="0.1.0", lifespan=lifespan)


def _auth(request: Request) -> dict:
    """校验 Bearer token；配了 token 才强制。返回该 token 的策略（含 name/限流/配额/可用目标）。"""
    authz = request.headers.get("authorization", "")
    token = authz[7:].strip() if authz[:7].lower() == "bearer " else ""
    if not _tokens:
        return anon_policy()
    info = _tokens.get(token)
    if not info:
        METRICS.inc("gw_auth_fail_total")
        raise HTTPException(401, "无效或缺失的网关 token")
    return info


def _split(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    msgs = [{"role": m["role"], "content": m.get("content", "")}
            for m in messages if m.get("role") in ("user", "assistant")]
    return system, msgs


@app.get("/healthz")
async def healthz():
    return {"status": "ok" if _router is not None else "starting",
            "backend": settings.backend, "model": settings.model,
            "targets": list(_router.targets) if _router else []}


class _NoTargetAllowed(Exception):
    """该 token 的 allow_targets 把所有可路由目标都过滤掉了 → 403。"""


def _route(model: str, allow: list[str]):
    targets = _router.route(model)
    if allow:
        targets = [(n, b) for n, b in targets if n in allow]
    if not targets:
        raise _NoTargetAllowed("无可用上游（受 token 的 allow_targets 限制）")
    return targets


def _serve_generate(system: str, msgs: list[dict], model: str,
                    max_tokens: int, temperature: float, allow: list[str]) -> tuple[str, str]:
    """按路由顺序尝试目标，首个成功即返回 (目标名, 文本)；全失败则抛错（跨上游故障转移）。"""
    last: Exception | None = None
    for tname, backend in _route(model, allow):
        try:
            text = backend.generate(system, msgs, max_tokens=max_tokens, temperature=temperature)
            return tname, text
        except Exception as exc:  # noqa: BLE001
            last = exc
            METRICS.inc("gw_target_fail_total", {"target": tname})
            logger.warning("目标 %s 失败，转下一个：%s", tname, exc)
    raise RuntimeError(f"所有上游均失败：{last}")


def _serve_stream(system: str, msgs: list[dict], model: str,
                  max_tokens: int, temperature: float, allow: list[str]) -> tuple[str, list[str]]:
    last: Exception | None = None
    for tname, backend in _route(model, allow):
        try:
            pieces = list(backend.stream(system, msgs, max_tokens=max_tokens, temperature=temperature))
            return tname, pieces
        except Exception as exc:  # noqa: BLE001
            last = exc
            METRICS.inc("gw_target_fail_total", {"target": tname})
    raise RuntimeError(f"所有上游均失败：{last}")


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return METRICS.render()


@app.get("/v1/models")
async def models(request: Request):
    _auth(request)
    return {"object": "list", "data": [{"id": settings.model, "object": "model", "owned_by": "ark"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    info = _auth(request)
    name = info["name"]
    ok, reason = _quota.check(info)             # 限流 + 日配额（计数即放行）
    if not ok:
        METRICS.inc("gw_quota_reject_total", {"token": name, "reason": reason})
        raise HTTPException(429, f"超出{'限流' if reason == 'rate' else '日配额'}（token={name}）")
    allow = info.get("allow_targets") or []
    # 资源上限（防 DoS/成本放大）：先按 Content-Length 卡体积，再卡消息条数/prompt 字符/输出 token
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_body_bytes:
        raise HTTPException(413, "请求体过大")
    raw = await request.body()
    if len(raw) > settings.max_body_bytes:      # Content-Length 缺失/不实时也兜住
        raise HTTPException(413, "请求体过大")
    body = json.loads(raw or b"{}")
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages 不能为空")
    if len(messages) > settings.max_messages:
        raise HTTPException(413, "消息条数过多")
    if sum(len(str(m.get("content", ""))) for m in messages) > settings.max_prompt_chars:
        raise HTTPException(413, "prompt 过长")
    system, msgs = _split(messages)
    model = body.get("model", settings.model)
    max_tokens = min(int(body.get("max_tokens", 320)), settings.max_output_tokens)   # 封顶输出
    temperature = float(body.get("temperature", 0.7))
    start = time.perf_counter()

    if body.get("stream"):
        async def gen():
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            try:
                tname, pieces = await asyncio.to_thread(
                    _serve_stream, system, msgs, model, max_tokens, temperature, allow)
            except Exception as exc:  # noqa: BLE001
                METRICS.inc("gw_requests_total", {"token": name, "status": "error"})
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            for piece in pieces:
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            METRICS.inc("gw_requests_total", {"token": name, "status": "200", "target": tname})
            METRICS.inc("gw_completion_chars_total", {"token": name}, value=sum(len(p) for p in pieces))
            METRICS.observe("gw_request_seconds", time.perf_counter() - start, {"token": name})
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        tname, text = await asyncio.to_thread(
            _serve_generate, system, msgs, model, max_tokens, temperature, allow)
    except _NoTargetAllowed as exc:
        METRICS.inc("gw_requests_total", {"token": name, "status": "403"})
        raise HTTPException(403, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        METRICS.inc("gw_requests_total", {"token": name, "status": "error"})
        raise HTTPException(502, f"上游推理失败：{exc}") from exc
    METRICS.inc("gw_requests_total", {"token": name, "status": "200", "target": tname})
    METRICS.inc("gw_completion_chars_total", {"token": name}, value=len(text))   # 用量计量(成本归属)
    METRICS.observe("gw_request_seconds", time.perf_counter() - start, {"token": name})
    return JSONResponse(
        headers={"X-Served-Target": tname},     # 本次实际命中的上游（pool/external），便于观测
        content={
            "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text), "total_tokens": len(text)},
        })


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run("gateway.app:app", host=os.getenv("GW_HOST", "0.0.0.0"),  # nosec B104 — 容器内绑全网卡
                port=int(os.getenv("GW_PORT", "8080")), reload=False)
