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
from fastapi.responses import PlainTextResponse, StreamingResponse

from app.metrics import Metrics
from gateway.config import build_backend, load_gateway_settings
from gateway.tokens import load_tokens

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

settings = load_gateway_settings()
METRICS = Metrics()
_backend = None
_tokens: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backend, _tokens
    _tokens = load_tokens(settings.tokens_file)
    if not _tokens:
        logger.warning("⚠ 网关未配置 token（开放模式，仅 dev）——生产务必配 GW_TOKENS/tokens.yaml")
    _backend = build_backend(settings)
    logger.info("推理网关就绪：backend=%s model=%s tokens=%d", settings.backend, settings.model, len(_tokens))
    yield


app = FastAPI(title="ArkNarrator 推理网关", version="0.1.0", lifespan=lifespan)


def _auth(request: Request) -> str:
    """校验 Bearer token；配了 token 才强制。返回 token 名（计量用）。"""
    authz = request.headers.get("authorization", "")
    token = authz[7:].strip() if authz[:7].lower() == "bearer " else ""
    if not _tokens:
        return "anon"
    info = _tokens.get(token)
    if not info:
        METRICS.inc("gw_auth_fail_total")
        raise HTTPException(401, "无效或缺失的网关 token")
    return info["name"]


def _split(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    msgs = [{"role": m["role"], "content": m.get("content", "")}
            for m in messages if m.get("role") in ("user", "assistant")]
    return system, msgs


@app.get("/healthz")
async def healthz():
    return {"status": "ok" if _backend is not None else "starting",
            "backend": settings.backend, "model": settings.model}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return METRICS.render()


@app.get("/v1/models")
async def models(request: Request):
    _auth(request)
    return {"object": "list", "data": [{"id": settings.model, "object": "model", "owned_by": "ark"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    name = _auth(request)
    body = await request.json()
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(400, "messages 不能为空")
    system, msgs = _split(messages)
    model = body.get("model", settings.model)
    max_tokens = int(body.get("max_tokens", 320))
    temperature = float(body.get("temperature", 0.7))
    start = time.perf_counter()

    if body.get("stream"):
        async def gen():
            cid = "chatcmpl-" + uuid.uuid4().hex[:12]
            try:
                it = await asyncio.to_thread(
                    lambda: list(_backend.stream(system, msgs, max_tokens=max_tokens, temperature=temperature)))
            except Exception as exc:  # noqa: BLE001
                METRICS.inc("gw_requests_total", {"token": name, "status": "error"})
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
                return
            for piece in it:
                chunk = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": model, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            done = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            METRICS.inc("gw_requests_total", {"token": name, "status": "200"})
            METRICS.observe("gw_request_seconds", time.perf_counter() - start, {"token": name})
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        text = await asyncio.to_thread(
            _backend.generate, system, msgs, max_tokens=max_tokens, temperature=temperature)
    except Exception as exc:  # noqa: BLE001
        METRICS.inc("gw_requests_total", {"token": name, "status": "error"})
        raise HTTPException(502, f"上游推理失败：{exc}") from exc
    METRICS.inc("gw_requests_total", {"token": name, "status": "200"})
    METRICS.observe("gw_request_seconds", time.perf_counter() - start, {"token": name})
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
        "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": len(text), "total_tokens": len(text)},
    }


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run("gateway.app:app", host=os.getenv("GW_HOST", "0.0.0.0"),  # nosec B104 — 容器内绑全网卡
                port=int(os.getenv("GW_PORT", "8080")), reload=False)
