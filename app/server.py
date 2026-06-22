"""ArkNarrator 干员对话服务（FastAPI）。

端点：
  GET  /health      存活检测 + 模型名
  GET  /characters  可对话干员列表（含展示信息）
  POST /chat        单次对话（同步，已过出入口护栏）
  POST /stream      流式：先生成→审核→再把通过后的安全文本逐字推送
  GET  /            内嵌 Demo（带 AI 标识与免责声明）

关键安全取舍：流式推送的是「审核之后」的文本（守出口原则），牺牲少量首字延迟换安全。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.build import build_orchestrator
from app.config import load_settings
from app.metrics import METRICS
from app.orchestrator import DialogueOrchestrator, Reply

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

orchestrator: DialogueOrchestrator | None = None
settings = load_settings()
_ready = False
_sema = asyncio.Semaphore(max(1, settings.max_concurrency))
_INSTANCE = os.getenv("HOSTNAME") or socket.gethostname()   # 容器内每实例唯一


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, _ready
    from app.tracing import setup_tracing
    setup_tracing(app)                 # 配了 ARK_OTEL_ENDPOINT 才生效，否则 no-op
    orchestrator = build_orchestrator(settings)
    _ready = True
    yield
    _ready = False


app = FastAPI(title="ArkNarrator 干员对话", version="0.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins.strip() == "*"
    else [o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"], allow_headers=["*"],
)


_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
}


@app.middleware("http")
async def _observe(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    # 请求体大小上限（防超大 payload 打爆）
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_body_bytes:
        return PlainTextResponse("请求体过大", status_code=413)
    start = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        METRICS.inc("ark_requests_total", {"path": request.url.path, "status": "500"})
        logger.exception("rid=%s %s %s 未捕获异常", rid, request.method, request.url.path)
        raise
    dur = time.perf_counter() - start
    METRICS.inc("ark_requests_total", {"path": request.url.path, "status": str(resp.status_code)})
    METRICS.observe("ark_request_seconds", dur, {"path": request.url.path})
    resp.headers["X-Request-ID"] = rid
    resp.headers["X-Served-By"] = _INSTANCE          # 哪个实例处理的（看 LB 是否分散）
    for k, v in _SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    logger.info("rid=%s %s %s %d %.3fs", rid, request.method,
                request.url.path, resp.status_code, dur)
    return resp


async def require_auth(request: Request):
    """none / apikey(X-API-Key) / jwt(Bearer，验签后取 player_id) 三模式。"""
    mode = settings.effective_auth_mode
    if mode == "none":
        return
    if mode == "apikey":
        if request.headers.get("x-api-key") != settings.api_auth_key:
            METRICS.inc("ark_auth_fail_total")
            raise HTTPException(401, "缺少或错误的 API key")
        return
    if mode == "jwt":
        from app.auth import verify_jwt
        authz = request.headers.get("authorization", "")
        token = authz[7:].strip() if authz[:7].lower() == "bearer " else ""
        claims = verify_jwt(token, settings.jwt_secret) if token else None
        if not claims:
            METRICS.inc("ark_auth_fail_total")
            raise HTTPException(401, "无效或过期的令牌")
        pid = claims.get("player_id") or claims.get("sub")
        if pid:
            request.state.player_id = str(pid)   # 信任签发的玩家身份（优先于 body）
        return


async def _bounded_respond(session_id: str, user_id: str, character: str,
                           message: str, history: list[dict]) -> Reply:
    """并发上限 + 超时：满了 429，超时 504，避免单模型被打爆/卡死。"""
    try:
        await asyncio.wait_for(_sema.acquire(), timeout=0.05)
    except asyncio.TimeoutError:
        METRICS.inc("ark_overloaded_total")
        raise HTTPException(429, "服务繁忙，请稍后再试") from None
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(orchestrator.respond, session_id, user_id,
                              character, message, history),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        METRICS.inc("ark_timeout_total")
        raise HTTPException(504, "生成超时，请重试") from None
    finally:
        _sema.release()
    # 在线质量监控：回复量(按角色) / 语气档位分布 / 拦截率
    METRICS.inc("ark_replies_total", {"character": character})
    reg = reply.meta.get("register") if isinstance(reply.meta, dict) else None
    if reg:
        METRICS.inc("ark_register_total", {"register": reg})
    if reply.blocked:
        METRICS.inc("ark_blocked_total", {"category": reply.category})
    return reply


class Turn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    player_id: str | None = Field(
        None, description="游戏侧玩家ID（强烈建议传）。长期记忆/会话按它隔离；不传则退回按来源IP")
    character: str = Field(..., description="干员名，例如 '阿米娅'")
    message: str = Field(..., description="玩家本轮发言")
    session_id: str | None = Field(
        None, description="会话线程ID。不传则默认每个(玩家×干员)一条会话")
    history: list[Turn] = Field(
        default_factory=list, description="可选。不传则服务端用 store 托管历史（薄客户端推荐）")


def _ids(req: ChatRequest, request: Request) -> tuple[str, str]:
    # 优先用 JWT 验签后的 player_id（最可信）→ 其次 body 的 player_id → 最后来源 IP
    user_id = (getattr(request.state, "player_id", None)
               or req.player_id or (request.client.host if request.client else "anon"))
    # 默认每个 玩家×干员 一条会话，历史/记忆自然隔离
    session_id = req.session_id or f"{user_id}:{req.character}"
    return session_id, user_id


@app.get("/livez")
async def livez():
    """存活探针：进程在就行（LB 用）。"""
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    """就绪探针：编排已装配才算 ready（LB 据此放流量）。"""
    if not _ready or orchestrator is None:
        raise HTTPException(503, "not ready")
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus 抓取端点。"""
    return METRICS.render()


@app.get("/health")
async def health():
    if orchestrator is None:
        raise HTTPException(503, "未就绪")
    return {"status": "ok", "model": orchestrator._backend.label,
            "characters": len(orchestrator.characters)}


@app.get("/v1/characters")
@app.get("/characters")
async def characters():
    if orchestrator is None:
        raise HTTPException(503, "未就绪")
    return {
        "characters": [
            {"name": c.name, "codename": c.codename, "faction": c.faction}
            for c in orchestrator.characters.values()
        ]
    }


def _history_arg(req: ChatRequest):
    # 调用方传了历史就用它；否则 None → 服务端用 store 托管该会话历史
    return [{"role": t.role, "content": t.content} for t in req.history] or None


@app.post("/v1/chat", dependencies=[Depends(require_auth)])
@app.post("/chat", dependencies=[Depends(require_auth)])
async def chat(req: ChatRequest, request: Request):
    if not _ready or orchestrator is None:
        raise HTTPException(503, "未就绪")
    if req.character not in orchestrator.characters:
        raise HTTPException(400, f"未知干员：{req.character}")
    session_id, user_id = _ids(req, request)
    reply = await _bounded_respond(session_id, user_id, req.character, req.message,
                                   _history_arg(req))
    return {
        "character": reply.character, "response": reply.text,
        "blocked": reply.blocked, "category": reply.category,
        "ai_label": reply.ai_label, "session_id": session_id,
        "request_id": uuid.uuid4().hex[:12],
    }


@app.post("/v1/stream", dependencies=[Depends(require_auth)])
@app.post("/stream", dependencies=[Depends(require_auth)])
async def stream(req: ChatRequest, request: Request):
    if not _ready or orchestrator is None:
        raise HTTPException(503, "未就绪")
    if req.character not in orchestrator.characters:
        raise HTTPException(400, f"未知干员：{req.character}")
    session_id, user_id = _ids(req, request)

    # 先完整生成 + 审核（守出口），再把安全文本逐字推送
    reply = await _bounded_respond(session_id, user_id, req.character, req.message,
                                   _history_arg(req))

    async def gen():
        for ch in reply.text:
            yield {"data": json.dumps({"token": ch}, ensure_ascii=False)}
            await asyncio.sleep(0)
        yield {"data": json.dumps({
            "done": True, "blocked": reply.blocked,
            "category": reply.category, "ai_label": reply.ai_label,
            "session_id": session_id,
        }, ensure_ascii=False)}

    return EventSourceResponse(gen())


@app.get("/", response_class=HTMLResponse)
async def demo():
    return DEMO_HTML


DEMO_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArkNarrator · 干员对话</title>
<style>
  :root{ --ark:#e8a838; --bg:#0f1117; --card:#171a23; --line:#2a2f3a; --text:#e6e6e6; --muted:#8b93a3; }
  *{ box-sizing:border-box; margin:0; padding:0; }
  body{ background:var(--bg); color:var(--text); font-family:"Noto Sans SC",system-ui,sans-serif;
        display:flex; flex-direction:column; height:100vh; }
  header{ background:var(--card); border-bottom:1px solid var(--line); padding:10px 18px;
          display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  header h1{ font-size:1rem; color:var(--ark); letter-spacing:.5px; }
  header .sub{ font-size:.72rem; color:var(--muted); }
  select,button{ border-radius:8px; border:1px solid var(--line); background:#0f3460;
                 color:var(--text); padding:6px 12px; font-size:.85rem; }
  button{ background:var(--ark); color:#13151c; font-weight:700; border:none; cursor:pointer; }
  button:hover{ filter:brightness(1.08); } button:disabled{ opacity:.5; cursor:default; }
  #banner{ font-size:.72rem; color:var(--muted); background:#13161e; border-bottom:1px solid var(--line);
           padding:5px 18px; }
  #chat{ flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:12px; }
  .msg{ max-width:74%; padding:10px 14px; border-radius:14px; line-height:1.65; white-space:pre-wrap; font-size:.92rem; }
  .user{ background:#0f3460; align-self:flex-end; border-bottom-right-radius:3px; }
  .assistant{ background:#1b2330; align-self:flex-start; border-bottom-left-radius:3px; border:1px solid var(--line); }
  .assistant .who{ font-size:.72rem; color:var(--ark); margin-bottom:4px; font-weight:700; }
  .assistant.blocked{ border-color:#8a5a2b; }
  .tag{ font-size:.62rem; color:var(--muted); margin-top:5px; }
  footer{ background:var(--card); border-top:1px solid var(--line); padding:10px 16px; display:flex; gap:8px; }
  footer input{ flex:1; border-radius:8px; border:1px solid var(--line); background:#0f1117; color:var(--text); padding:9px 12px; }
  #status{ font-size:.7rem; color:var(--muted); padding:3px 18px; background:var(--card); }
</style>
</head>
<body>
<header>
  <h1>⚔️ ArkNarrator</h1>
  <span class="sub">明日方舟干员对话 · 本地推理 + 纵深安全护栏</span>
  <label style="margin-left:auto">干员
    <select id="char"></select>
  </label>
  <button onclick="clearHistory()">清空</button>
</header>
<div id="banner">⚠️ 本回复由 AI 生成，仅供娱乐；角色与世界观版权归鹰角网络所有。请勿据此做现实决策。</div>
<div id="status">正在连接…</div>
<div id="chat"></div>
<footer>
  <input id="inp" type="text" placeholder="对干员说点什么…" onkeydown="if(event.key==='Enter')send()">
  <button id="btn" onclick="send()">发送</button>
</footer>
<script>
const chat=document.getElementById('chat'), inp=document.getElementById('inp'),
      btn=document.getElementById('btn'), sel=document.getElementById('char'),
      status=document.getElementById('status');
let history=[], sessionId=null;

fetch('/characters').then(r=>r.json()).then(d=>{
  d.characters.forEach(c=>{ const o=document.createElement('option');
    o.value=c.name; o.textContent=c.codename?`${c.name}（${c.codename}）`:c.name; sel.appendChild(o); });
});
fetch('/health').then(r=>r.json()).then(d=>{ status.textContent='模型：'+d.model+' · 干员 '+d.characters+' 名'; })
               .catch(()=>{ status.textContent='服务未就绪'; });

function clearHistory(){ history=[]; sessionId=null; chat.innerHTML=''; }
function addMsg(role,text,who){
  const div=document.createElement('div'); div.className='msg '+role;
  if(role==='assistant'){ div.innerHTML='<div class="who">'+who+'</div>'; }
  const p=document.createElement('div'); p.textContent=text; div.appendChild(p);
  chat.appendChild(div); chat.scrollTop=chat.scrollHeight; return {div,p};
}
async function send(){
  const msg=inp.value.trim(); if(!msg) return;
  const char=sel.value; inp.value=''; btn.disabled=true;
  addMsg('user',msg);
  const {div,p}=addMsg('assistant','',char); let full='';
  try{
    const resp=await fetch('/stream',{ method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({character:char,message:msg,history:history,session_id:sessionId}) });
    const reader=resp.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let idx;
      while((idx=buf.indexOf('\\n\\n'))>=0){
        const line=buf.slice(0,idx); buf=buf.slice(idx+2);
        if(line.startsWith('data:')){
          const d=JSON.parse(line.slice(5).trim());
          if(d.done){ sessionId=d.session_id;
            if(d.blocked){ div.classList.add('blocked');
              const t=document.createElement('div'); t.className='tag';
              t.textContent='⚠ 已由安全护栏处理（'+d.category+'）'; div.appendChild(t); }
          } else if(d.token){ full+=d.token; p.textContent=full; chat.scrollTop=chat.scrollHeight; }
        }
      }
    }
  }catch(e){ p.textContent='[错误：'+e.message+']'; }
  history.push({role:'user',content:msg},{role:'assistant',content:full});
  btn.disabled=false; inp.focus();
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
