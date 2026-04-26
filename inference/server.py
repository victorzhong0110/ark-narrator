"""
ArkNarrator FastAPI inference server.

Endpoints
---------
GET  /health              — liveness check
GET  /characters          — list available roleplay characters
POST /chat                — single-turn roleplay (non-streaming)
POST /stream              — SSE streaming roleplay
GET  /                    — interactive demo UI (HTML)

Start
-----
  # default: Qwen adapter
  uvicorn inference.server:app --reload

  # use Gemma adapter
  MODEL_KEY=gemma uvicorn inference.server:app --reload
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from inference.engine import ArkNarratorEngine, CHARACTER_CARDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

engine: ArkNarratorEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    model_key = os.getenv("MODEL_KEY", "qwen")
    engine = ArkNarratorEngine(model_key=model_key)
    yield


app = FastAPI(title="ArkNarrator API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str           # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    character: str = Field(..., description="干员名，例如 '能天使'")
    message: str = Field(..., description="用户本轮发言")
    history: list[Turn] = Field(default_factory=list, description="历史对话轮次")
    max_tokens: int = Field(300, ge=50, le=1000)
    temperature: float = Field(0.8, ge=0.1, le=2.0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if engine is None:
        raise HTTPException(503, "Engine not loaded")
    return {"status": "ok", "model": engine.label}


@app.get("/characters")
async def characters():
    return {"characters": list(CHARACTER_CARDS.keys())}


@app.post("/chat")
async def chat(req: ChatRequest):
    if engine is None:
        raise HTTPException(503, "Engine not loaded")
    if req.character not in CHARACTER_CARDS:
        raise HTTPException(400, f"Unknown character: {req.character}")

    history = [{"role": t.role, "content": t.content} for t in req.history]
    output = await asyncio.to_thread(
        engine.generate,
        req.character, history, req.message,
        req.max_tokens, req.temperature,
    )
    return {"character": req.character, "response": output}


@app.post("/stream")
async def stream(req: ChatRequest):
    if engine is None:
        raise HTTPException(503, "Engine not loaded")
    if req.character not in CHARACTER_CARDS:
        raise HTTPException(400, f"Unknown character: {req.character}")

    history = [{"role": t.role, "content": t.content} for t in req.history]

    async def generator():
        async for chunk in engine.stream(
            req.character, history, req.message,
            req.max_tokens, req.temperature,
        ):
            yield {"data": json.dumps({"token": chunk}, ensure_ascii=False)}
        yield {"data": json.dumps({"token": "[DONE]"})}

    return EventSourceResponse(generator())


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------

DEMO_HTML = """\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ArkNarrator · 干员对话 Demo</title>
<style>
  :root { --ark: #e8a838; --bg: #1a1a2e; --card: #16213e; --text: #e0e0e0; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: "Noto Sans SC", sans-serif;
         display: flex; flex-direction: column; height: 100vh; }
  header { background: var(--card); border-bottom: 2px solid var(--ark);
           padding: 12px 20px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.1rem; color: var(--ark); }
  select, button, input { border-radius: 6px; border: 1px solid #444; background: #0f3460;
                           color: var(--text); padding: 6px 12px; font-size: 0.9rem; }
  select:focus, input:focus { outline: 2px solid var(--ark); }
  #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 72%; padding: 10px 14px; border-radius: 12px; line-height: 1.6; white-space: pre-wrap; }
  .user { background: #0f3460; align-self: flex-end; border-bottom-right-radius: 2px; }
  .assistant { background: #1b4332; align-self: flex-start; border-bottom-left-radius: 2px; }
  .assistant .speaker { font-size: 0.75rem; color: var(--ark); margin-bottom: 4px; font-weight: bold; }
  footer { background: var(--card); border-top: 1px solid #333; padding: 10px 16px;
           display: flex; gap: 8px; }
  footer input { flex: 1; }
  button { background: var(--ark); color: #1a1a2e; font-weight: bold; border: none; cursor: pointer; }
  button:hover { filter: brightness(1.1); }
  button:disabled { opacity: 0.5; cursor: default; }
  #status { font-size: 0.75rem; color: #888; padding: 2px 20px; background: var(--card); }
</style>
</head>
<body>
<header>
  <h1>⚔️ ArkNarrator</h1>
  <label>干员：<select id="char"></select></label>
  <label>温度：<input type="range" id="temp" min="0.1" max="1.5" step="0.1" value="0.8"
                      style="width:80px" oninput="document.getElementById('tempVal').textContent=this.value">
    <span id="tempVal">0.8</span></label>
  <button onclick="clearHistory()">清空对话</button>
</header>
<div id="status">正在连接…</div>
<div id="chat"></div>
<footer>
  <input id="inp" type="text" placeholder="输入消息…" onkeydown="if(event.key==='Enter')send()">
  <button id="btn" onclick="send()">发送</button>
</footer>
<script>
const chat = document.getElementById('chat');
const inp  = document.getElementById('inp');
const btn  = document.getElementById('btn');
const sel  = document.getElementById('char');
const status = document.getElementById('status');
let history = [];

fetch('/characters').then(r=>r.json()).then(d=>{
  d.characters.forEach(c=>{ const o=document.createElement('option'); o.value=o.textContent=c; sel.appendChild(o); });
});
fetch('/health').then(r=>r.json()).then(d=>{ status.textContent = '模型：' + d.model; })
               .catch(()=>{ status.textContent='服务未就绪'; });

function clearHistory(){ history=[]; chat.innerHTML=''; }

function addMsg(role, text, speaker){
  const div=document.createElement('div');
  div.className='msg '+role;
  if(role==='assistant'){ div.innerHTML='<div class="speaker">'+speaker+'</div>'; }
  const p=document.createElement('div');
  p.textContent=text;
  div.appendChild(p);
  chat.appendChild(div);
  chat.scrollTop=chat.scrollHeight;
  return p;
}

async function send(){
  const msg=inp.value.trim(); if(!msg) return;
  const char=sel.value;
  inp.value=''; btn.disabled=true;
  addMsg('user', msg);

  const p=addMsg('assistant','', char);
  let full='';

  try {
    const resp=await fetch('/stream',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ character:char, message:msg,
                             history:history,
                             temperature:parseFloat(document.getElementById('temp').value) })
    });
    const reader=resp.body.getReader(); const dec=new TextDecoder();
    let buf='';
    while(true){
      const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let idx;
      while((idx=buf.indexOf('\\n\\n'))>=0){
        const line=buf.slice(0,idx); buf=buf.slice(idx+2);
        if(line.startsWith('data:')){
          const payload=JSON.parse(line.slice(5).trim());
          if(payload.token==='[DONE]') break;
          full+=payload.token; p.textContent=full;
          chat.scrollTop=chat.scrollHeight;
        }
      }
    }
  } catch(e){ p.textContent='[错误：'+e.message+']'; }

  history.push({role:'user',content:msg},{role:'assistant',content:full});
  btn.disabled=false; inp.focus();
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def demo():
    return DEMO_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "inference.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
