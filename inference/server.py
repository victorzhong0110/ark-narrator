"""
FastAPI inference server with SSE streaming.
GET  /health
POST /generate      — single response
POST /stream        — SSE streaming response
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

from inference.engine import ArkNarratorEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="ArkNarrator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: ArkNarratorEngine = None


@app.on_event("startup")
async def startup():
    global engine
    engine = ArkNarratorEngine(
        base_model=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        adapter_path=os.getenv("ADAPTER_PATH", "./checkpoints/qwen2_5_ark/final"),
    )


class GenerateRequest(BaseModel):
    instruction: str
    context: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.8


@app.get("/health")
async def health():
    return {"status": "ok", "model": "ArkNarrator-Qwen2.5-7B-LoRA"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    result = engine.generate(
        req.instruction, req.context, req.max_new_tokens, req.temperature
    )
    return {"result": result}


@app.post("/stream")
async def stream(req: GenerateRequest):
    async def event_generator():
        for token in engine.stream(req.instruction, req.context, req.max_new_tokens):
            yield {"data": token}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", 8000)))
