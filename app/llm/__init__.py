"""LLM 后端：可插拔。mlx（本地真模型）/ scripted（无需模型，测试与管线演示）。"""

from app.llm.base import LLMBackend, Message
from app.llm.factory import get_backend

__all__ = ["LLMBackend", "Message", "get_backend"]
