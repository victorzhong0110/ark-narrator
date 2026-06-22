"""PooledAPIBackend：面向「自注册节点池」的后端——分布式系统的客户端侧负载均衡。

app 层从 Store（Redis）读取活节点列表（节点靠心跳自注册、TTL 过期自动下线），每次请求
轮询挑一个节点调用，失败自动转移到下一个。新 Mac 一注册就被发现，无需改 nginx、无需重启。
"""

from __future__ import annotations

import logging
from typing import Callable, Iterator

from app.llm.api_backend import APIBackend
from app.llm.base import Message
from app.store.base import Store

logger = logging.getLogger(__name__)


class PooledAPIBackend:
    def __init__(self, store: Store, model: str, *, group: str = "models",
                 api_key: str = "EMPTY", client_factory: Callable[[str], object] | None = None):
        self.label = f"pool:{model}"
        self._store = store
        self._model = model
        self._group = group
        self._key = api_key or "EMPTY"
        self._client_factory = client_factory      # 测试可注入 addr→client
        self._backends: dict[str, APIBackend] = {}

    def _backend_for(self, addr: str) -> APIBackend:
        if addr not in self._backends:
            client = self._client_factory(addr) if self._client_factory else None
            self._backends[addr] = APIBackend(
                self._model, f"http://{addr}/v1", self._key, client=client)
        return self._backends[addr]

    def _ordered(self) -> list[str]:
        """活节点列表，轮询起点错开（简单的 round-robin）。"""
        nodes = self._store.live_nodes(self._group)
        if not nodes:
            return []
        # 用计数器旋转起点；存于 store 让多 app 实例也大致均摊
        i = self._store.incr(f"rr:{self._group}", 1) % len(nodes)
        return nodes[i:] + nodes[:i]

    def generate(self, system: str, messages: list[Message], *,
                 max_tokens: int = 320, temperature: float = 0.7) -> str:
        nodes = self._ordered()
        if not nodes:
            raise RuntimeError("集群中没有活节点（检查 Mac 节点是否在心跳注册）")
        last: Exception | None = None
        for addr in nodes:
            try:
                return self._backend_for(addr).generate(
                    system, messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:  # noqa: BLE001 — 故障转移到下一节点
                logger.warning("节点 %s 失败，转移：%s", addr, exc)
                last = exc
        raise RuntimeError(f"所有节点都失败：{last}")

    def stream(self, system: str, messages: list[Message], *,
               max_tokens: int = 320, temperature: float = 0.7) -> Iterator[str]:
        nodes = self._ordered()
        if not nodes:
            raise RuntimeError("集群中没有活节点")
        last: Exception | None = None
        for addr in nodes:
            try:
                yield from self._backend_for(addr).stream(
                    system, messages, max_tokens=max_tokens, temperature=temperature)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("节点 %s 流式失败，转移：%s", addr, exc)
                last = exc
        raise RuntimeError(f"所有节点都失败：{last}")
