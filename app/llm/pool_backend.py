"""PooledAPIBackend：面向「自注册节点池」的后端——分布式系统的客户端侧负载均衡。

app 层从 Store（Redis）读取活节点列表（节点靠心跳自注册、TTL 过期自动下线），每次请求
轮询挑一个节点调用，失败自动转移到下一个。新 Mac 一注册就被发现，无需改 nginx、无需重启。
"""

from __future__ import annotations

import ipaddress
import logging
import time
from typing import Callable, Iterator

from app.llm.api_backend import APIBackend
from app.llm.base import Message
from app.store.base import Store

logger = logging.getLogger(__name__)

_CB_THRESHOLD = 3      # 连续失败几次后熔断该节点
_CB_COOLDOWN = 10.0    # 熔断冷却秒数（期内不再试，省连接超时）


def _addr_ok(addr: str, allowlist: list[str]) -> bool:
    """校验节点地址防 SSRF：始终拒绝云元数据/链路本地/组播/未指定段；配了白名单则只放行白名单前缀。"""
    host = addr.rpartition(":")[0] if ":" in addr else addr
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
            return False     # 169.254.169.254(云元数据)/0.0.0.0/组播 等一律拒
    except ValueError:
        pass                 # 主机名（非 IP），交给白名单判断
    if allowlist:
        return any(host.startswith(p) for p in allowlist)
    return True


class PooledAPIBackend:
    def __init__(self, store: Store, model: str, *, group: str = "models",
                 api_key: str = "EMPTY", disable_thinking: bool = True,
                 allowlist: list[str] | None = None,
                 client_factory: Callable[[str], object] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.label = f"pool:{model}"
        self._store = store
        self._model = model
        self._group = group
        self._key = api_key or "EMPTY"
        self._allowlist = allowlist or []
        self._disable_thinking = disable_thinking   # 自托管 Qwen 节点默认关思考
        self._client_factory = client_factory       # 测试可注入 addr→client
        self._backends: dict[str, APIBackend] = {}
        self._clock = clock
        self._fails: dict[str, tuple[int, float]] = {}   # addr → (连续失败数, 最近失败时刻)

    def _circuit_open(self, addr: str, now: float) -> bool:
        f = self._fails.get(addr)
        return bool(f and f[0] >= _CB_THRESHOLD and (now - f[1]) < _CB_COOLDOWN)

    def _record(self, addr: str, ok: bool, now: float) -> None:
        if ok:
            self._fails.pop(addr, None)
        else:
            n = self._fails.get(addr, (0, 0.0))[0] + 1
            self._fails[addr] = (n, now)

    def _backend_for(self, addr: str) -> APIBackend:
        if addr not in self._backends:
            client = self._client_factory(addr) if self._client_factory else None
            self._backends[addr] = APIBackend(
                self._model, f"http://{addr}/v1", self._key,
                disable_thinking=self._disable_thinking, client=client)
        return self._backends[addr]

    def _ordered(self) -> list[str]:
        """活节点列表：轮询起点错开；熔断中的节点排到最后（仅在健康节点都失败时才半开试探）。"""
        nodes = [n for n in self._store.live_nodes(self._group) if _addr_ok(n, self._allowlist)]
        if not nodes:
            return []
        i = self._store.incr(f"rr:{self._group}", 1) % len(nodes)
        rotated = nodes[i:] + nodes[:i]
        now = self._clock()
        healthy = [n for n in rotated if not self._circuit_open(n, now)]
        tripped = [n for n in rotated if self._circuit_open(n, now)]
        return healthy + tripped     # 健康优先，熔断的留作最后兜底（半开）

    def generate(self, system: str, messages: list[Message], *,
                 max_tokens: int = 320, temperature: float = 0.7) -> str:
        nodes = self._ordered()
        if not nodes:
            raise RuntimeError("集群中没有活节点（检查 Mac 节点是否在心跳注册）")
        last: Exception | None = None
        for addr in nodes:
            try:
                out = self._backend_for(addr).generate(
                    system, messages, max_tokens=max_tokens, temperature=temperature)
                self._record(addr, True, self._clock())
                return out
            except Exception as exc:  # noqa: BLE001 — 故障转移到下一节点
                self._record(addr, False, self._clock())
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
                self._record(addr, True, self._clock())
                return
            except Exception as exc:  # noqa: BLE001
                self._record(addr, False, self._clock())
                logger.warning("节点 %s 流式失败，转移：%s", addr, exc)
                last = exc
        raise RuntimeError(f"所有节点都失败：{last}")
