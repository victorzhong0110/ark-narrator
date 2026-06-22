"""节点心跳注册器（自包含，模型节点上跑，只依赖 redis）。

把本节点地址按 TTL 心跳写进 Redis 的服务注册表；app 层（PooledAPIBackend）据此自动发现。
退出时主动撤下本节点。键格式与 app/store/redis.py 的 live_nodes 一致（nodes:<group> 有序集合）。

环境变量：
  ARK_REDIS_URL   redis://<app/redis主机>:6379/0
  ARK_NODE_ADDR   本节点对外地址 host:port（默认自动探测局域网 IP + ARK_NODE_PORT）
  ARK_NODE_PORT   默认 8080
  ARK_NODE_GROUP  默认 models
  ARK_NODE_TTL    心跳存活秒数，默认 15（每 1/3 周期续一次）
"""

from __future__ import annotations

import os
import socket
import time


def _detect_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:  # noqa: BLE001
        return "127.0.0.1"


def main() -> None:
    import redis

    url = os.getenv("ARK_REDIS_URL", "redis://localhost:6379/0")
    group = os.getenv("ARK_NODE_GROUP", "models")
    port = os.getenv("ARK_NODE_PORT", "8080")
    addr = os.getenv("ARK_NODE_ADDR") or f"{_detect_ip()}:{port}"
    ttl = float(os.getenv("ARK_NODE_TTL", "15"))
    key = f"nodes:{group}"

    r = redis.Redis.from_url(url)
    print(f"[registrar] 注册 {addr} → {url}（组 {group}，TTL {ttl}s）。Ctrl-C 撤下。")
    try:
        while True:
            r.zadd(key, {addr: time.time() + ttl})
            time.sleep(max(1.0, ttl / 3))
    except KeyboardInterrupt:
        pass
    finally:
        r.zrem(key, addr)       # 主动下线
        print(f"\n[registrar] 已撤下 {addr}")


if __name__ == "__main__":
    main()
