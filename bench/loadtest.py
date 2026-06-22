"""轻量压测器（pure Python + httpx）：测吞吐 / 延迟分布 / 错误率 / LB 分散。

用法：
  python bench/loadtest.py --url http://localhost:8000/v1/chat -n 2000 -c 50
注：对 scripted 后端压测测的是「服务层管线」（LB/并发/背压）吞吐，不是模型吞吐；
模型吞吐需对真模型节点单独基准（见容量模型）。
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

import httpx


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


async def _run(url: str, n: int, conc: int, character: str) -> dict:
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    status: Counter = Counter()
    served: Counter = Counter()

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def one(i: int):
            payload = {"player_id": f"load-{i % 500}", "character": character,
                       "message": f"压测第{i}条"}
            async with sem:
                t = time.perf_counter()
                try:
                    r = await client.post(url, json=payload)
                    lat.append(time.perf_counter() - t)
                    status[r.status_code] += 1
                    served[r.headers.get("x-served-by", "?")] += 1
                except Exception as exc:  # noqa: BLE001
                    status[f"err:{type(exc).__name__}"] += 1

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0

    ok = status.get(200, 0)
    return {
        "n": n, "concurrency": conc, "wall_s": round(wall, 2),
        "throughput_rps": round(n / wall, 1) if wall else 0,
        "ok": ok, "status": dict(status),
        "p50_ms": round(_pctl(lat, 0.50) * 1000, 1),
        "p95_ms": round(_pctl(lat, 0.95) * 1000, 1),
        "p99_ms": round(_pctl(lat, 0.99) * 1000, 1),
        "served_by": dict(served),
    }


def summarize(res: dict) -> str:
    lines = [
        f"请求 {res['n']} | 并发 {res['concurrency']} | 用时 {res['wall_s']}s",
        f"吞吐 {res['throughput_rps']} req/s | 成功 {res['ok']}/{res['n']}",
        f"延迟 p50 {res['p50_ms']}ms / p95 {res['p95_ms']}ms / p99 {res['p99_ms']}ms",
        f"状态分布 {res['status']}",
        f"LB 分散（各实例处理数）{res['served_by']}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/chat")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("-c", "--concurrency", type=int, default=50)
    ap.add_argument("--character", default="能天使")
    args = ap.parse_args()
    res = asyncio.run(_run(args.url, args.n, args.concurrency, args.character))
    print(summarize(res))


if __name__ == "__main__":
    main()
