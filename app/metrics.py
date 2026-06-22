"""极简指标（Prometheus 文本格式，零依赖）。

只做计数器 + 直方图，够覆盖「请求量 / 延迟 / 拦截率 / 错误率」这些运营核心信号，
经 /metrics 暴露给 Prometheus 抓取。线程安全。
"""

from __future__ import annotations

import threading
from collections import defaultdict

_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)

Labels = tuple[tuple[str, str], ...]


def _key(labels: dict[str, str] | None) -> Labels:
    return tuple(sorted((labels or {}).items()))


def _fmt_labels(labels: Labels, extra: tuple[str, str] | None = None) -> str:
    items = list(labels) + ([extra] if extra else [])
    if not items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in items)
    return "{" + inner + "}"


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, Labels], float] = defaultdict(float)
        self._h_sum: dict[tuple[str, Labels], float] = defaultdict(float)
        self._h_count: dict[tuple[str, Labels], float] = defaultdict(float)
        self._h_bucket: dict[tuple[str, Labels], dict[float, float]] = defaultdict(
            lambda: dict.fromkeys(_BUCKETS, 0.0)
        )

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1.0) -> None:
        with self._lock:
            self._counters[(name, _key(labels))] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        k = (name, _key(labels))
        with self._lock:
            self._h_sum[k] += value
            self._h_count[k] += 1
            for b in _BUCKETS:
                if value <= b:
                    self._h_bucket[k][b] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            seen_types: set[str] = set()
            for (name, labels), val in sorted(self._counters.items()):
                if name not in seen_types:
                    lines.append(f"# TYPE {name} counter")
                    seen_types.add(name)
                lines.append(f"{name}{_fmt_labels(labels)} {val:g}")
            for (name, labels), cnt in sorted(self._h_count.items()):
                if name not in seen_types:
                    lines.append(f"# TYPE {name} histogram")
                    seen_types.add(name)
                cumulative = 0.0
                for b in _BUCKETS:
                    cumulative = self._h_bucket[(name, labels)][b]
                    lines.append(f'{name}_bucket{_fmt_labels(labels, ("le", str(b)))} {cumulative:g}')
                lines.append(f'{name}_bucket{_fmt_labels(labels, ("le", "+Inf"))} {cnt:g}')
                lines.append(f"{name}_sum{_fmt_labels(labels)} {self._h_sum[(name, labels)]:g}")
                lines.append(f"{name}_count{_fmt_labels(labels)} {cnt:g}")
        return "\n".join(lines) + "\n"


# 进程级全局实例
METRICS = Metrics()
