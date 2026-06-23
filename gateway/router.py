"""路由：每个请求在「内部机队」和「外部 API」之间选上游，灵活调度。

策略：
- 按模型名：路由表把 model 映射到目标（pool / external）。
- 默认「内部优先、外部兜底」：先内部，内部无健康节点/失败再外部（省钱抗故障）。
- 百分比分流：可配 X% 走外部优先（A/B、限内部负载、难任务上外部强模型）。
返回「按序尝试的目标列表」，网关依次尝试 = 跨上游故障转移。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Router:
    targets: dict[str, object]              # name -> backend(实现 generate/stream)
    table: dict[str, str] = field(default_factory=dict)   # model -> target name
    default: str = "pool"                   # 未命中表/auto 时的主目标
    fallback: str | None = None             # 兜底目标（通常 external）
    split: float = 0.0                      # [0,1] 走外部优先的比例（仅默认策略）
    _n: int = 0

    def route(self, model: str) -> list[tuple[str, object]]:
        explicit = self.table.get(model)
        if explicit and explicit != "auto" and explicit in self.targets:
            order = [explicit]
            if self.fallback and self.fallback != explicit:
                order.append(self.fallback)         # 显式目标也给个兜底
        else:
            self._n += 1
            external_first = (self.split > 0 and self.fallback
                              and (self._n % 100) < round(self.split * 100))
            if external_first:
                order = [self.fallback, self.default]
            else:
                order = [self.default]
                if self.fallback and self.fallback != self.default:
                    order.append(self.fallback)
        seen: set[str] = set()
        out: list[tuple[str, object]] = []
        for name in order:
            if name in self.targets and name not in seen:
                seen.add(name)
                out.append((name, self.targets[name]))
        return out
