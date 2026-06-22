"""世界观 / IP 配置（WorldProfile）。

把「明日方舟 / 泰拉 / 干员」这类 IP 专有词，从引擎里抽出来变成配置——换一个 IP
（接公司自家作品）只改这份配置，安全护栏、角色卡渲染等下游代码都不动。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorldProfile:
    work: str           # 作品名，如「明日方舟」
    universe: str       # 世界/大陆名，如「泰拉」
    role_term: str      # 角色统称，如「干员」
    ip_owner: str       # 版权方，如「鹰角网络」


# 参考实现：明日方舟（公开数据）。换 IP 时替换为公司自己的 WorldProfile。
ARKNIGHTS = WorldProfile(
    work="明日方舟", universe="泰拉", role_term="干员", ip_owner="鹰角网络"
)

DEFAULT_WORLD = ARKNIGHTS


def load_world(path: Path | None) -> WorldProfile:
    """从 data/world.yaml 读 WorldProfile；缺省回退明日方舟。

    YAML 形如：work: 星海纪元 / universe: 银河 / role_term: 探员 / ip_owner: 某某公司
    """
    if path is None or not path.exists():
        return DEFAULT_WORLD
    try:
        import yaml

        d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return WorldProfile(
            work=str(d.get("work", ARKNIGHTS.work)).strip(),
            universe=str(d.get("universe", ARKNIGHTS.universe)).strip(),
            role_term=str(d.get("role_term", ARKNIGHTS.role_term)).strip(),
            ip_owner=str(d.get("ip_owner", ARKNIGHTS.ip_owner)).strip(),
        )
    except Exception:  # noqa: BLE001 — 配置坏了不应中断启动
        return DEFAULT_WORLD
