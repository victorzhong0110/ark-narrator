"""数据源适配器：把不同来源映射成统一的 OperatorIR。

- PRTSSource：参考实现，读公开的明日方舟游戏数据（由 fetch_operator.py / mine_stories.py
  抓取并缓存到 data/raw/）。
- GenericJSONSource：读「符合 OperatorIR schema 的 JSON」——公司把内部数据导出成这个
  schema 即可接入（最低接入成本，无需写代码）。

公司也可以直接实现 OperatorSource 协议对接自家数据库/API，下游全部复用。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from app.ingest.ir import OperatorIR, StoryLine, VoiceLine

logger = logging.getLogger(__name__)


class OperatorSource(Protocol):
    def fetch(self, identifier: str) -> OperatorIR:
        """按名字/ID 取一个角色的全部接入数据。"""
        ...


class GenericJSONSource:
    """读公司导出的 JSON（OperatorIR schema）。最低接入成本：导数据，不写代码。

    支持两种布局：
    - 单文件：path 指向一个角色的 JSON。
    - 目录：path 是目录，按 <name>.json 取文件。
    """

    def __init__(self, path: Path):
        self._path = path

    def fetch(self, identifier: str) -> OperatorIR:
        target = self._path
        if self._path.is_dir():
            target = self._path / f"{identifier}.json"
        if not target.exists():
            raise FileNotFoundError(f"找不到角色数据文件：{target}")
        return OperatorIR.load(target)


class PRTSSource:
    """明日方舟公开数据适配器（读 data/raw/ 下的缓存）。"""

    def __init__(self, raw_dir: Path):
        self._raw = raw_dir

    def fetch(self, identifier: str) -> OperatorIR:
        op_path = self._raw / f"operator_{identifier}.json"
        st_path = self._raw / f"stories_{identifier}.json"
        if not op_path.exists():
            raise FileNotFoundError(
                f"未找到 {op_path}；请先运行 scripts/fetch_operator.py {identifier}"
            )
        op = json.loads(op_path.read_text(encoding="utf-8"))
        story_lines: list[StoryLine] = []
        if st_path.exists():
            st = json.loads(st_path.read_text(encoding="utf-8"))
            for ln in st.get("lines", []):
                story_lines.append(StoryLine(
                    text=str(ln.get("text", "")).strip(),
                    interlocutor=str(ln.get("interlocutor", "")).strip(),
                    prev=str(ln.get("prev", "")).strip(),
                    scene=str(ln.get("act", "")).strip(),
                ))
        return OperatorIR(
            name=op.get("name", identifier),
            codename="",                      # 公开数据无英文代号，留待人工补
            faction="",                       # 同上（card 脚手架里标 TODO）
            profile_facts=tuple(
                s.get("text", "").strip() for s in op.get("stories", []) if s.get("text", "").strip()
            ),
            voice_lines=tuple(
                VoiceLine(text=v.get("text", "").strip(), title=v.get("title", "").strip())
                for v in op.get("voices", []) if v.get("text", "").strip()
            ),
            story_lines=tuple(s for s in story_lines if s.text),
        )


def get_source(kind: str, *, raw_dir: Path | None = None, data_path: Path | None = None) -> OperatorSource:
    kind = kind.lower()
    if kind == "prts":
        return PRTSSource(raw_dir or Path("data/raw"))
    if kind == "generic":
        if data_path is None:
            raise ValueError("generic 源需要 --data 指向 JSON 文件或目录")
        return GenericJSONSource(data_path)
    raise ValueError(f"未知数据源：{kind}（应为 prts 或 generic）")
