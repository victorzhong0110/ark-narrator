"""OperatorIR —— 干员的标准中间表示（数据契约）。

任何数据源都先转成它，下游（建知识库、角色卡脚手架、RAG、语气档位）只认它。
公司接入时，把自家数据映射到这几个字段即可，不关心引擎内部。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VoiceLine:
    """一条语音台词（第一人称、自成一句——角色声音的金标）。"""

    text: str
    title: str = ""          # 语音条目名，如「任命助理」


@dataclass(frozen=True)
class StoryLine:
    """剧情里的一句台词（带语境：对谁说、在回应什么、哪个场景）。"""

    text: str
    interlocutor: str = ""   # 这句话说给谁
    prev: str = ""           # 她在回应的上一句（语境）
    scene: str = ""          # 场景/章节名


@dataclass(frozen=True)
class OperatorIR:
    """一个角色的全部接入数据。"""

    name: str
    codename: str = ""                          # 外文/代号，仅展示
    faction: str = ""                           # 所属阵营/组织
    profile_facts: tuple[str, ...] = field(default_factory=tuple)   # 档案事实（第三人称）
    voice_lines: tuple[VoiceLine, ...] = field(default_factory=tuple)
    story_lines: tuple[StoryLine, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (f"{self.name}（{self.codename}）| 阵营={self.faction} | "
                f"档案 {len(self.profile_facts)} 条 | 语音 {len(self.voice_lines)} 条 | "
                f"剧情 {len(self.story_lines)} 条")

    # ---- 序列化：便于公司把内部数据导出成本 schema ----
    def to_dict(self) -> dict:
        return {
            "name": self.name, "codename": self.codename, "faction": self.faction,
            "profile_facts": list(self.profile_facts),
            "voice_lines": [{"title": v.title, "text": v.text} for v in self.voice_lines],
            "story_lines": [
                {"text": s.text, "interlocutor": s.interlocutor,
                 "prev": s.prev, "scene": s.scene}
                for s in self.story_lines
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OperatorIR":
        return cls(
            name=str(d["name"]).strip(),
            codename=str(d.get("codename", "")).strip(),
            faction=str(d.get("faction", "")).strip(),
            profile_facts=tuple(str(x).strip() for x in d.get("profile_facts", []) if str(x).strip()),
            voice_lines=tuple(
                VoiceLine(text=str(v.get("text", "")).strip(), title=str(v.get("title", "")).strip())
                for v in d.get("voice_lines", []) if str(v.get("text", "")).strip()
            ),
            story_lines=tuple(
                StoryLine(
                    text=str(s.get("text", "")).strip(),
                    interlocutor=str(s.get("interlocutor", "")).strip(),
                    prev=str(s.get("prev", "")).strip(),
                    scene=str(s.get("scene", "")).strip(),
                )
                for s in d.get("story_lines", []) if str(s.get("text", "")).strip()
            ),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "OperatorIR":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
