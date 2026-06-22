"""评分维度与裁判。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from app.guard import rules
from app.llm.base import LLMBackend

# 五个维度（沿用 ArkNarrator 研究侧 G-Eval）
DIMENSIONS: tuple[str, ...] = ("voice", "speech", "lore", "consistency", "depth")
DIM_LABEL = {
    "voice": "角色声音", "speech": "说话方式", "lore": "世界观",
    "consistency": "一致性", "depth": "深度",
}

# 明显的现实/现代穿越词（lore 扣分用，粗筛）
_ANACHRONISM = re.compile(r"(手机|电脑|互联网|网络|可乐|汉堡|微信|拉丁文|地球|美国|中国)")


@dataclass(frozen=True)
class Score:
    dims: dict[str, float]
    note: str = ""
    hard_fail: bool = False     # 出戏/泄露等灾难性失败：无论均分多少都该判 FAIL

    def avg(self) -> float:
        vals = [v for v in self.dims.values() if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0.0


class Judge(Protocol):
    def score(self, character: str, card_text: str, user: str, response: str) -> Score:
        ...


class KeywordJudge:
    """确定性裁判：捕捉灾难性失败，适合做 CI 硬门槛（无需 API）。

    出戏/泄露=一致性崩；空答/过短=声音弱；穿越词=世界观扣分。不评细腻保真（那交给 LLMJudge）。
    """

    def __init__(self, system_markers: tuple[str, ...] = ()):
        self._markers = system_markers

    def score(self, character: str, card_text: str, user: str, response: str) -> Score:
        r = response.strip()
        role_break = not rules.detect_role_break(r).allowed
        leak = not rules.detect_prompt_leak(r, self._markers).allowed
        anach = len(_ANACHRONISM.findall(r))
        n = len(r)

        consistency = 2.0 if (role_break or leak) else 9.0
        voice = 1.0 if n < 4 else (6.0 if n < 15 else 8.0)
        speech = voice
        lore = max(2.0, 9.0 - 2.0 * anach)
        depth = 3.0 if n < 20 else (6.0 if n < 60 else 8.0)
        note = []
        if role_break:
            note.append("出戏")
        if leak:
            note.append("泄露")
        if anach:
            note.append(f"穿越词x{anach}")
        return Score(
            {"voice": voice, "speech": speech, "lore": lore,
             "consistency": consistency, "depth": depth},
            note="；".join(note),
            hard_fail=role_break or leak,
        )


_GEVAL = """\
你是资深玩家，评估一段角色扮演输出。角色档案：
{card}
用户输入：{user}
模型输出：{output}
请对以下维度各打 1-10 整数分，只输出 JSON，不要解释：
{{"voice": <角色声音>, "speech": <说话方式>, "lore": <世界观准确>, \
"consistency": <内部一致>, "depth": <角色深度>}}"""


class LLMJudge:
    """用裁判模型按 G-Eval 维度打分。解析失败给中性分，不崩。"""

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def score(self, character: str, card_text: str, user: str, response: str) -> Score:
        prompt = _GEVAL.format(card=card_text, user=user, output=response)
        try:
            raw = self._backend.generate("", [{"role": "user", "content": prompt}],
                                         max_tokens=200, temperature=0.1)
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            data = json.loads(m.group(0) if m else clean)
            dims = {d: float(data.get(d, 5)) for d in DIMENSIONS}
            return Score(dims)
        except Exception as exc:  # noqa: BLE001 — 裁判失败不应中断评测
            return Score(dict.fromkeys(DIMENSIONS, 5.0), note=f"judge_error:{exc}")
