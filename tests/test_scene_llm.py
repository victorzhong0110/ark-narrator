"""LLM 场景判定器：解析标签 + 出错/无法解析时回退启发式。"""

from __future__ import annotations

from app.llm.scripted_backend import ScriptedBackend
from app.registers import Register
from app.scene import HeuristicSceneTagger, LLMSceneTagger


def _backend(reply: str) -> ScriptedBackend:
    return ScriptedBackend(lambda system, msgs: reply)


def test_llm_tag_parses_label():
    # 关键词判不出（无线索词），但 LLM 读懂语义判为安抚
    tagger = LLMSceneTagger(_backend("安抚"))
    assert tagger.tag("我搞砸了一切，觉得自己一无是处") == Register.COMFORT


def test_llm_tag_each_label():
    for label, reg in [("日常", Register.BANTER), ("挑衅", Register.TAUNT),
                       ("庄重", Register.SOLEMN), ("任务", Register.BUSINESS)]:
        assert LLMSceneTagger(_backend(label)).tag("...") == reg


def test_llm_tag_picks_first_label_when_noisy():
    # 输出带噪声但含合法标签
    assert LLMSceneTagger(_backend("应该是 安抚 吧")).tag("x") == Register.COMFORT


def test_llm_tag_falls_back_on_unparseable():
    # LLM 返回无法识别 → 回退启发式（靠关键词）
    tagger = LLMSceneTagger(_backend("我不确定哦"))
    assert tagger.tag("你肯定打不过我") == Register.TAUNT


def test_llm_tag_falls_back_on_error():
    def boom(system, msgs):
        raise RuntimeError("backend down")

    tagger = LLMSceneTagger(ScriptedBackend(boom), fallback=HeuristicSceneTagger())
    assert tagger.tag("我好难过，撑不下去了") == Register.COMFORT
