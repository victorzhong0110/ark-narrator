"""语气档位：分类器、场景判定、按档位取示范。"""

from __future__ import annotations

from app.config import ROOT
from app.rag.retriever import LexicalRetriever
from app.rag.store import load_lore
from app.registers import Register, classify_register
from app.scene import HeuristicSceneTagger


def test_classify_comfort():
    assert classify_register("我今天好难过，感觉撑不住了") == Register.COMFORT


def test_classify_taunt():
    assert classify_register("就你这水平也敢跟我打一架？") == Register.TAUNT


def test_classify_business():
    assert classify_register("这次任务的奖金怎么算？") == Register.BUSINESS


def test_classify_solemn():
    assert classify_register("你愿意为了守护他们赌上生命吗？") == Register.SOLEMN


def test_classify_default_banter():
    assert classify_register("今天天气真好呀") == Register.BANTER


def test_scene_tagger_uses_history_as_fallback():
    tagger = HeuristicSceneTagger()
    # 本轮无情绪线索，但上一轮玩家在示弱 → 沿用 comfort
    history = [{"role": "user", "content": "我好害怕，撑不住了"},
               {"role": "assistant", "content": "..."}]
    assert tagger.tag("那……你说呢", history) == Register.COMFORT


def test_register_exemplars_return_matching_register():
    chunks = load_lore(ROOT / "data" / "lore")
    r = LexicalRetriever(chunks)
    ex = r.register_exemplars("能天使", Register.TAUNT.value, "黑手党", k=3)
    assert ex
    assert all(c.type == "story" and c.character == "能天使" for c in ex)
    assert any(c.register == Register.TAUNT.value for c in ex)


def test_register_exemplars_falls_back_when_register_empty():
    chunks = load_lore(ROOT / "data" / "lore")
    r = LexicalRetriever(chunks)
    # solemn 在该事件里没有 → 应退回任意剧情台词而非空
    ex = r.register_exemplars("能天使", Register.SOLEMN.value, "起誓", k=2)
    assert ex


def test_card_has_per_scene_register_styles(characters):
    exu = characters["能天使"]
    assert exu.register_styles, "应配置分场景姿态"
    assert {"comfort", "solemn"} <= set(exu.register_styles)


def test_orchestrator_injects_scene_stance(make_orchestrator):
    seen: dict[str, str] = {}

    def responder(system, msgs):
        seen["system"] = system
        return "好的"

    orch = make_orchestrator(responder)
    orch.respond("s", "u", "能天使", "我好难过，感觉撑不下去了")
    # 系统提示应带上「当前氛围=温柔安抚」+ 该场景下她的姿态描述
    assert "温柔安抚" in seen["system"]
    assert "你在这种情境下的样子" in seen["system"]
