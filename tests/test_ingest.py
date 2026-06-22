"""数据接入层：OperatorIR 契约、通用 JSON 源、源无关的建库。

核心断言：一份「非明日方舟」的公司数据，能零改代码流过同一条管线。
"""

from __future__ import annotations

from app.config import ROOT
from app.ingest import OperatorIR, get_source
from app.ingest.ir import StoryLine, VoiceLine
from app.world import ARKNIGHTS, WorldProfile, load_world

EXAMPLE = ROOT / "data" / "companies" / "example" / "lyra.json"


def test_ir_roundtrip():
    ir = OperatorIR(
        name="测试", codename="Test", faction="某局",
        profile_facts=("事实一",),
        voice_lines=(VoiceLine(text="台词", title="问候"),),
        story_lines=(StoryLine(text="剧情台词", interlocutor="甲", prev="甲：你好", scene="序章"),),
    )
    again = OperatorIR.from_dict(ir.to_dict())
    assert again == ir


def test_generic_source_loads_non_arknights():
    src = get_source("generic", data_path=EXAMPLE)
    ir = src.fetch("星澪")
    assert ir.name == "星澪"
    assert "星海纪元" in ir.faction          # 明显不是明日方舟
    assert ir.profile_facts and ir.voice_lines and ir.story_lines
    # 剧情台词带语境（对谁说、回应什么）
    assert any(s.interlocutor for s in ir.story_lines)


def test_build_chunks_is_source_agnostic():
    from scripts.build_operator_kb import build_chunks  # noqa: PLC0415

    ir = get_source("generic", data_path=EXAMPLE).fetch("星澪")
    chunks = build_chunks(ir)
    assert chunks
    assert all(c["character"] == "星澪" for c in chunks)
    assert {"archive", "voice", "story"} <= {c["type"] for c in chunks}
    # 剧情 chunk 带语气档位标签
    assert any(c["type"] == "story" and c.get("register") for c in chunks)


def test_world_profile_default_and_load():
    assert load_world(None) == ARKNIGHTS
    assert load_world(ROOT / "data" / "world.yaml").work == "明日方舟"


def test_world_closure_is_ip_agnostic():
    from app.characters.cards import CharacterCard, render_system_prompt

    card = CharacterCard(name="星澪")
    world = WorldProfile(work="星海纪元", universe="银河", role_term="探员", ip_owner="某公司")
    prompt = render_system_prompt(card, world=world)
    assert "你正在扮演星海纪元探员「星澪」" in prompt
    assert "银河" in prompt and "泰拉" not in prompt and "明日方舟" not in prompt
