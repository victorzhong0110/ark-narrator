"""角色卡加载与 system prompt 渲染。"""

from __future__ import annotations

from app.characters.cards import render_system_prompt


def test_seed_characters_loaded(characters):
    for name in ("阿米娅", "凯尔希", "能天使", "德克萨斯", "陈"):
        assert name in characters


def test_system_prompt_has_world_closure(characters):
    card = characters["凯尔希"]
    prompt = render_system_prompt(card)
    assert "你正在扮演明日方舟干员「凯尔希」" in prompt
    assert "[扮演规则" in prompt           # 世界观封闭/安全前置说明
    assert "不承认自己是AI" in prompt


def test_system_prompt_injects_lore(characters):
    card = characters["阿米娅"]
    prompt = render_system_prompt(card, lore_block="- 阿米娅是罗德岛的领袖。")
    assert "阿米娅是罗德岛的领袖" in prompt


def test_card_fields_immutable(characters):
    card = characters["能天使"]
    assert isinstance(card.fallback_lines, tuple)
    assert isinstance(card.politics_deflections, tuple)
    assert card.fallback_lines  # 种子卡应配了兜底话术
