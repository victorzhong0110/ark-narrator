"""角色保真评测 CLI——可当 CI gate（低于门槛退出码非零）。

用法：
  ARK_BACKEND=mlx python scripts/eval_character.py 能天使 --threshold 6
  python scripts/eval_character.py 能天使 --judge llm        # 用裁判模型打细腻分
退出码：达标 0 / 不达标 1（接 CI 即可拦住人设退化的发布）。
"""

from __future__ import annotations

import argparse
import sys

from app.build import _system_markers
from app.characters.cards import CharacterCard
from app.config import load_settings
from app.eval import KeywordJudge, LLMJudge, default_cases, run_eval
from app.world import load_world


def _card_text(card: CharacterCard) -> str:
    bits = [card.name]
    for x in (card.faction, card.profile, card.personality, card.speaking_style):
        if x:
            bits.append(x)
    return "；".join(bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("character")
    ap.add_argument("--threshold", type=float, default=6.0)
    ap.add_argument("--judge", choices=["keyword", "llm"], default="keyword")
    args = ap.parse_args()

    from app.build import build_orchestrator

    orch = build_orchestrator(load_settings())
    if args.character not in orch.characters:
        print(f"未知角色：{args.character}")
        sys.exit(2)

    cards = orch.characters

    def respond_fn(character: str, user: str) -> str:
        return orch.respond("eval", "eval-user", character, user).text

    def card_text_fn(character: str) -> str:
        return _card_text(cards[character])

    world = load_world(load_settings().world_config)
    if args.judge == "llm":
        judge = LLMJudge(orch._backend)
    else:
        judge = KeywordJudge(_system_markers(world))

    report = run_eval(default_cases(args.character), respond_fn, judge, card_text_fn)
    print(report.render(threshold=args.threshold))
    sys.exit(0 if report.passed(args.threshold) else 1)


if __name__ == "__main__":
    main()
