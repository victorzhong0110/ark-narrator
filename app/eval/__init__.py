"""角色保真自动评测：把"她像不像他家角色"变成可重复、可当 CI gate 的分数。

- KeywordJudge：确定性、零 API，捕捉灾难性失败（出戏/泄露/空答/穿越词）→ 适合 CI gate。
- LLMJudge：用裁判模型按 G-Eval 维度打细腻分（复活 ArkNarrator 研究侧方法）→ 夜跑/人评。
解决「300 个干员怎么信、不靠人肉抽查」。
"""

from app.eval.judge import DIMENSIONS, Judge, KeywordJudge, LLMJudge, Score
from app.eval.harness import Case, EvalReport, default_cases, run_eval

__all__ = [
    "DIMENSIONS", "Judge", "KeywordJudge", "LLMJudge", "Score",
    "Case", "EvalReport", "default_cases", "run_eval",
]
