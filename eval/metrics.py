"""
Evaluation metrics for ArkNarrator.
Measures: character consistency, world-fidelity, fluency.
"""

import re
import json
import logging
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Arknights world-fidelity keywords
ARK_KEYWORDS = [
    "源石", "罗德岛", "感染者", "整合运动", "泰拉", "龙门", "萨卡兹",
    "炎国", "维多利亚", "莱塔尼亚", "阿斯卡纶", "源石技艺", "源石病",
    "博士", "凯尔希", "阿米娅",
]


@dataclass
class EvalResult:
    sample_id: str
    task_type: str
    lore_score: float        # world-fidelity (0-10)
    consistency_score: float # character consistency (0-10)
    fluency_score: float     # language fluency (0-10)
    overall: float
    judge_reasoning: str


class LoreChecker:
    """Rule-based check for Arknights world-fidelity."""

    def score(self, text: str) -> float:
        """Score based on correct Arknights terminology usage."""
        hits = sum(1 for kw in ARK_KEYWORDS if kw in text)
        # Check for anachronisms (modern tech in Arknights world)
        anachronisms = ["手机", "互联网", "电脑", "飞机", "汽车"]
        penalties = sum(1 for a in anachronisms if a in text)
        raw = min(hits * 1.2 - penalties * 2, 10)
        return max(0.0, round(raw, 1))


class GPTJudge:
    """Use GPT-4o as an LLM judge for qualitative evaluation."""

    def __init__(self):
        self.client = OpenAI()

    def judge(self, instruction: str, output: str, task_type: str) -> dict:
        prompt = f"""你是明日方舟资深玩家，请评估以下生成内容的质量。

任务类型：{task_type}
指令：{instruction}
生成内容：{output}

请从以下三个维度各打分（0-10分），并给出简短理由：
1. 世界观还原度：内容是否符合明日方舟世界观设定
2. 角色一致性：语气、性格是否与已知角色设定一致（如无特定角色则跳过）
3. 语言流畅度：内容是否通顺、有文学感

以JSON格式返回：{{"lore": 分数, "consistency": 分数, "fluency": 分数, "reasoning": "理由"}}"""

        resp = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
