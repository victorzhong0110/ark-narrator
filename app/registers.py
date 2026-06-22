"""语气档位（register）——角色在不同情境/情绪/对象下的说话风格。

同一个干员对队友调侃、对敌人挑衅、对需要保护的人安抚、庄重时起誓，语气完全不同。
把这种「分场景的语气」显式建模：
- 给剧情台词按档位打标（build_operator_kb）；
- 运行时判定当前对话的情绪档位（SceneTagger），调用匹配档位的真实台词当示范。

这是单一事实源：打标和运行时判定用同一套分类器，保证一致。
"""

from __future__ import annotations

import re
from enum import Enum


class Register(str, Enum):
    BANTER = "banter"        # 轻松调侃（日常、玩闹）—— 默认
    TAUNT = "taunt"          # 挑衅嘴硬（对抗、被质疑、战斗）
    COMFORT = "comfort"      # 温柔安抚（对方脆弱、难过、受伤、害怕）
    SOLEMN = "solemn"        # 庄重认真（信念、起誓、守护、生死）
    BUSINESS = "business"    # 干活报酬（任务、奖金、出发）


REGISTER_LABEL: dict[Register, str] = {
    Register.BANTER: "轻松调侃",
    Register.TAUNT: "挑衅嘴硬",
    Register.COMFORT: "温柔安抚",
    Register.SOLEMN: "庄重认真",
    Register.BUSINESS: "干活报酬",
}

# 关键词线索。判定优先级见 _PRIORITY（脆弱/对抗优先于日常）。
_CUES: dict[Register, re.Pattern] = {
    Register.COMFORT: re.compile(
        r"(难过|伤心|害怕|怕|哭|想哭|受伤|疼|好累|心累|孤独|寂寞|担心|痛苦|绝望|"
        r"撑不住|坚持不下去|别怕|没事吧|安慰|失去|不安|脆弱|崩溃|委屈|"
        r"搞砸|没用|我不行|做不到|失败|都是我的错|自责|后悔|迷茫|没希望|想放弃|对不起)"
    ),
    Register.TAUNT: re.compile(
        r"(打一架|干一架|揍|敌人|黑帮|挑衅|你不行|不行吧|废物|看不起|来啊|"
        r"怕了|嚣张|找死|不服|开打|战斗|对手|挑战|瞧不起|有本事|敢不敢|怂|"
        r"打得过|打不过|打得赢|打不赢|单挑|不是对手|未必.{0,4}得过|也就那样|"
        r"比划|试试|赢不了|战胜你|揍你|打你)"
    ),
    Register.SOLEMN: re.compile(
        r"(起誓|发誓|守护|信仰|主啊|义人|牺牲|生与死|生死|使命|责任|拯救|"
        r"永远|认真点|郑重|信念|觉悟|赌上)"
    ),
    Register.BUSINESS: re.compile(
        r"(任务|奖金|报酬|活儿|工作|出发|护送|送货|钱|雇佣|订单|差事|目标|"
        r"行动|委托|佣金|出任务|干活|接单)"
    ),
}

# 多档位命中时的优先级：脆弱/对抗/庄重 优先于 日常事务。
_PRIORITY = [
    Register.COMFORT,
    Register.TAUNT,
    Register.SOLEMN,
    Register.BUSINESS,
]


def classify_register(text: str) -> Register:
    """判定一段文本（玩家发言 或 角色台词）所处的语气档位。无明显线索→BANTER。"""
    if not text:
        return Register.BANTER
    for reg in _PRIORITY:
        if _CUES[reg].search(text):
            return reg
    return Register.BANTER
