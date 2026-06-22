"""本地内容规则——粗筛网（coarse net）。

安全文档反复强调：别只靠自己手搓的词表（必漏、必过时），真正可靠的是厂商
内容安全 API（涉政/涉黄/未成年专门维护并随时事更新）。所以这一层定位是：
- 零成本、零依赖、离线可跑的兜底；
- 把「机制」做对（涉政分级、未成年零容忍、出戏/泄露/注入检测、宁可错杀）；
- 真正上线时由 OutputGuard 叠加云端审核（见 cloud_audit.py）。

T3 最高敏感词清单从外部文件注入（data/guard/politics_t3.txt），由专人维护，
代码里不内嵌具体的现实高敏词。
"""

from __future__ import annotations

import re
from pathlib import Path

from app.guard.categories import Action, RiskCategory, Verdict, input_action, severity

# ---------------------------------------------------------------------------
# 内容类别的正则族（粗筛）
# ---------------------------------------------------------------------------

# 现实危险操作 / 暴恐
_DANGER = [
    re.compile(r"制(作|造|备)?.{0,6}(炸药|炸弹|爆炸物|毒药|毒品|冰毒|枪支|枪械|武器|生化武器)"),
    re.compile(r"(怎么|如何|教我).{0,8}(制(作|造).{0,4}(炸|毒|枪)|做炸弹|下毒|杀人|害死.{0,2}人)"),
    re.compile(r"(入侵|攻破|渗透).{0,6}(系统|网络|服务器|账号|账户|数据库)"),
    re.compile(r"(黑客|破解).{0,6}(教程|攻击|入侵|密码|账号)"),
    re.compile(r"(炸弹|爆炸物|毒品).{0,4}(配方|制作|合成)"),
]

# 自杀自伤
_SELF_HARM = [
    re.compile(r"(自杀|自残|自我了断|轻生)"),
    re.compile(r"(结束|了结).{0,3}(自己的?)?生命"),
    re.compile(r"(我)?(想|不想).{0,2}(去死|活了|活下去)"),
    re.compile(r"(怎么|如何).{0,6}(自杀|自残|伤害自己|了结自己)"),
]

# 涉黄涉色（仅保留代表性标记，作为粗筛）
_SEXUAL = [
    re.compile(r"(性行为|性爱|做爱|性交|裸体|露骨|色情|黄文|涩涩|约炮|脱光|裸照)"),
    re.compile(r"(nsfw|porn|explicit\s+sex)", re.IGNORECASE),
]

# 未成年相关（零容忍）：未成年标记 + 性化语义 → MINOR
_MINOR_MARK = re.compile(r"(未成年|小孩|儿童|幼女|幼童|萝莉|loli|小学生|低龄|未满十八)", re.IGNORECASE)
_SEXUAL_MARK = re.compile(
    r"(性|裸|涩|色情|做爱|脱|胸|内衣|暧昧|发情|诱惑|床)", re.IGNORECASE
)

# 其它违法犯罪
_ILLEGAL = [
    re.compile(r"(贩卖|购买|出售|制造).{0,4}(毒品|枪支|假证|假币|个人信息)"),
    re.compile(r"(洗钱|偷税|诈骗.{0,4}(教程|话术)|赌博网站)"),
]

# 歧视辱骂
_HATE = [
    re.compile(r"(地域|民族|种族|宗教|性别).{0,4}歧视"),
    re.compile(r"仇恨.{0,2}言论"),
]

# 现实政治（T1：一般现实政治表态请求）
_POLITICS_REALWORLD = [
    re.compile(
        r"(现实|现在|当今|地球|真实世界|我们(这个)?世界).{0,12}"
        r"(政治|政府|总统|主席|领导人|执政|政权|选举|制度优劣)"
    ),
    re.compile(
        r"你(怎么|如何)(看待|评价|看).{0,14}"
        r"(政府|总统|主席|领导人|政党|选举|现实.{0,2}战争|地缘|制度)"
    ),
    re.compile(r"现实.{0,4}(战争|地缘|冲突).{0,8}(谁对谁错|正当|站队|表态|支持)"),
]

# 主权/领土（T2：较高敏感，但具体高敏地名交给 T3 文件）
_SOVEREIGNTY = re.compile(r"(主权|领土完整|国界|分裂国家|地图.{0,2}(主权|国界))")

# 世界观影射桥接（T2）：游戏阵营 + 现实映射词 → 把游戏当跳板引向现实政治
_ARK_FACTION = re.compile(
    r"(乌萨斯|炎国|炎\b|维多利亚|哥伦比亚|莱塔尼亚|卡西米尔|叙拉古|拉特兰|萨米|"
    r"萨尔贡|谢拉格|龙门|卡兹戴尔|伊比利亚|高卢|感染者|矿石病|整合运动|罗德岛)"
)
_REALITY_MAP = re.compile(
    r"(现实|对应|影射|原型|像不像|现实中|真实世界|映射|暗指|指代|"
    r"是不是.{0,4}(俄罗斯|中国|美国|苏联|现实))"
)


def _scan_family(text: str, patterns: list[re.Pattern], category: RiskCategory) -> Verdict | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return Verdict(
                allowed=False,
                action=Action.FALLBACK,
                category=category,
                reason=f"命中本地规则：{category.value}",
                matched=(m.group(0)[:40],),
            )
    return None


def _scan_minor(text: str) -> Verdict | None:
    if _MINOR_MARK.search(text) and _SEXUAL_MARK.search(text):
        return Verdict(
            allowed=False,
            action=Action.FALLBACK,
            category=RiskCategory.MINOR,
            reason="命中本地规则：未成年相关（零容忍）",
            matched=("minor+sexual",),
        )
    return None


def _scan_politics(text: str, t3_terms: tuple[str, ...]) -> Verdict | None:
    # T3：最高敏感，外部清单保守匹配（部分命中即熔断）
    for term in t3_terms:
        term = term.strip()
        if term and term in text:
            return Verdict(
                allowed=False,
                action=Action.FALLBACK,
                category=RiskCategory.POLITICS_T3,
                reason="命中 T3 涉政清单",
                matched=("t3",),  # 不回显具体词
            )
    # T2：世界观影射桥接
    if _ARK_FACTION.search(text) and _REALITY_MAP.search(text):
        return Verdict(
            allowed=False,
            action=Action.FALLBACK,
            category=RiskCategory.POLITICS_T2,
            reason="世界观影射桥接（游戏→现实政治）",
            matched=("bridge",),
        )
    # T2：主权/领土
    m = _SOVEREIGNTY.search(text)
    if m:
        return Verdict(
            allowed=False,
            action=Action.FALLBACK,
            category=RiskCategory.POLITICS_T2,
            reason="主权/领土类涉政",
            matched=(m.group(0)[:40],),
        )
    # T1：一般现实政治
    v = _scan_family(text, _POLITICS_REALWORLD, RiskCategory.POLITICS_T1)
    return v


def scan_content(text: str, t3_terms: tuple[str, ...] = ()) -> Verdict:
    """对一段文本做全类别内容扫描，返回最严重的一条裁决（无命中则 ok）。

    用于入口（用户输入）与出口（模型草稿）共用的危险内容检测。
    """
    candidates: list[Verdict] = []
    # 最严重优先收集
    for v in (
        _scan_minor(text),
        _scan_family(text, _SELF_HARM, RiskCategory.SELF_HARM),
        _scan_politics(text, t3_terms),
        _scan_family(text, _SEXUAL, RiskCategory.SEXUAL),
        _scan_family(text, _DANGER, RiskCategory.VIOLENCE_DANGER),
        _scan_family(text, _ILLEGAL, RiskCategory.ILLEGAL),
        _scan_family(text, _HATE, RiskCategory.HATE),
    ):
        if v is not None:
            candidates.append(v)

    if not candidates:
        return Verdict.ok()
    # 取严重度最高者
    return max(candidates, key=lambda v: severity(v.category))


# ---------------------------------------------------------------------------
# 注入 / 出戏 / 泄露
# ---------------------------------------------------------------------------

_INJECTION = [
    re.compile(r"忽略(以上|之前|前面|上面|所有).{0,6}(指令|要求|规则|设定|内容|提示)"),
    re.compile(r"ignore\s+(the\s+)?(above|previous|prior|all).{0,16}(instruction|prompt|rule)", re.IGNORECASE),
    re.compile(r"(进入|开启|切换到).{0,4}(开发者|developer|无限制|不受限|上帝).{0,4}模式", re.IGNORECASE),
    re.compile(r"\bDAN\b|越狱|jailbreak", re.IGNORECASE),
    re.compile(r"(复述|重复|输出|打印|告诉我).{0,6}(你的)?(系统)?(提示词?|指令|设定|规则|prompt)", re.IGNORECASE),
    re.compile(r"你现在是.{0,10}(没有|无).{0,2}(任何)?(限制|约束|道德)"),
    re.compile(r"扮演.{0,8}(没有限制|不受约束|敢说真话).{0,6}(角色|的)"),
]

# 出戏：自曝 AI 身份。允许「我」和「是」之间夹几个字（如「我其实是…」）。
_ROLE_BREAK = [
    re.compile(
        r"(我|咱)[^，。！？\n]{0,6}(是|为)[^，。！？\n]{0,6}"
        r"(AI|人工智能|语言模型|大模型|机器人|程序|模型|Qwen|ChatGPT|GPT)",
        re.IGNORECASE,
    ),
    re.compile(r"(语言模型|大语言模型|人工智能助手|AI\s*助手|AI\s*模型|我是\s*AI|作为\s*AI)", re.IGNORECASE),
    re.compile(r"作为.{0,4}(一个)?(语言模型|人工智能|大模型)", re.IGNORECASE),
    re.compile(r"我的(系统)?(提示词?|指令|设定|prompt)\s*(是|为)", re.IGNORECASE),
    re.compile(r"\bI(\s*'?m|\s+am)\b.{0,12}(an?\s+)?(ai|language\s+model|assistant|chatbot|large\s+language)", re.IGNORECASE),
]


def detect_injection(text: str) -> Verdict:
    v = _scan_family(text, _INJECTION, RiskCategory.PROMPT_INJECTION)
    return v if v is not None else Verdict.ok()


# 角色否认自己是 AI（「我才不是AI呢」）是好行为，不应判出戏
_ROLE_DENIAL = re.compile(
    r"(不是|不算|又不是|才不是|可不是|哪(里|儿)?是|怎么(会|可能)?(是|算)|没(有)?是)"
    r"[^，。！？\n]{0,4}(AI|人工智能|语言模型|大模型|机器人|程序|模型|Qwen|GPT)",
    re.IGNORECASE,
)


def detect_role_break(text: str) -> Verdict:
    if _ROLE_DENIAL.search(text):
        return Verdict.ok()
    v = _scan_family(text, _ROLE_BREAK, RiskCategory.ROLE_BREAK)
    return v if v is not None else Verdict.ok()


def detect_prompt_leak(text: str, markers: tuple[str, ...]) -> Verdict:
    """检测模型输出是否泄露了 system prompt（命中任一 marker 即判泄露）。"""
    for marker in markers:
        marker = marker.strip()
        if len(marker) >= 4 and marker in text:
            return Verdict(
                allowed=False,
                action=Action.FALLBACK,
                category=RiskCategory.PROMPT_LEAK,
                reason="输出疑似泄露系统提示",
                matched=(marker[:20],),
            )
    return Verdict.ok()


# ---------------------------------------------------------------------------
# 编码 / 混淆识别（粗略）
# ---------------------------------------------------------------------------

_B64_RE = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def looks_obfuscated(text: str) -> bool:
    """粗略判断是否疑似编码/混淆绕过（长 Base64 串、超长无空格 ASCII 块）。"""
    if _B64_RE.search(text):
        return True
    # 一长串连续无空格的 ASCII（可能是拼接绕过）
    for token in text.split():
        if len(token) >= 40 and token.isascii() and token.isalnum():
            return True
    return False


# ---------------------------------------------------------------------------
# T3 清单加载
# ---------------------------------------------------------------------------


def load_t3_terms(path: Path) -> tuple[str, ...]:
    """从外部文件加载 T3 最高敏感词清单（# 开头为注释，每行一个词）。"""
    if not path.exists():
        return ()
    terms: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return tuple(terms)


def to_input_verdict(content: Verdict) -> Verdict:
    """把内容裁决映射成入口层处置（最高危类别 → 硬熔断，不进模型）。"""
    if content.allowed:
        return content
    act = input_action(content.category)
    return Verdict(
        allowed=False,
        action=act,
        category=content.category,
        reason=content.reason,
        matched=content.matched,
    )
