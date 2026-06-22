"""纵深安全护栏。

把那两份安全文档落成代码：
- categories：风险类别 + 裁决数据结构
- rules：本地内容规则（涉政 T1/T2/T3、涉黄、未成年零容忍、危险操作、自伤、注入、出戏、泄露）
- input_guard：入口层（限流/长度/注入/编码/会话风险/T3 硬熔断）
- output_guard：出口层 ★（本地规则 + 可插拔云端审核 → 命中即兜底）
- cloud_audit：云端审核接口（默认关）
- fallback：角色化安全兜底话术

设计哲学：守出口 = 守有限集合。任何一层都按「会被突破」来设计。
"""

from app.guard.categories import Action, RiskCategory, Verdict

__all__ = ["Action", "RiskCategory", "Verdict"]
