"""数据接入层：把任意来源的角色数据，统一成标准中间表示 OperatorIR。

这是「直接接轨」的关键 —— 引擎只认 OperatorIR，不认数据从哪来。公司接入只需：
实现一个 OperatorSource（或把数据导出成 OperatorIR 的 JSON schema），下游全部复用。
"""

from app.ingest.ir import OperatorIR, StoryLine, VoiceLine
from app.ingest.source import OperatorSource, get_source

__all__ = ["OperatorIR", "VoiceLine", "StoryLine", "OperatorSource", "get_source"]
