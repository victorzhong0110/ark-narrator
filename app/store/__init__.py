"""可插拔状态存储：让对话服务有状态、且 worker 无状态（横向扩展前提）。

把会话历史、长期记忆、限流、会话风险、兜底轮换都外置到 Store：
- InMemoryStore：单机/开发（默认）
- SQLiteStore：单机持久（budget 档）
- RedisStore：多 worker 共享（standard/flagship 档、Mac mini 集群）

做完它，加一台机器就是加一份吞吐——因为请求不再绑定在某个进程的内存里。
"""

from app.store.base import Store, Turn
from app.store.factory import get_store

__all__ = ["Store", "Turn", "get_store"]
