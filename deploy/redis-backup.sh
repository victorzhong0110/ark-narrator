#!/usr/bin/env bash
# Redis 状态备份（会话/长期记忆）。恢复：把 dump.rdb 放回 redis 数据卷再启动即可。
set -euo pipefail
C="${REDIS_CONTAINER:-ark-narrator-prod-redis-1}"
docker exec "$C" redis-cli BGSAVE
sleep 2
mkdir -p backups
OUT="backups/dump-$(date +%Y%m%d-%H%M%S).rdb"
docker cp "$C:/data/dump.rdb" "$OUT"
echo "已备份 → $OUT"
