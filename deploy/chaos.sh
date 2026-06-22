#!/usr/bin/env bash
# 混沌演练：边压边注入故障，验证系统不雪崩、优雅降级。
# 前提：先起 compose-prod（docker compose -f deploy/compose-prod.yml up -d --scale app=3）。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
URL="${URL:-http://localhost:8000/v1/chat}"
CF="deploy/compose-prod.yml"

echo "[chaos] 后台持续压测…"
( cd "$ROOT" && python bench/loadtest.py --url "$URL" -n 6000 -c 40 > /tmp/chaos_load.txt 2>&1 ) &
LOAD=$!
sleep 2

echo "[chaos] 故障1：杀掉一个 app 副本（验证 LB 绕过坏实例）"
docker kill ark-narrator-prod-app-2 >/dev/null 2>&1 || true
sleep 3

echo "[chaos] 故障2：重启 Redis（验证 ResilientStore 降级，不 500）"
( cd "$ROOT" && docker compose -f "$CF" restart redis >/dev/null 2>&1 ) || true

wait "$LOAD"
echo "[chaos] 压测结果（注入故障期间）："
cat /tmp/chaos_load.txt
