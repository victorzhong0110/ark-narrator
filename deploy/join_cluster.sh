#!/usr/bin/env bash
# 一键加入集群：在「新 Mac mini」上跑这一个脚本即可成为模型节点。
# 它会：装依赖 → 起 mlx_lm.server（OpenAI 兼容）→ 心跳注册进 Redis。
# app 层（ARK_BACKEND=pool）会在一个心跳周期内自动发现本节点并开始路由——无需改 nginx、无需重启。
#
# 用法：
#   ARK_REDIS_URL=redis://<app/redis主机>:6379/0 bash deploy/join_cluster.sh
# 可选：ARK_MODEL_PATH / ARK_NODE_PORT / ARK_NODE_GROUP / ARK_NODE_ADDR
set -euo pipefail

: "${ARK_REDIS_URL:?请设置 ARK_REDIS_URL=redis://<app/redis主机>:6379/0}"
MODEL="${ARK_MODEL_PATH:-mlx-community/Qwen3-8B-4bit}"
PORT="${ARK_NODE_PORT:-8080}"
GROUP="${ARK_NODE_GROUP:-models}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> 安装依赖（mlx-lm + redis）"
pip install -q mlx-lm redis

echo "==> 启动模型服务 mlx_lm.server（$MODEL，端口 $PORT）"
python -m mlx_lm.server --model "$MODEL" --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
sleep 5   # 等服务起来

echo "==> 心跳注册进集群（组 $GROUP）。Ctrl-C 退出即自动下线本节点。"
ARK_NODE_GROUP="$GROUP" ARK_NODE_PORT="$PORT" python "$HERE/registrar.py"
