# 部署（P1：IaC + 全栈一键 + HA 基础）

两条一键路径，按规模选。机密走 `deploy/.env`（compose）/ Secret（K8s），不入库。

## 单主机：docker compose（standard 档，无 GPU 也能验证 IaC/HA 链路）

```bash
cp deploy/.env.example deploy/.env        # 按需填真模型/审核/鉴权

# 基础（Redis 持久化 + nginx LB + 可扩 app）
docker compose -f deploy/compose-prod.yml up -d --scale app=3

# + Redis Sentinel 高可用（主挂自动切换）
docker compose -f deploy/compose-prod.yml --profile ha up -d --scale app=3

# 一键全停
docker compose -f deploy/compose-prod.yml --profile ha down
```
- 默认 `ARK_BACKEND=scripted`（无需 GPU，验证 LB→app→Redis 整条链路）；真模型把 `.env` 里
  `ARK_BACKEND` 改 `api`/`pool` 指向你的推理服务（Mac 机队或 vLLM）。
- app 无状态，状态全在 Redis → `--scale app=N` 即加吞吐。

## 集群：Kubernetes（kustomize，自带 HPA/探针/Ingress）

```bash
kubectl apply -k deploy/k8s/overlays/standard     # 3 副本起步，HPA 3→12
kubectl apply -k deploy/k8s/overlays/flagship     # 6 副本起步，HPA 6→24，更大上下文 + aliyun 审核
```
含：Deployment（就绪/存活探针 + 资源限额）、Service、Ingress、HPA、Secret（占位，生产用
External Secrets/Vault 注入）、Redis StatefulSet（持久化）。

## Redis 高可用

- compose：`--profile ha` 起 master + replica + 3 sentinel；app 设 `ARK_REDIS_SENTINELS=sentinel:26379`
  即走 Sentinel（`app/store/redis.py` 已支持，主挂自动连新主）。
- K8s：基线是单实例 + 持久化（消除「重启丢状态」）；生产 HA 换 Bitnami redis（Sentinel/Cluster）
  或托管 Redis，app 侧只需设 `ARK_REDIS_SENTINELS`。

## 已验证 / 待真环境

- ✅ 本地已验证：`docker compose config` 渲染通过；`kubectl kustomize` 两档构建通过、补丁生效
  （副本数/生成参数/审核 provider/HPA 上限/独立 namespace 均按档差异化）。
- ⏳ 需真环境：完整 schema 校验（需集群或 kubeconform）、Sentinel 真故障切换、HPA 真扩缩、
  跨节点端到端——这些需要一个真 K8s 集群 / 多机来跑（见 docs/PRODUCTION_READINESS.md）。
