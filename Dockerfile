# 容器镜像：api 后端 + 服务（standard/flagship 档）。
# 注：MLX/budget 档是 Apple Silicon 原生跑，不在容器内（mlx 无 linux 支持）。
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY app ./app
COPY data ./data

ENV ARK_BACKEND=api ARK_STORE=redis
EXPOSE 8000

# 用就绪/存活探针做容器健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/livez').status==200 else 1)" || exit 1

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
