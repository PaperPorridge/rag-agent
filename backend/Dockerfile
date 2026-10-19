# KnowFlow API Dockerfile
#
# 【构建】
#   docker build -t knowflow-api .
#
# 【运行】
#   docker run -p 8000:8000 --env-file .env knowflow-api
#
# 【多阶段构建的好处】
#   阶段 1 (builder): 安装编译依赖、pip install
#   阶段 2 (runtime): 只带运行时需要的文件
#   → 最终镜像不含 gcc/头文件，体积从 1.2GB 缩到 ~300MB
#   → 攻击面更小（编译工具链不进生产镜像）

# ---------------------------------------------------------------------------
# 阶段 1：构建（安装依赖）
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# 不生成 .pyc，不缓冲 stdout（日志实时输出）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 先装系统依赖（编译 Python 包需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先只拷 requirements.txt → Docker 缓存这层，代码变了不用重装依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ---------------------------------------------------------------------------
# 阶段 2：运行（最小镜像）
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/install/bin:$PATH \
    PYTHONPATH=/install/lib/python3.12/site-packages

WORKDIR /app

# 从 builder 阶段只拷贝装好的包（不带编译工具链）
COPY --from=builder /install /install

# 拷贝应用代码
COPY . .

# 非 root 用户运行（安全最佳实践）
RUN useradd -m -u 1000 knowflow && chown -R knowflow:knowflow /app
USER knowflow

# 暴露端口
EXPOSE 8000

# 健康检查（K8s / docker compose 都会用到）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# 启动命令
CMD ["python", "-m", "uvicorn", "knowflow.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
