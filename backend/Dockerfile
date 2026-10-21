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
#   → 最终镜像不含 gcc/头文件，体积明显缩小
#   → 攻击面更小（编译工具链不进生产镜像）
#   [待测] 具体体积数字请以 `docker images` 实测为准，不要引用估算值

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

# 分层缓存顺序：依赖清单和源码必须都进构建层，hatchling 才能打包。
#   COPY pyproject.toml + knowflow/ → pip install → 依赖装进 /install
# 依赖层的缓存命中介于「改依赖」和「改代码」之间；要极致缓存可先导出
# requirements.lock 只装第三方依赖，再 --no-deps 装本包（个人项目从简）。
COPY pyproject.toml .
COPY knowflow/ ./knowflow/

# extras 必须显式列出：server 带 uvicorn/fastapi，queue 带 aiokafka，ingest 带 pypdf，
# 否则 CMD 的 `python -m uvicorn` 会 No module named uvicorn（本次修的坑）
RUN pip install --no-cache-dir --prefix=/install ".[rag,data,queue,server,ingest]"

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
