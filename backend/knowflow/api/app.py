"""
M6 · FastAPI 应用工厂

【为什么用工厂函数而不是全局 `app = FastAPI()`？】
  1. 测试时能建多个 app 实例（不同配置）
  2. 能在创建时注入依赖（如测试用 FakeRAG）
  3. 避免 import 时就创建 app（import 副作用）

【启动方式】
  开发:  uvicorn knowflow.api.app:app --reload
  测试:  from fastapi.testclient import TestClient; client = TestClient(app)
  生产:  uvicorn knowflow.api.app:app --host 0.0.0.0 --workers 4
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowflow.api.routes import router

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    进程生命周期：启动时建表 + 起进程内 Worker + 回填 BM25，关闭时优雅停。

    【为什么开发模式要把 Worker 塞进 API 进程？】
      InMemoryQueue 只在本进程可见。独立跑 python -m knowflow.queue.worker
      消费不到 API 进程队列里的消息 —— 文档永远停在 pending。
      Kafka 模式下 Worker 有独立容器消费共享 topic，这里就不该再起一个，
      否则两个消费者抢消息（幂等能保证正确，但浪费）。
    """
    from knowflow.api.deps import get_queue, get_rag
    from knowflow.db.repo import db_on_status
    from knowflow.db.session import init_db
    from knowflow.queue import InMemoryQueue
    from knowflow.queue.worker import IngestWorker

    init_db()

    # 回填 BM25：向量在 Qdrant 持久，倒排索引在内存 —— 重启后必须重建
    try:
        rag = get_rag()
        n = rag.restore_bm25_from_store()
        log.info("startup_bm25_restore", n_chunks=n)
    except Exception as exc:
        log.warning("startup_bm25_restore_failed", error=str(exc))

    queue = get_queue()
    worker_task: asyncio.Task[Any] | None = None
    worker: IngestWorker | None = None

    if isinstance(queue, InMemoryQueue):
        # 必须用 get_rag() 单例：自己 new 一个 RAGPipeline 会写进另一套内存索引，
        # /ask 永远查不到刚入库的文档 —— 这是本项目最容易踩的坑。
        worker = IngestWorker(
            queue=queue,
            ingest_fn=get_rag().ingest_batch,
            on_status=db_on_status,
        )
        worker_task = asyncio.create_task(worker.run_forever())
        log.info("inprocess_worker_started", mode="InMemoryQueue")

    yield

    if worker_task is not None:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        if worker is not None:
            log.info("inprocess_worker_stopped", stats=worker.stats.summary())


def create_app(
    *,
    title: str = "KnowFlow API",
    version: str = "0.1.0",
    enable_cors: bool = True,
) -> FastAPI:
    """
    创建 FastAPI 应用实例。

    参数:
        title     OpenAPI 文档标题（/docs 页面显示）
        version   API 版本号
        enable_cors  允许跨域（前端本地开发时需要）
    """
    app = FastAPI(
        title=title,
        version=version,
        description="KnowFlow 企业知识库问答平台 —— 混合检索 / 融合 Agent 问答 / 异步入库",
        docs_url="/docs",        # Swagger UI 交互式文档
        redoc_url="/redoc",      # ReDoc 风格文档
        openapi_url="/openapi.json",  # OpenAPI schema（生成客户端用）
        lifespan=lifespan,
    )

    # CORS：显式来源列表（allow_origins=["*"] + allow_credentials=True 是规范非法组合，
    # 浏览器会拒绝带凭证的通配符请求）。开发指向 Vite dev server，生产改配置。
    if enable_cors:
        from knowflow.core.config import get_settings

        origins = [
            o.strip() for o in get_settings().cors_origins.split(",") if o.strip()
        ]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,  # 本项目无 cookie 鉴权
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 挂载路由
    app.include_router(router)

    # SPA 静态托管（仅当 frontend/dist 构建产物存在时生效，开发走 Vite dev server）
    _mount_spa(app)

    # 全局异常处理器 —— 统一错误响应格式（前端不用解析两种格式）
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        from fastapi.responses import JSONResponse
        import structlog

        log = structlog.get_logger("api")
        log.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误", "type": type(exc).__name__},
        )

    return app


def _mount_spa(app: FastAPI) -> None:
    """
    把前端构建产物挂到根路径。dist 不存在就跳过 —— 后端可以独立跑。

    【为什么 catch-all 要注册在最后？】
      FastAPI 按注册顺序匹配路由。catch-all GET /{path} 会吞掉一切，
      所以必须放在 /api、/docs 之后，让具体路由优先命中。
    """
    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if not (dist / "index.html").is_file():
        return

    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="spa-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_catch_all(full_path: str) -> FileResponse:
        # 前端路由（/chat /docs 页面等）都回落到 index.html，交给客户端路由
        candidate = dist / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(dist / "index.html")

    log.info("spa_mounted", dist=str(dist))


# uvicorn 入口：uvicorn knowflow.api.app:app --reload
app = create_app()
