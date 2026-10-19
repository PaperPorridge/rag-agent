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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knowflow.api.routes import router


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
        description="KnowFlow 企业级 RAG 知识库智能体平台 —— 混合检索 / Function Calling Agent / 异步入库",
        docs_url="/docs",        # Swagger UI 交互式文档
        redoc_url="/redoc",      # ReDoc 风格文档
        openapi_url="/openapi.json",  # OpenAPI schema（生成客户端用）
    )

    # CORS：允许前端跨域调用（生产环境应该限制到具体域名）
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],       # 生产改成 ["https://your-domain.com"]
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 挂载路由
    app.include_router(router)

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


# uvicorn 入口：uvicorn knowflow.api.app:app --reload
app = create_app()
