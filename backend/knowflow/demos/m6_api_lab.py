"""
M6 演示：FastAPI 服务化实验室

运行:
    cd backend
    python -m knowflow.demos.m6_api_lab

本演示覆盖:
  1. 创建应用 + 路由注册 —— FastAPI 工厂模式
  2. 健康检查接口        —— GET /health
  3. RAG 问答接口        —— POST /ask（同步 200）
  4. 文档异步入库        —— POST /documents（202 Accepted）
  5. 状态查询            —— GET /documents/{id}
  6. 依赖注入            —— 测试时替换真 RAG 为假 RAG
  7. OpenAPI 文档        —— 自动生成的接口文档结构

环境说明: 不用启动真实 HTTP 服务器，用 FastAPI 的 TestClient 直接调接口。
          启动真实服务:  uvicorn knowflow.api.app:app --reload
          然后浏览器打开 http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from knowflow.api.app import create_app
from knowflow.api.deps import get_rag, reset_singletons


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# ---------------------------------------------------------------------------
# 1. 创建应用
# ---------------------------------------------------------------------------
def demo_create_app() -> TestClient:
    section("1. 创建 FastAPI 应用 —— 工厂模式")

    app = create_app(title="KnowFlow API Demo")
    client = TestClient(app)

    # 列出所有注册的路由
    routes = [r.path for r in app.routes if hasattr(r, "methods")]
    print(f"  已注册路由 ({len(routes)} 个):")
    for path in sorted(routes):
        print(f"    {path}")

    print("  [OK] 应用创建成功，路由全部挂载")
    return client


# ---------------------------------------------------------------------------
# 2. 健康检查
# ---------------------------------------------------------------------------
def demo_health(client: TestClient) -> None:
    section("2. 健康检查 —— GET /health")

    resp = client.get("/api/v1/health")
    data = resp.json()
    print(f"  状态码: {resp.status_code}")
    print(f"  响应:   {json.dumps(data, ensure_ascii=False, indent=2)}")
    print("  [OK] K8s liveness probe 就打这个接口")


# ---------------------------------------------------------------------------
# 3. RAG 问答接口
# ---------------------------------------------------------------------------
def demo_ask(client: TestClient) -> None:
    section("3. RAG 问答 —— POST /ask（同步 200）")

    # 正常请求
    resp = client.post("/api/v1/ask", json={
        "question": "差旅报销有时间限制吗？",
        "top_k": 3,
    })
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    print(f"  答案:   {data['answer'][:60]}...")
    print(f"  来源:   {data['sources']}")
    print(f"  引用合法: {data['citations_valid']}")
    print(f"  成本:   CNY {data['cost_cny']:.4f}")

    # 参数校验：question 为空 → 422（Pydantic 自动拦截）
    resp = client.post("/api/v1/ask", json={"question": ""})
    print(f"\n  空问题测试: 状态码={resp.status_code}（422 = 参数校验失败）")
    if resp.status_code == 422:
        detail = resp.json()["detail"][0]
        print(f"    错误: {detail['msg']}")

    print("  [OK] 正常返回 200 + JSON，参数错误返回 422，不用手写 if/else")


# ---------------------------------------------------------------------------
# 4. 文档异步入库（202）
# ---------------------------------------------------------------------------
def demo_upload(client: TestClient) -> str:
    section("4. 文档上传 —— POST /documents（202 Accepted）")

    resp = client.post("/api/v1/documents", json={
        "source": "差旅报销制度.pdf",
        "text": "员工差旅报销需在出差结束后30天内提交报销单，逾期需部门总监特批。",
        "owner_id": "user-001",
    })
    print(f"  状态码: {resp.status_code}（202 = 已接受，正在处理）")
    data = resp.json()
    print(f"  响应:   {json.dumps(data, ensure_ascii=False, indent=2)}")

    doc_id = data["doc_id"]
    print(f"\n  关键点：接口立刻返回了，入库是异步的（Worker 在后台做）")
    print(f"  前端拿 doc_id 轮询 GET /api/v1/documents/{doc_id} 查进度")
    print("  [OK] 上传接口不被解析阻塞，HTTP 不超时")
    return doc_id


# ---------------------------------------------------------------------------
# 5. 状态查询
# ---------------------------------------------------------------------------
def demo_status(client: TestClient, doc_id: str) -> None:
    section("5. 文档状态 —— GET /documents/{id}")

    resp = client.get(f"/api/v1/documents/{doc_id}")
    print(f"  状态码: {resp.status_code}")
    print(f"  响应:   {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")

    # 404 测试
    resp = client.get("/api/v1/documents/nonexistent-id")
    print(f"\n  查不存在的文档: 状态码={resp.status_code}（404）")
    print("  [OK] 状态机: pending → processing → ready / failed")


# ---------------------------------------------------------------------------
# 6. 依赖注入 —— 测试时替换真 RAG
# ---------------------------------------------------------------------------
class FakeRAG:
    """假 RAG：不调 LLM、不花钱、结果确定。测试专用。"""

    def ask(self, question: str, top_k: int = 5):
        from types import SimpleNamespace
        return SimpleNamespace(
            answer="这是假答案（测试模式，没调 LLM）",
            question=question,
            sources=["fake.pdf#0"],
            retrieved=[],
            citations_valid=True,
            citation_errors=[],
            llm=None,
        )


def demo_dependency_injection() -> None:
    section("6. 依赖注入 —— 测试时一行替换真 RAG 为假 RAG")

    reset_singletons()
    app = create_app()

    # 关键：一行替换依赖！
    app.dependency_overrides[get_rag] = lambda: FakeRAG()

    client = TestClient(app)
    resp = client.post("/api/v1/ask", json={"question": "测试问题"})
    data = resp.json()

    print(f"  状态码: {resp.status_code}")
    print(f"  答案:   {data['answer']}")
    print(f"  成本:   CNY {data['cost_cny']}（假 RAG 没调 LLM，成本为 0）")

    # 清理：移除替换，恢复真 RAG
    app.dependency_overrides.clear()
    reset_singletons()

    print()
    print("  对比：")
    print("    不用 DI:  测试跑 rag.ask() → 真调 LLM → 花钱、结果不确定、网络假失败")
    print("    用 DI:    dependency_overrides[get_rag] = fake → 1ms 跑完、成本 0、结果确定")
    print("  [OK] 这就是依赖注入存在的理由")


# ---------------------------------------------------------------------------
# 7. OpenAPI 文档
# ---------------------------------------------------------------------------
def demo_openapi(client: TestClient) -> None:
    section("7. OpenAPI 文档 —— 自动生成，不用手写")

    resp = client.get("/openapi.json")
    schema = resp.json()

    print(f"  标题: {schema['info']['title']}")
    print(f"  版本: {schema['info']['version']}")
    print(f"  接口数: {len(schema['paths'])}")
    print(f"\n  自动生成的接口文档:")
    for path, methods in sorted(schema["paths"].items()):
        for method in methods:
            print(f"    {method.upper():6s} {path}")

    print(f"\n  可交互文档:")
    print(f"    Swagger UI:  /docs      （能直接在网页上填参数试接口）")
    print(f"    ReDoc:       /redoc     （阅读友好）")
    print(f"    Schema:      /openapi.json （生成 TypeScript 客户端用）")
    print("  [OK] 不写一行文档代码，接口文档就有了")


# ---------------------------------------------------------------------------
async def main() -> None:
    print(__doc__)

    reset_singletons()
    client = demo_create_app()
    demo_health(client)
    demo_ask(client)
    doc_id = demo_upload(client)
    demo_status(client, doc_id)
    demo_dependency_injection()
    demo_openapi(client)

    section("完成")
    print("  概念回顾:")
    print("    · 工厂模式      create_app() 可创建多个实例，测试可注入假依赖")
    print("    · Pydantic 契约  请求/响应模型 = 校验 + 序列化 + 文档 三合一")
    print("    · 依赖注入      路由声明「要什么」，外部决定「给什么」，测试可替换")
    print("    · 状态码        200 同步完成 / 202 异步已接受 / 422 参数错误 / 404 不存在")
    print("    · OpenAPI       不写文档，/docs 自动生成交互式文档")
    print()
    print("  启动真实服务:")
    print("    uvicorn knowflow.api.app:app --reload")
    print("    浏览器打开 http://127.0.0.1:8000/docs")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
