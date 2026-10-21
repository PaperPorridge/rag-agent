# KnowFlow 知流 · 企业知识库问答平台

> 上传文档 → 智能问答（带出处引用）→ 计算题自动切 Agent 工具循环
> React + FastAPI + Qdrant + PostgreSQL + BAAI/bge-m3 + MiMo API

## 这是什么

一个**浏览器可用的企业知识库 Agent**。核心检索与 Agent 链路手写实现（不用 LangChain 跑主流程），存储与向量化用生产级组件，为评估量化打底。

**核心能力**

- **文档入库**：PDF 文本层解析 / txt / md，异步队列 + 状态机（pending → processing → ready/failed）
- **混合检索 RAG**：bge-m3 向量 + BM25 并行召回 → RRF(k=60) 融合 → CrossEncoder 精排 → 引用溯源校验
- **融合问答**：Router 自动分流——事实问题直答，计算/多步问题走 Agent 工具循环，单入口 `/ask` 返回链路徽章与工具轨迹
- **ReAct Agent**：知识库检索 + AST 白名单计算器，调用轨迹完整可溯
- **Web 前端**：React + Vite 双面板 SPA，文档管理 + 多轮对话 + 出处折叠 + 工具轨迹展示

## 快速开始

```bash
# 一次性
cp .env.example .env                # 填入 MIMO_API_KEY
cd backend && pip install -e ".[rag,data,server,dev,ingest]"
cd frontend && npm install
docker compose up -d db qdrant      # PostgreSQL 16 + Qdrant

# 终端 1
cd backend && uvicorn knowflow.api.app:app --reload --port 8000

# 终端 2
cd frontend && npm run dev          # http://localhost:5173
```

打开 http://localhost:5173 → 上传 PDF → 问事实问题（带出处）→ 问计算问题（看工具轨迹）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 组件健康 |
| POST | `/api/v1/ask` | 融合问答（`chain: rag\|agent`） |
| POST | `/api/v1/documents` | 粘贴文本入库（202） |
| POST | `/api/v1/documents/upload` | 文件上传 PDF/txt/md（202/415/422） |
| GET | `/api/v1/documents` | 文档列表 |
| DELETE | `/api/v1/documents/{id}` | 删除文档 + 向量 |

## 目录

```
backend/knowflow/
  core/       config / LLMClient / prompt 注册表
  rag/        chunking / embeddings / 混合检索 / RRF / rerank / pipeline
  agent/      ReAct 循环 / 工具注册表 / Router 分流 / 融合链路
  ingest/     PDF 文本层解析 + 白名单
  queue/      Kafka / InMemory 工厂 + Worker（幂等/重试/DLQ）
  db/         SQLAlchemy 2.0 模型 + session + 状态写回
  api/        FastAPI 路由 / 契约 / 依赖注入 / lifespan
frontend/     React 18 + Vite 5 + TypeScript（双面板 SPA）
```

## 诚实边界

- 扫描版 PDF 无 OCR，明确 422 拒收
- 无登录鉴权（内网演示）
- Agent 工具为只读两件套（知识检索 + 计算器）
- 检索按单轮问题做，history 只调节生成

## 部署

```bash
cd frontend && npm run build     # → frontend/dist
cd backend && uvicorn knowflow.api.app:app --host 0.0.0.0 --port 8000
# FastAPI 自动托管 dist，单进程即全栈
```
