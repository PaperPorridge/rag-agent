# KnowFlow 知流 · 企业级 RAG 知识库智能体平台

> 面向企业内部文档的智能问答与自动化 Agent 平台。
> 技术栈：Python / FastAPI / LangChain / Qdrant / PostgreSQL / Redis / Kafka / Docker / MiMo API

## 这是什么

一个**从文档入库到智能问答的完整工程**，而不是 Jupyter 里的 Demo：

- 多格式文档（PDF / Markdown / Word）**异步入库**：上传 → Kafka → 解析 → 分块 → 向量化 → 索引
- **混合检索 RAG**：向量检索 + BM25 + Rerank，答案带**引用溯源**
- **Function Calling Agent**：知识库检索、Text2SQL、计算器等工具的多轮 ReAct 编排
- **自动化评估**：RAGAS 指标（faithfulness / answer_relevancy / context_precision）回归
- **一键部署**：`docker compose up` 拉起 API + Worker + Qdrant + PostgreSQL + Redis + Kafka

## 快速开始

```bash
# 1. 安装依赖（Python 3.12+）
cd backend && pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env   # 填入 MIMO_API_KEY

# 3. 跑通第一课：工程化调用 LLM
python -m knowflow.demos.m0_hello_llm

# 4. 启动全栈（M8 后可用）
docker compose -f infra/docker-compose.yml up -d
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/00-学习路线图.md](docs/00-学习路线图.md) | 总学习路径、技术栈 技能对照、进度表 |
| [docs/01-Prompt工程.md](docs/01-Prompt工程.md) | M1：提示词工程化 |

## 架构

```
                    ┌──────────────────────┐
                    │   Web / API Client   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   FastAPI 服务层      │  ← M6
                    │  /chat /agent /docs  │
                    └──────────┬───────────┘
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  RAG 引擎     │ │ Agent 引擎    │ │  评估引擎    │
      │  混合检索     │ │ ReAct + FC   │ │  RAGAS       │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
      ┌──────▼────────────────▼────────────────▼───────┐
      │  存储: Qdrant · PostgreSQL · Redis · MinIO      │
      └──────────────────────┬─────────────────────────┘
                             │
      ┌──────────────────────▼─────────────────────────┐
      │  Kafka 异步流水线: 上传 → 解析 → 向量化 → 通知  │  ← M5
      └────────────────────────────────────────────────┘
```

## 学习路径

见 [docs/00-学习路线图.md](docs/00-学习路线图.md)。每完成一个模块，更新其中的进度表。
