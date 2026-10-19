# KnowFlow 知流 · 企业级 RAG 知识库智能体平台

> 面向企业内部文档的智能问答与自动化 Agent 平台。
> 技术栈：Python / FastAPI / Qdrant / PostgreSQL / Redis / Kafka / Docker / MiMo API

## 这是什么

一个**从文档入库到智能问答的完整工程**，而不是 Jupyter 里的 Demo。核心检索与 Agent 链路**手写实现了一遍**，又用 LangChain 做了对照版本，所以清楚框架在封装什么、什么场景该用、什么场景该绕开。

**已落地（M0~M8，代码可跑）**

- **Prompt 工程化**：Jinja2 模板 + 版本注册表（v1.0.0 / v1.1.0 可回滚）+ 结构化输出校验 + 注入三层防御
- **混合检索 RAG**：向量 + BM25 并行召回 → **RRF(k=60)** 融合 → **CrossEncoder 精排** → 引用溯源校验
- **Function Calling Agent**：ReAct 多工具编排，含参数四重校验、AST 白名单计算器、SQL 只读三重守卫、HITL 人工确认
- **数据层可靠性**：SQLAlchemy 2.0 五张业务表 + 事务边界 + 连接池调优 + 令牌桶限流 + 缓存降级
- **消息队列异步入库**：Kafka 抽象层 + Worker 消费者 + 幂等消费 + 三类异常分流 + 死信队列
- **REST 服务化**：FastAPI + Pydantic 契约 + 依赖注入 + 202 异步入库 + OpenAPI 自动文档
- **评估体系**：Faithfulness / Answer Relevancy / Context Precision / Context Recall 四指标 + LLM-as-judge 适配层
- **容器化部署**：多阶段 Dockerfile + docker-compose 六服务编排 + 云组件映射

**文档格式说明（诚实边界）**：当前入库接口接受**纯文本**字段（`POST /documents` 的 `text`）。PDF / Word 解析预留了 `DocParser` 接口，尚未接入 pypdf / python-docx。不要把它当成已交付的多格式能力。

**评估说明**：四指标由手写 `SimpleEvaluator`（关键词匹配，零成本，适合 CI 回归）计算；`RagasEvaluator` 适配层已就绪，但 `ragas` 尚未加入依赖组，正式 LLM-as-judge 评估**待实测**。

## 快速开始

```bash
# 1. 配置环境变量（在仓库根目录执行）
cp .env.example .env        # 填入你的 MIMO_API_KEY

# 2. 安装依赖（Python 3.12+）
cd backend && pip install -e ".[dev]"

# 3. 跑通第一课：工程化调用 LLM
python -m knowflow.demos.m0_hello_llm

# 4. 启动全栈（需要 Docker）
cd backend && docker compose up -d
#    起 6 个服务：api + worker + db(PostgreSQL) + redis + kafka + qdrant
#    API 文档: http://localhost:8000/docs
```

10 个 demo 实验室可逐个跑：

```bash
python -m knowflow.demos.m1_prompt_lab      # Prompt 版本管理
python -m knowflow.demos.m2_rag_lab         # 混合检索 RAG
python -m knowflow.demos.m3_agent_lab       # ReAct Agent
python -m knowflow.demos.m4_data_lab        # 数据层 / 限流
python -m knowflow.demos.m45_langchain_lab  # LangChain 对照
python -m knowflow.demos.m5_queue_lab       # 消息队列
python -m knowflow.demos.m6_api_lab         # FastAPI
python -m knowflow.demos.m7_eval_lab        # 评估体系
python -m knowflow.demos.m8_deploy_lab      # 部署讲解
```

> 部分 demo 会真调 MiMo API（`m0` / `m2` / `m3` / `m7`），需有效 `MIMO_API_KEY` 并产生少量费用。`m4` / `m5` / `m6` / `m8` 零成本可跑。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/00-学习路线图.md](docs/00-学习路线图.md) | 总学习路径、技术栈 技能对照、进度表 |
| [docs/01-Prompt工程.md](docs/01-Prompt工程.md) | M1：提示词工程化 |
| [docs/02-RAG.md](docs/02-RAG.md) | M2：分块 / Embedding / 混合检索 / 重排 / 引用 |
| [docs/03-Agent与Function-Calling.md](docs/03-Agent与Function-Calling.md) | M3：ReAct 循环、工具 Schema、安全护栏 |
| [docs/04-数据层.md](docs/04-数据层.md) | M4：SQLAlchemy 2.0、事务、Redis、限流 |
| [docs/05-LangChain对照.md](docs/05-LangChain对照.md) | M4.5：手写 vs LangChain 取舍 |
| [docs/06-消息队列.md](docs/06-消息队列.md) | M5：Kafka、幂等消费、重试与 DLQ |
| [docs/07-服务化.md](docs/07-服务化.md) | M6：FastAPI、依赖注入、202 异步 |
| [docs/08-评估体系.md](docs/08-评估体系.md) | M7：RAG 四指标、LLM-as-judge |
| [docs/09-容器化与部署.md](docs/09-容器化与部署.md) | M8：Docker、compose、云映射、Linux 运维 |
| [docs/resume-素材.md](docs/resume-素材.md) | 项目要点、可写数字清单、自查速查 |

## 架构

```
                    ┌──────────────────────┐
                    │   Web / API Client   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   FastAPI 服务层      │  ← M6
                    │ /health /ask         │
                    │ /agent /documents    │
                    └──────────┬───────────┘
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  RAG 引擎     │ │ Agent 引擎    │ │  评估引擎    │
      │ 向量+BM25     │ │ ReAct + FC   │ │ SimpleEvaluator│
      │ RRF + Rerank │ │ 安全护栏      │ │ + RAGAS 适配 │
      └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
             │                │                │
      ┌──────▼────────────────▼────────────────▼───────┐
      │  存储: Qdrant · PostgreSQL · Redis              │
      └──────────────────────┬─────────────────────────┘
                             │
      ┌──────────────────────▼─────────────────────────┐
      │  Kafka 异步流水线: 上传 → 解析 → 向量化 → 通知  │  ← M5
      └────────────────────────────────────────────────┘
```

## 部署

```bash
cd backend
docker compose up -d          # 起 6 个服务
docker compose logs -f api    # 看 API 日志
docker compose down           # 一键全关
```

本地组件与云服务的映射（代码零改动，换配置）：

| 本地 | AWS | GCP | Azure |
|---|---|---|---|
| Docker 容器 | ECS / EKS | Cloud Run / GKE | Container Instances / AKS |
| PostgreSQL | RDS | Cloud SQL | Azure DB for PG |
| Redis | ElastiCache | Memorystore | Azure Cache |
| Kafka | MSK | Pub/Sub | Event Hubs |
| 对象存储 | S3 | Cloud Storage | Blob Storage |

## 学习路径

见 [docs/00-学习路线图.md](docs/00-学习路线图.md)。每完成一个模块，更新其中的进度表。
