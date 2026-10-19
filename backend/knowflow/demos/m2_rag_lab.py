"""
M2 演示：RAG 核心实验室

运行:
    cd backend
    pip install -e ".[rag]"
    python -m knowflow.demos.m2_rag_lab

本演示覆盖:
  1. 分块策略对比 —— 固定分块切断句子，递归分块尊重结构
  2. 语义检索   —— 不同词、相同含义能命中
  3. 精确检索   —— 型号/编号只有 BM25 能命中（向量检索会失败）
  4. 混合检索   —— RRF 融合两头都命中
  5. Rerank     —— 真正相关的排到 top-1
  6. 端到端问答 —— 带 [来源: ...] 引用 + 引用合法性校验

说明: 默认用 HashingEmbedder（零依赖、确定性），保证任何机器都能跑通链路。
      装了 sentence-transformers 后改 USE_REAL_EMBEDDING=True 可切换语义向量。
"""

from __future__ import annotations

import sys
import time

from knowflow.core.llm import LLMClient
from knowflow.rag.chunking import FixedChunker, RecursiveChunker
from knowflow.rag.embeddings import create_embedder
from knowflow.rag.pipeline import RAGPipeline
from knowflow.rag.retrieval import BM25Index, Reranker, RetrievedChunk, reciprocal_rank_fusion

USE_REAL_EMBEDDING = True  # 改 True 需先 pip install -e ".[rag]" 并下载模型

# ---------------------------------------------------------------------------
# 企业知识库样例：故意包含「语义问题」和「精确编号问题」两类
# ---------------------------------------------------------------------------
CORPUS: list[tuple[str, str]] = [
    (
        "员工差旅报销需在出差结束后30天内提交报销单，逾期需部门总监特批。"
        "住宿标准为一线城市每晚500元，二线城市每晚350元。机票需提前预订经济舱。",
        "差旅报销制度.pdf",
    ),
    (
        "费用核销流程：提交发票原件至财务部，财务在5个工作日内完成审核。"
        "单笔金额超过5000元需附上事前审批邮件截图。",
        "费用核销SOP.md",
    ),
    (
        "公司开发机型号为 ThinkPad-X1-Carbon-Gen12，标配内存32GB。"
        "显卡扩展坞型号为 RTX-4090-24G，申请需走IT工单，工单模板编号 IT-FORM-2024-001。",
        "IT设备清单.md",
    ),
    (
        "新员工入职第一周需完成：领取门禁卡、开通企业邮箱、参加安全培训。"
        "试用期考核由直属主管在入职90天内完成。",
        "入职指引.pdf",
    ),
    (
        "年假规则：入职满一年享有10天年假，满三年15天。"
        "年假需提前3个工作日在OA系统申请，连续休假超过5天需总监审批。",
        "休假管理办法.pdf",
    ),
]


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def demo_chunking() -> None:
    section("1) 分块策略对比：固定 vs 递归")
    text = (
        "公司开发机型号为 ThinkPad-X1-Carbon-Gen12，标配内存32GB。"
        "显卡扩展坞型号为 RTX-4090-24G，申请需走IT工单。"
        "工单模板编号为 IT-FORM-2024-001，请勿直接找IT同事故批。"
    )
    print(f"原文({len(text)}字符): {text}\n")

    fixed = FixedChunker(chunk_size=50, overlap=10).split(text, "设备.md")
    recursive = RecursiveChunker(chunk_size=50, overlap=10).split(text, "设备.md")

    print(f"【固定分块 chunk_size=50】→ {len(fixed)} 块，注意句子被切断:")
    for c in fixed:
        print(f"  · {c.text!r}")

    print(f"\n【递归分块 chunk_size=50】→ {len(recursive)} 块，尽量贴合句边界:")
    for c in recursive:
        print(f"  · {c.text!r}")

    print("\n结论: 固定分块会把「RTX-4090-24G」切成两半，检索回来就是残缺碎片。")
    print("      递归分块按段落→句子→词逐级切，语义单元更完整。")


def demo_retrieval_modes(rag: RAGPipeline) -> None:
    section("2) 向量检索 vs BM25：各自的能力边界")

    # -- 语义查询：换了词，意思相同 --
    semantic_q = "怎么申请休假？"
    print(f"\n[语义查询] {semantic_q!r}  （答案在《休假管理办法》里，但没出现「休假」二字）")
    hits = rag.search_only(semantic_q, top_k=3, use_rerank=False)
    for i, h in enumerate(hits, 1):
        print(f"  {i}. [{h.citation}] score={h.final_score:.4f} | {h.chunk.text[:40]}...")
    print("  → 向量检索按语义命中；BM25 若只有「年假/OA」这些词，会漏。")

    # -- 精确查询：型号编号 --
    exact_q = "RTX-4090-24G 是什么？"
    print(f"\n[精确查询] {exact_q!r}  （考查符号串匹配）")

    # 只看向量
    vec_only = rag.store.search(rag.embedder.embed_query(exact_q), top_k=3)
    print("  向量检索 top-3:")
    for i, m in enumerate(vec_only, 1):
        print(f"    {i}. [{m.chunk.source}#{m.chunk.chunk_index}] score={m.score:.4f}")

    # 只看 BM25
    bm25_only = rag.bm25.search(exact_q, top_k=3)
    print("  BM25 检索 top-3:")
    for i, m in enumerate(bm25_only, 1):
        print(f"    {i}. [{m.chunk.source}#{m.chunk.chunk_index}] score={m.score:.4f}")
    print("  → 型号 RTX-4090-24G 是符号串，语义空间没有「相近」概念，向量容易跑偏；")
    print("    BM25 做词面匹配反而稳。所以必须混合。")


def demo_hybrid() -> None:
    section("3) 混合检索 + RRF 融合")

    # 构造两路排名（刻意让它们擅长不同查询）
    bm25 = BM25Index()
    chunks = RecursiveChunker(chunk_size=200, overlap=20).split_documents(CORPUS)
    bm25.index(chunks)

    query = "RTX-4090-24G 显卡扩展坞怎么申请？"
    print(f"查询: {query!r}\n")

    bm25_rank: list[RetrievedChunk] = [
        RetrievedChunk(chunk=m.chunk, bm25_score=m.score, stage="bm25")
        for m in bm25.search(query, top_k=10)
    ]
    # 用词面重叠模拟一路"语义"排名（真实场景是向量检索）
    q_tokens = set(BM25Index.tokenize(query))
    vector_rank: list[RetrievedChunk] = []
    for c in chunks:
        overlap = len(q_tokens & set(BM25Index.tokenize(c.text))) / max(len(q_tokens), 1)
        vector_rank.append(RetrievedChunk(chunk=c, vector_score=overlap, stage="vector"))
    vector_rank.sort(key=lambda x: x.vector_score, reverse=True)

    print("BM25 top-3:")
    for i, r in enumerate(bm25_rank[:3], 1):
        print(f"  {i}. [{r.citation}] {r.chunk.text[:42]}...")
    print("向量 top-3:")
    for i, r in enumerate(vector_rank[:3], 1):
        print(f"  {i}. [{r.citation}] {r.chunk.text[:42]}...")

    fused = reciprocal_rank_fusion([vector_rank, bm25_rank])
    print("\nRRF 融合 top-3:")
    for i, r in enumerate(fused[:3], 1):
        print(f"  {i}. [{r.citation}] rrf={r.rrf_score:.4f} | {r.chunk.text[:42]}...")

    print("\nRRF 公式: score(d) = Σ 1/(k + rank_i(d))，k=60")
    print("为什么不用分数加权: 余弦(0~1) 与 BM25(0~几十) 量纲不同，")
    print("归一化后还要调 α，换数据就失效。RRF 只用排名，几乎免调参。")


def demo_rerank() -> None:
    section("4) Rerank：粗排召回 → 精排提精度")
    print("双塔(bi-encoder): 问题、文档分别编码 → 快、可预计算，但不互相看")
    print("交叉(cross-encoder): [CLS]问题[SEP]文档[SEP] 一起喂 → 慢，但深度交互，精度高 20~40%")
    print("实践: 混合检索取 top-50（要召回）→ 交叉重排取 top-5（要精度）")

    reranker = Reranker()
    print(f"\n当前 reranker: {'真·交叉编码 ' + reranker._model_name if reranker.available else '降级为词面重叠（未装 sentence-transformers）'}")

    bm25 = BM25Index()
    chunks = RecursiveChunker(chunk_size=200, overlap=20).split_documents(CORPUS)
    bm25.index(chunks)
    query = "住宿报销标准是多少？"
    candidates = [
        RetrievedChunk(chunk=m.chunk, bm25_score=m.score)
        for m in bm25.search(query, top_k=5)
    ]
    print(f"\n查询: {query!r}")
    print("重排前:")
    for i, r in enumerate(candidates, 1):
        print(f"  {i}. [{r.citation}] {r.chunk.text[:40]}...")

    reranked = reranker.rerank(query, candidates, top_k=3)
    print("重排后 top-3:")
    for i, r in enumerate(reranked, 1):
        print(f"  {i}. [{r.citation}] rerank={r.rerank_score:.4f} | {r.chunk.text[:40]}...")


def demo_end_to_end() -> None:
    section("5) 端到端 RAG 问答（真实调用 MiMo）")
    embedder = create_embedder(
        "BAAI/bge-small-zh-v1.5" if USE_REAL_EMBEDDING else "hashing-256"
    )
    print(f"Embedding: {embedder.model_name} (dim={embedder.dimension})")

    reranker = Reranker() if USE_REAL_EMBEDDING else None
    rag = RAGPipeline(llm=LLMClient(), embedder=embedder, reranker=reranker)

    n = rag.ingest_batch(CORPUS)
    print(f"入库完成: {n} 个分块，来源 {len(CORPUS)} 份文档\n")

    questions = [
        "报销有时间限制吗？",            # 语义问题，答案在《差旅报销制度》
        "RTX-4090-24G 是什么？",         # 精确编号问题
        "公司有几台服务器？",             # 知识库没有 —— 应答「资料中未提及」
    ]

    for q in questions:
        t0 = time.perf_counter()
        ans = rag.ask(q, top_k=3)
        dt = (time.perf_counter() - t0) * 1000

        print(f"Q: {q}")
        print(f"A: {ans.answer.strip()}")
        print(f"   检索证据: {ans.sources}")
        print(f"   引用合法: {'✅' if ans.citations_valid else '❌ ' + str(ans.citation_errors)}")
        print(f"   耗时 {dt:.0f}ms | tokens={ans.to_dict()['tokens']} | 成本=¥{ans.to_dict()['cost_cny']:.6f}")
        print()

    print("观察:")
    print("  · 答案带 [来源: 文件#片段] 引用 → 可溯源，用户能验证")
    print("  · 知识库没有的问题，模型应答「资料中未提及」→ 抑制幻觉")
    print("  · 引用合法性校验拦截模型编造的引用编号")


def main() -> None:
    demo_chunking()
    demo_retrieval_modes(_build_demo_pipeline())
    demo_hybrid()
    demo_rerank()
    demo_end_to_end()

    section("M2 小结")
    print("你已经跑通: 分块 → 向量化 → 入库 → 混合检索 → RRF → Rerank → 带引用生成")
    print("取舍:")
    print("  · 分块是第一决定因素，递归分割 + overlap 是通用起点")
    print("  · 向量管语义、BM25 管精确匹配，RRF 融合（不怕分数量纲不同）")
    print("  · 粗排要召回、精排要精度，双塔 vs 交叉编码")
    print("  · 换 embedding 模型必须重建全部索引")
    print("  · RAG 降低幻觉但不消除 → M7 用 RAGAS 量化 faithfulness")
    print("\n✅ M2 完成。下一步 M3: Agent 与 Function Calling。")


def _build_demo_pipeline() -> RAGPipeline:
    embedder = create_embedder("hashing-256")
    rag = RAGPipeline(llm=LLMClient(), embedder=embedder)
    rag.ingest_batch(CORPUS)
    return rag


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
