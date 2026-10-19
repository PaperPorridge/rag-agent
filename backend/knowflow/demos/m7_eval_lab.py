"""
M7 演示：RAG 评估实验室（指标计算 / 评估报告 / 参数对比）

运行:
    cd backend
    python -m knowflow.demos.m7_eval_lab

本演示覆盖:
  1. 构建评估数据集      —— 5 条标注 QA 对
  2. 跑 RAG 生成答案      —— RAGPipeline 生成 answer + contexts
  3. 计算四个指标         —— Faithfulness / Relevancy / Precision / Recall
  4. 评估报告             —— 哪条差、为什么差
  5. 参数对比             —— 开关 Rerank，看指标变化
  6. 存入 EvalRun 表      —— 可追溯、可对比不同版本

环境说明: 用 SimpleEvaluator（关键词匹配，零依赖不花钱）。
          正式评估用 RagasEvaluator（LLM-as-judge，需要 API key）。
          两者跑同一批数据可对比差距。
"""

from __future__ import annotations

import time

from knowflow.eval import SimpleEvaluator, build_demo_dataset
from knowflow.rag.pipeline import RAGPipeline
from knowflow.rag.chunking import RecursiveChunker


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


# 语料（和 m2_rag_lab 一致）
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
        "显卡扩展坞型号为 RTX-4090-24G，申请需走IT工单。",
        "IT设备清单.md",
    ),
    (
        "年假规则：入职满一年享有10天年假，满三年15天。"
        "年假需提前3个工作日在OA系统申请，连续休假超过5天需总监审批。",
        "年假规则.md",
    ),
]


# ---------------------------------------------------------------------------
# 1. 构建评估数据集
# ---------------------------------------------------------------------------
def demo_dataset():
    section("1. 构建评估数据集 —— 没有测试集 = 没法衡量效果")

    dataset = build_demo_dataset()
    print(f"  {dataset.summary()}")
    for i, case in enumerate(dataset, 1):
        print(f"    [{i}] Q: {case.question}")
        print(f"        A: {case.ground_truth[:40]}...")
        print(f"        事实: {case.expected_facts}")

    print()
    print("  [OK] 有 ground_truth 才能算 Faithfulness，有 expected_facts 才能算 Recall")
    return dataset


# ---------------------------------------------------------------------------
# 2. 跑 RAG 生成答案
# ---------------------------------------------------------------------------
def demo_generate(dataset):
    section("2. 跑 RAG 系统 —— 生成 answer + contexts")

    rag = RAGPipeline(chunker=RecursiveChunker(chunk_size=512, overlap=64))
    rag.ingest_batch(CORPUS)

    for case in dataset:
        result = rag.ask(case.question, top_k=3)
        case.answer = result.answer
        case.contexts = [r.chunk.text for r in result.retrieved]

    print(f"  {len(dataset)} 条问题全部生成完毕")
    sample = dataset[0]
    print(f"\n  示例:")
    print(f"    Q: {sample.question}")
    print(f"    A: {sample.answer[:60]}...")
    print(f"    contexts: {len(sample.contexts)} 条")

    print("  [OK] answer + contexts 是评估的输入")
    return rag


# ---------------------------------------------------------------------------
# 3. 计算四个指标
# ---------------------------------------------------------------------------
def demo_evaluate(dataset):
    section("3. 计算四个指标 —— 量化 RAG 效果")

    evaluator = SimpleEvaluator()
    result = evaluator.evaluate(dataset)

    print(f"  {result.summary()}")
    print(f"\n  四个指标含义:")
    print(f"    Faithfulness      {result.faithfulness:.2f}  答案有没有编（幻觉检测）")
    print(f"    Answer Relevancy  {result.answer_relevancy:.2f}  答案和问题相关吗")
    print(f"    Context Precision {result.context_precision:.2f}  相关的排前面了吗")
    print(f"    Context Recall    {result.context_recall:.2f}  该检索的检索到了吗")

    print("  [OK] 四个数字 = RAG 系统的体检报告")
    return result


# ---------------------------------------------------------------------------
# 4. 评估报告 —— 哪条差、为什么差
# ---------------------------------------------------------------------------
def demo_report(dataset, result):
    section("4. 评估报告 —— 找出最差的用例")

    # 按 faithfulness 排序，找最差的
    sorted_details = sorted(result.details, key=lambda x: x["faithfulness"])

    print(f"  按 Faithfulness 排序（差的在前）:")
    for d in sorted_details[:3]:
        bar = "  " * 0
        print(f"    F={d['faithfulness']:.2f}  R={d['answer_relevancy']:.2f}  "
              f"P={d['context_precision']:.2f}  C={d['context_recall']:.2f}  "
              f"Q: {d['question'][:25]}")

    worst = sorted_details[0]
    if worst["faithfulness"] < 0.8:
        print(f"\n  最差用例: {worst['question']}")
        print(f"    Faithfulness={worst['faithfulness']} → 答案里有 context 没有的内容")
        print(f"    可能原因: 分块切断了关键信息 / 检索到的不相关 / 模型编造")
    else:
        print(f"\n  所有用例 Faithfulness >= 0.8，表现不错")

    print("  [OK] 评估报告 = 定位问题，不只是一个数字")


# ---------------------------------------------------------------------------
# 5. 参数对比 —— 开关 Rerank
# ---------------------------------------------------------------------------
def demo_compare(dataset):
    section("5. 参数对比 —— Rerank 开/关对指标的影响")

    rag = RAGPipeline(chunker=RecursiveChunker(chunk_size=512, overlap=64))
    rag.ingest_batch(CORPUS)

    evaluator = SimpleEvaluator()

    # 关闭 Rerank
    for case in dataset:
        result = rag.ask(case.question, top_k=3, use_rerank=False)
        case.answer = result.answer
        case.contexts = [r.chunk.text for r in result.retrieved]
    no_rerank = evaluator.evaluate(dataset)

    # 开启 Rerank
    for case in dataset:
        result = rag.ask(case.question, top_k=3, use_rerank=True)
        case.answer = result.answer
        case.contexts = [r.chunk.text for r in result.retrieved]
    with_rerank = evaluator.evaluate(dataset)

    print(f"  {'指标':<20} {'无 Rerank':>10} {'有 Rerank':>10} {'变化':>10}")
    print(f"  {'-' * 52}")
    for name, no_val, with_val in [
        ("Faithfulness", no_rerank.faithfulness, with_rerank.faithfulness),
        ("Answer Relevancy", no_rerank.answer_relevancy, with_rerank.answer_relevancy),
        ("Context Precision", no_rerank.context_precision, with_rerank.context_precision),
        ("Context Recall", no_rerank.context_recall, with_rerank.context_recall),
        ("Overall", no_rerank.overall, with_rerank.overall),
    ]:
        delta = with_val - no_val
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<20} {no_val:>10.2f} {with_val:>10.2f} {sign}{delta:>9.2f}")

    print(f"\n  [OK] 用数字说话: 「上 Rerank 后 Context Precision 提升了 X」")


# ---------------------------------------------------------------------------
# 6. 存入 EvalRun 表
# ---------------------------------------------------------------------------
def demo_persist(dataset, result):
    section("6. 存入 EvalRun 表 —— 可追溯、可对比")

    try:
        from knowflow.db.models import Base, EvalRun
        from knowflow.db.session import get_engine, session_scope

        engine = get_engine()
        Base.metadata.create_all(engine)

        with session_scope() as s:
            run = EvalRun(
                run_name=f"eval-{int(time.time())}",
                dataset="knowflow-demo-v1",
                metrics=result.to_dict(),
                config={"chunk_size": 512, "rerank": True, "evaluator": "simple"},
            )
            s.add(run)
            run_id = run.id

        # 查出来验证
        from sqlalchemy import select
        with session_scope() as s:
            runs = s.execute(select(EvalRun)).scalars().all()
            for r in runs:
                print(f"  EvalRun[{r.id[:8]}] {r.run_name} → {r.metrics}")

        print(f"\n  [OK] 评估结果存到 PG，改参数后跑一次就能对比两版的指标")
    except Exception as exc:
        print(f"  [跳过 DB 存储] {type(exc).__name__}: {exc}")
        print("  （演示模式，不影响评估结果）")


# ---------------------------------------------------------------------------
async def main() -> None:
    print(__doc__)

    dataset = demo_dataset()
    rag = demo_generate(dataset)
    result = demo_evaluate(dataset)
    demo_report(dataset, result)
    demo_compare(dataset)
    demo_persist(dataset, result)

    section("完成")
    print("  概念回顾:")
    print("    · 四指标      Faithfulness(幻觉) Relevancy(相关) Precision(排序) Recall(覆盖)")
    print("    · 评估数据集  人工标注(最贵最准) / 半自动(最常用) / 线上日志(最值钱)")
    print("    · LLM-as-judge  拆陈述句 → 逐句判断能否从文档推出 → 算比例")
    print("    · SimpleEvaluator  关键词匹配，免费快速，适合 CI")
    print("    · RagasEvaluator   LLM 判断，准确但花钱，适合正式报告")
    print("    · 参数对比    用数字说话，不靠感觉")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
