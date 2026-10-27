"""
M7 · 评估报告 CLI

运行:
    cd backend
    python -m knowflow.eval.report                      # 跑预置参数矩阵
    python -m knowflow.eval.report --configs all        # 跑全部预置
    python -m knowflow.eval.report --skip-ragas         # 只跑免费的 Simple
    python -m knowflow.eval.report --llm fake           # 离线冒烟，不联网不花钱

产出:
    eval_reports/<run_name>.md    可直接贴进简历/周报的量化报告
    EvalRun 表若干行              事后可横向对比不同版本

【为什么默认跑多套配置？】
  单跑一套只能得到四个数字，说服力有限。
  同一份金标准上跑「基线 / 换 embedding / 加 rerank / 调 top_k」，
  差值才是能写进简历的那句话——例如「加 CrossEncoder 精排后
  Context Precision 从 X 提到 Y」。

【为什么 ragas 只跑一套配置？】
  ragas 是 LLM-as-judge，25 条用例要打几十次判断题，慢且要花钱。
  报告的目的是「看 Simple 和 ragas 差多少」，不是给每套参数都打一遍分。
  所以默认只在生产配置上跑 ragas 做对照，其余配置跑免费的 Simple。
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import structlog

from knowflow.eval.dataset import EvalDataset, load_dataset, load_golden, load_golden_paraphrase
from knowflow.eval.metrics import EvalResult, RagasEvaluator, SimpleEvaluator
from knowflow.eval.runner import (
    DEFAULT_CONFIGS,
    PRESET_CONFIGS,
    EvalConfig,
    EvalRunResult,
    EvalRunner,
    FakeLLM,
    load_corpus,
)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# 按题型拆分 —— 计算题的 Faithfulness 天然偏低，混在一起会把总分拉失真
# ---------------------------------------------------------------------------
def _split_by_kind(run: EvalRunResult) -> dict[str, EvalResult]:
    """
    把一次运行的结果按 metadata.kind 拆成 factual / calc 两组。

    【为什么要拆？】
      计算题的标准答案里含推导出来的数字（如「1500 元 = 500×3」），
      这个数字在检索到的原文里不存在 —— Faithfulness 必然低分。
      但这不代表系统差：它是「推理」不是「幻觉」。
      不拆开看，计算题会把整体 Faithfulness 平均拉低，
      而那个数字既不是系统的问题，也不是能写进简历的结论。
    """
    buckets: dict[str, list] = {}
    for case, detail in zip(run.dataset, run.result.details):
        kind = str(case.metadata.get("kind", "factual"))
        buckets.setdefault(kind, []).append((case, detail))

    out: dict[str, EvalResult] = {}
    for kind, pairs in buckets.items():
        if not pairs:
            continue
        n = len(pairs)
        out[kind] = EvalResult(
            faithfulness=sum(d["faithfulness"] for _, d in pairs) / n,
            answer_relevancy=sum(d["answer_relevancy"] for _, d in pairs) / n,
            context_precision=sum(d["context_precision"] for _, d in pairs) / n,
            context_recall=sum(d["context_recall"] for _, d in pairs) / n,
            answer_correctness=sum(d["answer_correctness"] for _, d in pairs) / n,
            n_cases=n,
            details=[d for _, d in pairs],
            evaluator=run.result.evaluator,
            ragas_ran=run.result.ragas_ran,
        )
    return out


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------
def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def render_markdown(
    runs: list[EvalRunResult],
    dataset: EvalDataset,
    ragas_run: EvalRunResult | None,
    run_name: str,
    corpus_n: int,
) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# KnowFlow RAG 评估报告 · {run_name}")
    a("")
    a(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"- 金标准数据集：`{dataset.name}`，{len(dataset)} 条手写 QA")
    a(f"- 语料：{corpus_n} 份文档（随包发布于 `knowflow/eval/corpus/`）")
    a(f"- 参数组合：{len(runs)} 套")
    a("")

    # --- 主表：参数对比 -------------------------------------------------
    a("## 1. 参数对比")
    a("")
    a("同一份金标准上跑不同参数组合，差值即为该参数的贡献。")
    a("")
    a("| 配置 | embedding | rerank | top_k | Faithfulness | Relevancy | Prec | Recall | Correct | Overall | 耗时 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        c, m = r.config, r.result
        a(
            f"| `{c.name}` | `{c.embedder_model}` | {'是' if c.use_rerank else '否'} "
            f"| {c.top_k} | {_fmt(m.faithfulness)} | {_fmt(m.answer_relevancy)} "
            f"| {_fmt(m.context_precision)} | {_fmt(m.context_recall)} "
            f"| {_fmt(m.answer_correctness)} "
            f"| **{_fmt(m.overall)}** | {r.elapsed_s:.0f}s |"
        )
    a("")

    # 基线 → 生产必须显式定位这两个配置，不能用 runs[-1]。
    # 默认顺序是 [baseline, bge-m3, production, topk3]，runs[-1] 是 topk3，
    # 会把这个「rerank 贡献」小节错算成 top_k 的贡献。
    base = next((r for r in runs if r.config.name == "baseline"), None)
    prod = next((r for r in runs if r.config.name == "production"), None)
    if base is not None and prod is not None and base is not prod:
        a(f"**基线 `{base.config.name}` → 生产 `{prod.config.name}` 的变化**：")
        a("")
        for label, key in [
            ("Faithfulness", "faithfulness"),
            ("Answer Relevancy", "answer_relevancy"),
            ("Context Precision", "context_precision"),
            ("Context Recall", "context_recall"),
            ("Answer Correctness", "answer_correctness"),
            ("Overall", "overall"),
        ]:
            b, p = getattr(base.result, key), getattr(prod.result, key)
            delta = p - b
            a(f"- {label}: {b:.3f} → {p:.3f}（{'+' if delta >= 0 else ''}{delta:.3f}）")
        a("")

    # --- 按题型 ---------------------------------------------------------
    a("## 2. 按题型拆分")
    a("")
    a(
        "计算题的标准答案含推导出来的数字，这些数字在原文里不存在，"
        "Faithfulness 天然偏低——那是「推理」不是「幻觉」。拆开看才不冤枉检索层。"
    )
    a("")
    for r in runs:
        a(f"### `{r.config.name}`")
        a("")
        a("| 题型 | 条数 | Faithfulness | Relevancy | Prec | Recall | Correct |")
        a("|---|---|---|---|---|---|---|")
        for kind, m in sorted(_split_by_kind(r).items()):
            label = {"factual": "事实题", "calc": "计算题"}.get(kind, kind)
            a(f"| {label} | {m.n_cases} | {_fmt(m.faithfulness)} | {_fmt(m.answer_relevancy)} "
              f"| {_fmt(m.context_precision)} | {_fmt(m.context_recall)} "
              f"| {_fmt(m.answer_correctness)} |")
        a("")

    # --- 双评估器对比 ---------------------------------------------------
    a("## 3. Simple vs Ragas 双评估器")
    a("")
    if ragas_run is None:
        a("本次未运行 ragas（`--skip-ragas`）。只跑了 SimpleEvaluator。")
        a("")
    else:
        simple_run = next((x for x in runs if x.config.name == ragas_run.config.name), None)
        rm = ragas_run.result
        a(f"在配置 `{ragas_run.config.name}` 上同时跑两个评估器。")
        a("")
        a(f"- ragas 是否真正运行：**{'是' if rm.ragas_ran else '否'}**")
        if not rm.ragas_ran:
            a(f"- 降级原因：`{rm.fallback_reason}`")
        else:
            a(f"- 综合评估器标签：`{rm.evaluator}`")
            a(f"- 各指标来源：{rm.metrics_source}")
        a("")
        if simple_run is not None:
            sm = simple_run.result
            a("| 指标 | Simple | Ragas | 差值 |")
            a("|---|---|---|---|")
            for label, key in [
                ("Faithfulness", "faithfulness"),
                ("Answer Relevancy", "answer_relevancy"),
                ("Context Precision", "context_precision"),
                ("Context Recall", "context_recall"),
                ("Answer Correctness", "answer_correctness"),
                ("Overall", "overall"),
            ]:
                s, g = getattr(sm, key), getattr(rm, key)
                a(f"| {label} | {s:.3f} | {g:.3f} | {g - s:+.3f} |")
            a("")
            a(
                "> 两个评估器的口径不同：Simple 用关键词匹配，ragas 用 LLM 判断语义蕴含。"
                "Simple 会漏掉「换了说法」的正确答案（低估），也会放过「关键词撞上但语义不符」的答案（高估）。"
                "差值本身不是谁准谁不准的结论，而是说明**单看一个数字不够**。"
            )
            a("")

    # --- 最差用例 -------------------------------------------------------
    a("## 4. 最差用例（按 Faithfulness 升序）")
    a("")
    # 优先看生产配置：最差用例要在「最终要上线的那套参数」上找
    worst = next((r for r in runs if r.config.name == "production"), runs[-1])
    ranked = sorted(worst.result.details, key=lambda d: d["faithfulness"])[:5]
    a(f"配置 `{worst.config.name}` 上的最差 5 条：")
    a("")
    a("| Faith | Relev | Prec | Recall | 问题 |")
    a("|---|---|---|---|---|")
    for d in ranked:
        q = d["question"][:34] + ("…" if len(d["question"]) > 34 else "")
        a(f"| {d['faithfulness']:.3f} | {d['answer_relevancy']:.3f} | {d['context_precision']:.3f} "
          f"| {d['context_recall']:.3f} | {q} |")
    a("")

    # --- 边界 -----------------------------------------------------------
    a("## 5. 边界与口径")
    a("")
    a("- **评估对象是 `RAGPipeline.ask()`**，即「检索 + 生成」核心，不含 Router 分流与 Agent 工具循环。"
      "融合链路的 `retrieved` 是截断摘要，拿它算 Context Recall 会把「检索到了但被截断」误判成漏检。")
    a("- **Faithfulness 对计算题无意义**：答案里的数字是算出来的，不在原文里。看第 2 节的分组数据。")
    a("- **Context Precision / Recall 依赖 `expected_facts` 为原文逐字短语**。"
      "写金标准时若事实点跨分块边界或改写过，会低估检索质量。")
    a("- **ragas 未传 embeddings 时 `answer_relevancy` 走 Simple**（ragas 该项需要 embedding 服务），"
      "报告里按 `metrics_source` 如实标注，不冒充 ragas 分数。")
    a("- **样本量 25 条**，结论是方向性的，不是统计显著的。加样本才够硬。")
    a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvalRun 落库
# ---------------------------------------------------------------------------
def persist_run(run_name: str, dataset_name: str, run: EvalRunResult) -> str | None:
    """存一行 EvalRun，方便以后「改了参数再跑一次」横向对比。"""
    try:
        from knowflow.db.models import EvalRun
        from knowflow.db.session import session_scope

        with session_scope() as s:
            row = EvalRun(
                run_name=run_name,
                dataset=dataset_name,
                metrics=run.result.to_dict(),
                config=run.config.to_dict(),
            )
            s.add(row)
            return row.id
    except Exception as exc:  # 落库失败不该让整份报告作废
        log.warning("evalrun_persist_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _resolve_configs(spec: str) -> list[EvalConfig]:
    if spec in ("all", "*"):
        return [PRESET_CONFIGS[k] for k in DEFAULT_CONFIGS + ["topk10"] if k in PRESET_CONFIGS]
    out = []
    for name in [x.strip() for x in spec.split(",") if x.strip()]:
        if name not in PRESET_CONFIGS:
            raise SystemExit(f"未知配置 '{name}'，可选: {', '.join(PRESET_CONFIGS)}")
        out.append(PRESET_CONFIGS[name])
    return out


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK，报告里大量中文会显示成乱码。
    # 报告文件本身是 UTF-8，这里只把终端也对齐，失败了不影响出报告。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="KnowFlow RAG 评估报告")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                        help=f"参数组合，逗号分隔或 all。可选: {', '.join(PRESET_CONFIGS)}")
    parser.add_argument("--dataset", default=None, help="金标准 JSONL 路径（默认随包 golden.jsonl）")
    parser.add_argument("--paraphrase", action="store_true",
                        help="用近义改写版金标准（golden_paraphrase.jsonl），问题不逐字命中原文")
    parser.add_argument("--corpus", default=None, help="种子语料目录（默认随包 corpus/）")
    parser.add_argument("--with-distractors", action="store_true",
                        help="并入 corpus/distractors/ 下的干扰文档，让评测更难、更有区分度")
    parser.add_argument("--out", default="eval_reports", help="报告输出目录")
    parser.add_argument("--ragas-on", default="production",
                        help="在哪个配置上跑 ragas 做双评估器对照（默认 production）")
    parser.add_argument("--skip-ragas", action="store_true", help="只跑免费的 SimpleEvaluator")
    parser.add_argument("--llm", choices=["real", "fake"], default="real",
                        help="fake = 离线冒烟，不联网不花钱")
    parser.add_argument("--no-db", action="store_true", help="不写 EvalRun 表")
    args = parser.parse_args(argv)

    if args.dataset:
        dataset: EvalDataset = load_dataset(args.dataset)
    elif args.paraphrase:
        dataset: EvalDataset = load_golden_paraphrase()
    else:
        dataset: EvalDataset = load_golden()
    configs = _resolve_configs(args.configs)
    corpus_n = len(load_corpus(args.corpus, include_distractors=args.with_distractors))
    run_name = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    llm = FakeLLM() if args.llm == "fake" else None
    runner = EvalRunner(llm=llm)

    print(f"\n数据集 {dataset.summary()}   语料 {corpus_n} 份   配置 {len(configs)} 套")
    print(f"运行标识 {run_name}\n")

    runs: list[EvalRunResult] = []
    for cfg in configs:
        print(f"→ {cfg.name:<12} embedder={cfg.embedder_model}  rerank={cfg.use_rerank}  top_k={cfg.top_k}")
        run = runner.run(cfg, dataset, evaluator=SimpleEvaluator(), corpus_dir=args.corpus, include_distractors=args.with_distractors)
        runs.append(run)
        print(f"  {run.result.summary()}   ({run.elapsed_s:.0f}s)\n")

        if not args.no_db:
            persist_run(run_name, dataset.name, run)

    # ragas 对照：只跑一套，省时间也省钱
    ragas_run: EvalRunResult | None = None
    if not args.skip_ragas:
        target = next((c for c in configs if c.name == args.ragas_on), None)
        if target is None:
            print(f"跳过 ragas：配置 '{args.ragas_on}' 不在本次运行的配置里")
        elif args.llm == "fake":
            print("跳过 ragas：--llm fake 时不做 LLM-as-judge（没有真裁判）")
        else:
            print(f"→ ragas 对照（配置 {target.name}，会调 LLM 当裁判，产生少量费用）")
            try:
                # 关键：RagasEvaluator(llm_client=...) 才会把 MiMo 包成 judge。
                # 传 None 时 ragas 会去找 OPENAI_API_KEY，本环境只有 MIMO_API_KEY，
                # 结果永远降级成 Simple，ragas 对照形同虚设。
                ragas_run = runner.run(
                    target,
                    dataset,
                    evaluator=RagasEvaluator(llm_client=runner.llm),
                    corpus_dir=args.corpus,
                    include_distractors=args.with_distractors,
                )
                print(f"  {ragas_run.result.summary()}\n")
                if not args.no_db:
                    persist_run(f"{run_name}-ragas", dataset.name, ragas_run)
            except Exception as exc:
                print(f"  ragas 运行失败：{type(exc).__name__}: {exc}\n")

    md = render_markdown(runs, dataset, ragas_run, run_name, corpus_n)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_name}.md"
    out_path.write_text(md, encoding="utf-8")

    print("=" * 64)
    print(md)
    print("=" * 64)
    print(f"\n报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
