"""
M7 · 评估体系测试

【测什么？】
  1. SimpleEvaluator 四指标的算法 —— 用可手算的输入锁定预期分数
  2. 金标准数据集 —— 25 条、题型分布、以及**引用的文档必须真的存在**
  3. 评估执行器 —— FakeLLM + hashing-256 跑通「入库→检索→生成→打分」全链
  4. 降级留痕 —— ragas 没跑起来时必须写明原因，不能冒充 ragas 分数

【为什么全离线？】
  这些是 CI 回归守卫，要能在没网、没 API key、没装 ragas 的环境里跑。
  真调 MiMo / ragas 的那部分是「花钱的事」，放报告 CLI（report.py），
  不放测试。测试只锁契约，不锁效果。

【为什么不测指标「准不准」？】
  指标准不准是 ragas vs Simple 的对照问题，看报告第 3 节。
  这里测的是「算法没被改坏」——同样的输入还出同样的分数。
"""

from __future__ import annotations

import os
from pathlib import Path

# RAGPipeline 构造时会 get_settings()，而 mimo_api_key 是必填字段。
# 离线冒烟不需要真 key，先兜一个占位，避免没配 .env 的环境直接炸。
os.environ.setdefault("MIMO_API_KEY", "sk-offline-smoke-not-a-real-key")

import pytest

from knowflow.eval.dataset import (
    CORPUS_DIR,
    DISTRACTOR_DIR,
    GOLDEN_PATH,
    EvalCase,
    EvalDataset,
    load_dataset,
    load_golden,
    load_golden_paraphrase,
    save_dataset,
)
from knowflow.eval.metrics import EvalResult, RagasEvaluator, SimpleEvaluator, _ensure_ragas_importable, _nanmean
from knowflow.eval.runner import (
    DEFAULT_CONFIGS,
    PRESET_CONFIGS,
    EvalConfig,
    EvalRunner,
    FakeLLM,
    clone_dataset,
    load_corpus,
)


# ---------------------------------------------------------------------------
# 1. SimpleEvaluator 算法回归
# ---------------------------------------------------------------------------
class TestSimpleEvaluator:
    def test_faithfulness_supported_and_hallucinated(self):
        """答案里的信息能从 context 找到 → 1.0；编出来的 → 0.0。"""
        ev = SimpleEvaluator()
        ctx = ["一线城市每晚500元，二线城市每晚350元。"]

        assert ev._faithfulness("一线城市每晚500元。", ctx) == 1.0
        assert ev._faithfulness("报销上限是999元。", ctx) == 0.0

    def test_faithfulness_empty_inputs(self):
        ev = SimpleEvaluator()
        assert ev._faithfulness("", ["some context"]) == 0.0
        assert ev._faithfulness("有答案但没上下文", []) == 0.0

    def test_answer_relevancy_keyword_overlap(self):
        """问题关键词出现在答案里的比例。"""
        ev = SimpleEvaluator()
        q = "开发机 型号 是 什么？"
        assert ev._answer_relevancy(q, "开发机型号是 ThinkPad-X1。") == 1.0
        assert ev._answer_relevancy(q, "显示器是 27 英寸。") == 0.0

    def test_answer_relevancy_unspaced_chinese(self):
        """无空格中文问题（金标准的真实形态）不能把整句当成一个词。"""
        ev = SimpleEvaluator()
        q = "一线城市的住宿标准是多少？"
        good = ev._answer_relevancy(q, "一线城市住宿标准为每晚 500 元。")
        bad = ev._answer_relevancy(q, "显示器是 27 英寸的。")
        assert good > 0.0
        assert good > bad

    def test_answer_correctness_ground_truth_overlap(self):
        """Correctness 近似：标准答案的关键信息有多少出现在生成答案里。"""
        ev = SimpleEvaluator()
        gt = "一线城市每晚500元，二线城市每晚350元。"
        good = ev._answer_correctness(gt, "一线城市每晚 500 元。")
        bad = ev._answer_correctness(gt, "显示器是 27 英寸的。")
        assert good > 0.0
        assert good > bad
        assert ev._answer_correctness(gt, gt) == 1.0

    def test_context_precision_rank_weighted(self):
        """相关 chunk 排越前面分越高：命中的在第 2 位 → 0.5。"""
        ev = SimpleEvaluator()
        contexts = ["完全无关的文本", "一线城市每晚500元"]
        facts = ["一线城市每晚500元"]
        # P@2 = 1/2 = 0.5，除以相关块数 1 → 0.5
        assert ev._context_precision(contexts, facts) == pytest.approx(0.5)

    def test_context_precision_perfect_ranking_is_one(self):
        """
        相关块排第 1 位 = 完美排序，必须得满分。

        【这条守卫的是什么？】
          分母若写成 len(contexts)，top_k=5 时完美排序也只能拿 0.2，
          整张参数对比表会把「排序没问题」显示成「检索很差」。
        """
        ev = SimpleEvaluator()
        facts = ["一线城市每晚500元"]
        assert ev._context_precision(["一线城市每晚500元", "无关", "无关"], facts) == pytest.approx(1.0)
        # 没有任何相关块 → 0，不能除零
        assert ev._context_precision(["无关", "无关"], facts) == 0.0

    def test_context_recall_fact_coverage(self):
        """预期事实有几个被检索到。2 个里检索到 1 个 → 0.5。"""
        ev = SimpleEvaluator()
        contexts = ["一线城市每晚500元"]
        facts = ["一线城市每晚500元", "二线城市每晚350元"]
        assert ev._context_recall(contexts, facts) == pytest.approx(0.5)

    def test_result_marks_simple_as_source(self):
        """Simple 跑出来的结果必须自称 simple，不许冒充 ragas。"""
        ds = EvalDataset(
            name="t",
            cases=[
                EvalCase(
                    question="开发机 型号 是 什么？",
                    ground_truth="ThinkPad",
                    answer="开发机型号是 ThinkPad。",
                    contexts=["开发机型号为 ThinkPad-X1。"],
                    expected_facts=["ThinkPad-X1"],
                )
            ],
        )
        result = SimpleEvaluator().evaluate(ds)
        assert result.evaluator == "simple"
        assert result.ragas_ran is False
        assert result.n_cases == 1
        assert result.metrics_source["faithfulness"] == "simple"
        assert result.metrics_source["context_recall"] == "simple"
        assert result.metrics_source["answer_correctness"] == "simple"


# ---------------------------------------------------------------------------
# 2. 金标准数据集
# ---------------------------------------------------------------------------
class TestGoldenDataset:
    def test_golden_file_exists_and_has_25_cases(self):
        assert GOLDEN_PATH.exists(), f"金标准缺失: {GOLDEN_PATH}"
        assert len(load_golden()) == 25

    def test_kind_distribution(self):
        """20 条事实题 + 5 条计算题，题型标签必须齐全。"""
        ds = load_golden()
        kinds = [c.metadata.get("kind") for c in ds]
        assert kinds.count("factual") == 20
        assert kinds.count("calc") == 5

    def test_every_case_has_expected_facts(self):
        """没有 expected_facts 就算不出 Context Recall，等于这条没做评估。"""
        for case in load_golden():
            assert case.expected_facts, f"缺 expected_facts: {case.question}"
            assert case.ground_truth, f"缺 ground_truth: {case.question}"

    def test_referenced_documents_actually_exist(self):
        """
        金标准引用的每份文档都必须在语料里。

        【这条守卫的是什么？】
          M7 动工前 build_demo_dataset() 引用的「差旅报销制度.pdf」等 4 份文档
          在仓库里根本不存在 —— 评估跑出来的数字全部建立在空语料上。
          这类问题靠读代码发现不了，只能用测试钉死。
        """
        names = {p.name for p in CORPUS_DIR.glob("*.md")}
        assert names, f"语料目录为空: {CORPUS_DIR}"
        for case in load_golden():
            assert case.source in names, f"{case.question!r} 引用了不存在的文档 {case.source!r}"

    def test_every_source_is_covered_by_questions(self):
        """每份语料至少要有 3 条 QA，否则那份文档没被真正评估到。"""
        counts: dict[str, int] = {}
        for case in load_golden():
            counts[case.source] = counts.get(case.source, 0) + 1
        for name, n in counts.items():
            assert n >= 3, f"{name} 只有 {n} 条 QA，覆盖不足"

    def test_jsonl_roundtrip_preserves_metadata(self):
        """kind 标签要能往返 —— 报告按题型拆分靠它。"""
        ds = load_golden()
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "roundtrip.jsonl"
            save_dataset(ds, out)
            back = load_dataset(out)
        assert len(back) == len(ds)
        assert [c.metadata.get("kind") for c in back] == [c.metadata.get("kind") for c in ds]
        assert [c.question for c in back] == [c.question for c in ds]

    def test_paraphrase_golden_matches_verbatim_shape(self):
        """近义版只改问题措辞，source/kind/expected_facts 必须和逐字版一一对应。"""
        verbatim = load_golden()
        para = load_golden_paraphrase()
        assert len(para) == len(verbatim) == 25
        assert [c.source for c in para] == [c.source for c in verbatim]
        assert [c.metadata.get("kind") for c in para] == [c.metadata.get("kind") for c in verbatim]
        assert [c.expected_facts for c in para] == [c.expected_facts for c in verbatim]
        # 问题确实被改写了（至少一部分和原文不同），否则近义版没有意义
        assert any(c.question != v.question for c, v in zip(para, verbatim))


# ---------------------------------------------------------------------------
# 3. 语料加载
# ---------------------------------------------------------------------------
def test_load_corpus_returns_five_docs():
    docs = load_corpus()
    assert len(docs) == 5
    for text, source in docs:
        assert text.strip(), f"{source} 是空的"
        assert source.endswith(".md")


def test_load_corpus_include_distractors_adds_noise():
    """干扰语料是可选的额外语料：默认不加，显式开启后要能并入。"""
    base = load_corpus()
    full = load_corpus(include_distractors=True)
    assert len(base) == 5
    assert len(full) > len(base), "开了 include_distractors 却没加进干扰文档"
    assert DISTRACTOR_DIR.is_dir(), f"干扰语料目录缺失: {DISTRACTOR_DIR}"
    for text, source in full:
        assert text.strip(), f"{source} 是空的"
        assert source.endswith(".md")


# ---------------------------------------------------------------------------
# 4. 执行器全链（离线）
# ---------------------------------------------------------------------------
class TestEvalRunnerOffline:
    def test_clone_dataset_isolates_runs(self):
        """
        每套参数必须拿到干净数据集。

        不拷贝的话，第二套配置会看到第一套残留的 answer/contexts，
        对比出来的数字是假的 —— 而这类 bug 不报错，只是数字悄悄变高。
        """
        src = EvalDataset(
            name="t",
            cases=[EvalCase(question="q", ground_truth="g", answer="残留答案", contexts=["残留"])],
        )
        cloned = clone_dataset(src)
        assert cloned[0].answer == ""
        assert cloned[0].contexts == []
        # 源数据集不能被改
        assert src[0].answer == "残留答案"

    def test_preset_configs_have_distinct_collections(self):
        """不同 embedding 维度绝不能共用 collection，否则写入时维度校验就炸。"""
        collections = {c.collection for c in PRESET_CONFIGS.values()}
        assert len(collections) == len(PRESET_CONFIGS)

    def test_run_fills_answer_and_full_contexts(self):
        """
        离线跑通「入库 → 检索 → 生成 → 打分」全链。

        这里用 FakeLLM + hashing-256：不下载模型、不联网、不花钱，
        但分块、向量化、混合检索、打分全是真代码。
        """
        golden = load_golden()
        # 用标准答案当假 LLM 的回答 → Faithfulness 应该很高（答案来自原文）
        fake = FakeLLM({c.question: c.ground_truth for c in golden})
        runner = EvalRunner(llm=fake)

        cfg = EvalConfig(name="offline", embedder_model="hashing-256", use_rerank=False, top_k=3)
        run = runner.run(cfg, golden)

        assert run.n_ingested > 0, "语料没入库"
        assert run.result.n_cases == 25
        for case in run.dataset:
            assert case.answer, f"没生成答案: {case.question}"
            assert isinstance(case.contexts, list)
            # contexts 必须是完整 chunk 文本，不是 API 层截断的 snippet
            for ctx in case.contexts:
                assert len(ctx) > 0

        # 检索层至少该命中一部分 —— 全 0 说明检索链路坏了
        assert run.result.context_recall > 0.0, "一条都没检索到，检索链路有问题"
        # FakeLLM 直接回灌标准答案，Faithfulness 该接近满分
        assert run.result.faithfulness > 0.5, f"回灌标准答案仍 Faithfulness={run.result.faithfulness}"

    def test_run_retrieval_only_needs_no_llm_answer(self):
        """检索层评估不生成答案，用来快速调 top_k。"""
        golden = load_golden()
        runner = EvalRunner(llm=FakeLLM())
        cfg = EvalConfig(name="retr-only", embedder_model="hashing-256", top_k=3)
        result = runner.run_retrieval_only(cfg, golden)
        assert result.n_cases == 25
        assert result.context_recall > 0.0

    def test_default_configs_cover_three_axes(self):
        """默认配置必须覆盖 embedding / rerank / top_k 三个对比轴。"""
        named = {c.name: c for c in PRESET_CONFIGS.values()}
        embedders = {named[n].embedder_model for n in DEFAULT_CONFIGS}
        rerank_vals = {named[n].use_rerank for n in DEFAULT_CONFIGS}
        topks = {named[n].top_k for n in DEFAULT_CONFIGS}
        assert len(embedders) >= 2, f"embedding 轴没拉开: {embedders}"
        assert rerank_vals == {True, False}, f"rerank 轴没拉开: {rerank_vals}"
        assert len(topks) >= 2, f"top_k 轴没拉开: {topks}"


# ---------------------------------------------------------------------------
# 5. 降级必须留痕
# ---------------------------------------------------------------------------
class TestRagasFallbackSurfacing:
    def test_fallback_is_not_mistaken_for_ragas(self):
        """
        ragas 没跑起来时，结果不许自称 ragas。

        【这条守卫的是什么？】
          旧版捕获异常后静默 fallback 到 SimpleEvaluator，报告里写的
          「RAGAS 0.82」其实可能是关键词匹配的 0.82。简历项目经不起这么问。
        """
        ev = RagasEvaluator()

        def boom(_dataset):
            raise RuntimeError("simulated: ragas not installed")

        ev._evaluate_with_ragas = boom  # 模拟 ragas 不可用

        ds = EvalDataset(name="t", cases=[EvalCase(question="q", ground_truth="g")])
        result = ev.evaluate(ds)

        assert result.ragas_ran is False, "降级了却自称跑过 ragas"
        assert result.evaluator == "simple"
        assert "simulated" in result.fallback_reason, "降级原因没写出来"
        assert result.metrics_source["faithfulness"] == "simple"

    def test_simple_result_never_claims_ragas(self):
        result = SimpleEvaluator().evaluate(EvalDataset(name="t", cases=[]))
        assert result.ragas_ran is False
        assert result.evaluator == "simple"
        # to_dict 里也要带上来源，否则落库后追溯不了
        d = result.to_dict()
        assert d["ragas_ran"] is False
        assert d["metrics_source"]

    def test_ragas_importable_after_shim(self):
        """
        ragas 0.4.x 顶层 import 依赖已删除的 langchain_community.chat_models.vertexai，
        占位补丁要能兜住，否则整个 ragas 进不来、LLM-as-judge 形同虚设。
        """
        import importlib.util

        if importlib.util.find_spec("ragas") is None:
            pytest.skip("ragas 未安装")
        _ensure_ragas_importable()
        import ragas

        assert ragas.__version__

    def test_nanmean_skips_nan(self):
        """ragas 超时返回的 NaN 不能污染整个指标，聚合时跳过。"""
        assert _nanmean([0.5, 0.7]) == pytest.approx(0.6)
        assert _nanmean([0.5, float("nan"), 0.7]) == pytest.approx(0.6)

    def test_to_dict_sanitizes_nan(self):
        """NaN 不能进 JSON（落库会炸），to_dict 转成 None。"""
        r = EvalResult(faithfulness=float("nan"), answer_relevancy=0.5)
        d = r.to_dict()
        assert d["faithfulness"] is None
        assert d["answer_relevancy"] == 0.5

    def test_empty_dataset_does_not_divide_by_zero(self):
        result = SimpleEvaluator().evaluate(EvalDataset(name="empty"))
        assert result.n_cases == 0
        assert result.overall == 0.0
