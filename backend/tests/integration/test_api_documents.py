"""
S6 · 文档入库接口集成测试

【测什么？】
  1. 入库闭环：POST → 轮询至 ready 且 chunk_count > 0（守护「状态写回 + 进程内 Worker」）
  2. 列表/删除：上传后 GET /documents 含该行；DELETE → 200；GET → 404
  3. PDF + 引用富化：上传 sample.pdf → ask → retrieved 非空且字段完整

【为什么要轮询而不是 sleep 等一下？】
  Worker 是异步的，入库耗时随 embedding 变化（CPU 上 bge-m3 秒级）。
  固定 sleep 要么太短（假红）要么太长（白等）。轮询到超时才失败，稳。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowflow.api.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
POLL_INTERVAL = 0.5
POLL_TIMEOUT = 60.0  # bge-m3 CPU 向量化 + 首次下载都要留余量


@pytest.fixture(scope="module")
def client():
    """模块级 TestClient：lifespan 只启停一次 Worker，省重复建模型的开销。"""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _wait_ready(client: TestClient, doc_id: str) -> dict:
    """轮询直到 ready/failed，返回最终状态。超时则断言失败。"""
    deadline = time.time() + POLL_TIMEOUT
    last: dict = {}
    while time.time() < deadline:
        r = client.get(f"/api/v1/documents/{doc_id}")
        assert r.status_code == 200, r.text
        last = r.json()
        if last["status"] in ("ready", "failed"):
            return last
        time.sleep(POLL_INTERVAL)
    pytest.fail(f"文档 {doc_id} 未在 {POLL_TIMEOUT}s 内就绪，最后状态: {last}")


# ---------------------------------------------------------------------------
# 1. 入库闭环
# ---------------------------------------------------------------------------
def test_ingest_text_reaches_ready_with_chunks(client: TestClient):
    """纯文本文本入库 → pending → ready 且 chunk_count > 0。"""
    r = client.post(
        "/api/v1/documents",
        json={
            "source": "policy-tmp.txt",
            "text": (
                "公司差旅住宿标准。一线城市每晚不超过 500 元，"
                "二线城市 350 元，其他城市 250 元。\n\n"
                "餐饮补贴每天 100 元，不足一天按 50 元计。\n\n"
                "报销单需在出差结束后 15 个工作日内提交。"
            ),
        },
    )
    assert r.status_code == 202, r.text
    doc_id = r.json()["doc_id"]
    assert r.json()["status"] == "pending"

    final = _wait_ready(client, doc_id)
    assert final["status"] == "ready", f"入库失败: {final.get('error')}"
    assert final["chunk_count"] > 0
    assert final["error"] is None

    # 清理，避免影响其它用例的列表断言
    client.delete(f"/api/v1/documents/{doc_id}")


# ---------------------------------------------------------------------------
# 2. 列表 / 删除
# ---------------------------------------------------------------------------
def test_list_then_delete(client: TestClient):
    """上传后列表可见；删除后 200，再查 404。"""
    r = client.post(
        "/api/v1/documents",
        json={"source": "deleteme.txt", "text": "这是一份用来验证删除链路的临时文档。"},
    )
    assert r.status_code == 202
    doc_id = r.json()["doc_id"]
    _wait_ready(client, doc_id)

    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(item["doc_id"] == doc_id for item in body["items"])

    r = client.delete(f"/api/v1/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    r = client.get(f"/api/v1/documents/{doc_id}")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# 3. PDF 上传 + 问答引用富化
# ---------------------------------------------------------------------------
def test_pdf_upload_and_retrieved_enrichment(client: TestClient):
    """上传 PDF → ready；问 PDF 里有的事实 → retrieved 非空且字段齐。"""
    pdf_path = FIXTURES / "sample.pdf"
    assert pdf_path.is_file(), f"缺少夹具: {pdf_path}"

    with pdf_path.open("rb") as f:
        r = client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample.pdf", f, "application/pdf")},
        )
    assert r.status_code == 202, r.text
    doc_id = r.json()["doc_id"]

    final = _wait_ready(client, doc_id)
    assert final["status"] == "ready", f"PDF 入库失败: {final.get('error')}"
    assert final["chunk_count"] > 0

    r = client.post(
        "/api/v1/ask",
        json={"question": "What is the nightly hotel cap in Tier-1 cities?", "top_k": 5},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    # retrieved 必须带全字段，前端出处列表依赖它
    assert body["n_retrieved"] >= 1, f"检索为空: {body['answer'][:200]}"
    for it in body["retrieved"]:
        assert set(it) >= {"citation", "source", "chunk_index", "snippet", "score"}
        assert it["source"] == "sample.pdf"
        assert it["snippet"], "snippet 不能为空"
        assert isinstance(it["score"], (int, float))

    client.delete(f"/api/v1/documents/{doc_id}")


def test_unsupported_type_rejected_415(client: TestClient):
    """不支持的扩展名 → 415，且不落库。"""
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("evil.docx", b"not really a docx", "application/octet-stream")},
    )
    assert r.status_code == 415, r.text
    assert "docx" in r.json()["detail"].lower() or "不支持" in r.json()["detail"]


def test_scanned_pdf_rejected_422(client: TestClient):
    """无文本层的 PDF → 422 且信息明确（不是静默 ready）。"""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    import io

    buf = io.BytesIO()
    writer.write(buf)

    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")},
    )
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert "文本层" in detail or "扫描" in detail or "OCR" in detail
