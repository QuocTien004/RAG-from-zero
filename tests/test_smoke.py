"""Smoke test — kiểm tra nhanh các thành phần KHÔNG cần gọi API (chạy offline).

Chạy:  pytest    (hoặc:  python -m pytest)

Mục đích: đảm bảo logic chia chunk và tìm kiếm vector hoạt động đúng,
mà không tốn quota API và không cần mạng.
"""

import sys
from pathlib import Path

import numpy as np

# Cho phép import package 'rag' khi chạy pytest từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunker import chunk_text  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def test_chunk_text_respects_size_and_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 1                         # phải chia thành nhiều đoạn
    assert all(len(c) <= 300 for c in chunks)      # không đoạn nào vượt chunk_size
    # Đoạn 2 phải bắt đầu bằng phần đuôi chồng lấn của đoạn 1
    assert chunks[1][:50] == chunks[0][-50:]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("   ", chunk_size=100, overlap=10) == []


def test_vector_store_search_ranks_by_similarity():
    # 3 vector: doc0 trùng hướng câu hỏi, doc2 gần, doc1 vuông góc
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype=np.float32)
    metas = [{"text": f"doc{i}", "source": "t", "index": i} for i in range(3)]
    store = VectorStore.build(vectors, metas)

    results = store.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)

    assert len(results) == 2
    assert results[0].text == "doc0"               # giống nhất đứng đầu
    assert results[0].score >= results[1].score    # điểm giảm dần
