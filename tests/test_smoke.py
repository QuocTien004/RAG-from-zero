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


def test_vector_store_remove_and_add_sources():
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    metas = [
        {"text": "file1_chunk0", "source": "file1.md", "index": 0},
        {"text": "file2_chunk0", "source": "file2.md", "index": 0},
    ]
    store = VectorStore.build(vectors, metas)
    assert len(store) == 2
    assert store.sources == {"file1.md", "file2.md"}

    # Xóa file1
    store = store.remove_sources({"file1.md"})
    assert len(store) == 1
    assert store.sources == {"file2.md"}

    # Thêm file3
    new_vec = np.array([[0, 0, 1]], dtype=np.float32)
    new_meta = [{"text": "file3_chunk0", "source": "file3.md", "index": 0}]
    store = store.add_chunks(new_vec, new_meta)
    assert len(store) == 2
    assert store.sources == {"file2.md", "file3.md"}


def test_chunk_documents_preserves_image_path():
    from rag.chunker import chunk_documents
    from rag.loader import Document

    docs = [
        Document(source="test.txt", text="Hello world!"),
        Document(source="chart.png", text="[HÌNH ẢNH: chart.png] Biểu đồ", image_path="data/raw/chart.png"),
    ]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(chunks) == 2
    assert chunks[0].image_path is None
    assert chunks[1].image_path == "data/raw/chart.png"
    assert chunks[1].source == "chart.png"


def test_vector_store_retrieves_image_path():
    from rag.prompts import build_context

    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    metas = [
        {"text": "Text only chunk", "source": "doc.txt", "index": 0, "image_path": None},
        {"text": "Image chunk description", "source": "diagram.png", "index": 0, "image_path": "path/to/diagram.png"},
    ]
    store = VectorStore.build(vectors, metas)
    results = store.search(np.array([0, 1, 0], dtype=np.float32), top_k=1)
    assert len(results) == 1
    assert results[0].image_path == "path/to/diagram.png"

    context = build_context(results)
    assert "hình ảnh: path/to/diagram.png" in context


def test_settings_detects_llm_provider():
    from rag.config import load_settings

    settings = load_settings()
    assert settings.llm_provider in {"gemini", "ollama"}
    assert settings.ollama_model != ""


def test_ollama_llm_initialization():
    from rag.llm import OllamaLLM

    llm = OllamaLLM(model="qwen2.5:7b")
    assert llm._model == "qwen2.5:7b"
    assert "11434" in llm._base_url
