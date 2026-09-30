"""Cấu hình tập trung cho toàn bộ pipeline.

Mọi tham số (API key, tên model, kích thước chunk...) đều được đọc từ file .env
qua thư viện python-dotenv và gói gọn trong một đối tượng `Settings` duy nhất.
Nhờ vậy các module khác không cần biết biến môi trường ở đâu — chỉ nhận Settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Thư mục gốc của project = lùi 2 cấp từ file này: src/rag/config.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Nạp biến môi trường từ file .env ở thư mục gốc (nếu có)
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Toàn bộ cấu hình cần thiết, gom về một chỗ (immutable cho an toàn)."""

    # --- LLM Provider ('gemini' hoặc 'ollama') ---
    llm_provider: str
    ollama_model: str
    ollama_base_url: str

    # --- Gemini (dùng khi llm_provider='gemini') ---
    api_key: str
    chat_model: str

    # --- Embedding cục bộ, miễn phí (sentence-transformers) ---
    embed_model: str

    # --- Đường dẫn dữ liệu ---
    raw_dir: Path       # Nơi chứa tài liệu gốc (.txt, .md, .pdf, .docx, ảnh)
    store_path: Path    # Nơi lưu vector store đã build
    manifest_path: Path # Nơi lưu hash và thông tin tài liệu đã nạp (cho incremental ingestion)
    extracted_images_dir: Path # Nơi lưu hình ảnh trích xuất từ tài liệu

    # --- Tham số RAG ---
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_settings() -> Settings:
    """Đọc cấu hình từ môi trường và trả về đối tượng Settings."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Tự động nhận diện provider: nếu đặt rõ LLM_PROVIDER thì dùng nó,
    # ngược lại: nếu có API key thì dùng Gemini, không có thì mặc định dùng Ollama cục bộ.
    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not llm_provider:
        llm_provider = "gemini" if api_key else "ollama"

    data_dir = PROJECT_ROOT / "data"
    store_dir = data_dir / "processed"
    return Settings(
        llm_provider=llm_provider,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        api_key=api_key,
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest"),
        # Model embedding cục bộ (sentence-transformers) — miễn phí, không cần API
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
        raw_dir=data_dir / "raw",
        store_path=store_dir / "vector_store.npz",
        manifest_path=store_dir / "manifest.json",
        extracted_images_dir=store_dir / "extracted_images",
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        top_k=int(os.getenv("RAG_TOP_K", "4")),
    )
