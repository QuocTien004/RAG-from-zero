"""BƯỚC 2 — CHUNK: chia tài liệu dài thành các đoạn nhỏ (chunk) có chồng lấn.

Vì sao phải chia nhỏ?
  1. Model embedding và LLM đều có giới hạn độ dài đầu vào.
  2. Đoạn càng nhỏ, tìm kiếm càng "trúng đích" (retrieval chính xác hơn).
  3. Phần chồng lấn (overlap) giữ cho ngữ cảnh không bị cắt cụt giữa 2 đoạn.

Ở phiên bản cơ bản này ta chia theo số ký tự (đơn giản, dễ hình dung).
Nâng cao hơn có thể chia theo câu/đoạn văn hoặc theo số token.
"""

from __future__ import annotations

from dataclasses import dataclass

from .loader import Document


@dataclass
class Chunk:
    """Một đoạn văn bản nhỏ, kèm nguồn gốc để truy vết."""

    source: str  # File gốc
    index: int   # Thứ tự đoạn trong file (0, 1, 2, ...)
    text: str    # Nội dung đoạn


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Chia một chuỗi thành các đoạn dài `chunk_size`, mỗi đoạn lùi lại `overlap` ký tự.

    Ví dụ chunk_size=100, overlap=20: đoạn 1 = [0:100], đoạn 2 = [80:180], ...
    """
    text = text.strip()
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap phải nhỏ hơn chunk_size")

    chunks: list[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:  # Đã lấy tới cuối văn bản
            break
        start = end - overlap  # Lùi lại để tạo phần chồng lấn

    return chunks


def chunk_documents(
    documents: list[Document], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Áp dụng chunk_text cho danh sách tài liệu, trả về danh sách Chunk phẳng."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        for i, piece in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            all_chunks.append(Chunk(source=doc.source, index=i, text=piece))
    return all_chunks
