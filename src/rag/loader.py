"""BƯỚC 1 — LOAD: đọc tài liệu thô từ thư mục data/raw.

Hỗ trợ .txt, .md và .pdf. Mỗi file trở thành một `Document`.
Đây là "nguyên liệu đầu vào" cho toàn bộ pipeline RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Các định dạng văn bản thuần đọc trực tiếp được
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    """Một tài liệu đã đọc vào bộ nhớ."""

    source: str  # Tên file (tương đối) để sau này trích dẫn nguồn
    text: str    # Toàn bộ nội dung văn bản


def _read_pdf(path: Path) -> str:
    """Đọc nội dung text từ file PDF (cần thư viện pypdf)."""
    from pypdf import PdfReader  # import cục bộ: chỉ cần khi thực sự gặp PDF

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(raw_dir: Path) -> list[Document]:
    """Duyệt đệ quy thư mục raw_dir, đọc mọi file được hỗ trợ."""
    documents: list[Document] = []
    if not raw_dir.exists():
        return documents

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue  # Bỏ qua định dạng không hỗ trợ

        text = text.strip()
        if text:  # Bỏ qua file rỗng
            documents.append(
                Document(source=str(path.relative_to(raw_dir)), text=text)
            )

    return documents
