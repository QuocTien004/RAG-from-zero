"""BƯỚC 1 — LOAD: đọc tài liệu thô từ thư mục data/raw.

Hỗ trợ .txt, .md và .pdf. Mỗi file trở thành một `Document`.
Đây là "nguyên liệu đầu vào" cho toàn bộ pipeline RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hashlib

# Các định dạng văn bản thuần đọc trực tiếp được
_TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    """Một tài liệu đã đọc vào bộ nhớ."""

    source: str  # Tên file (tương đối) để sau này trích dẫn nguồn
    text: str    # Toàn bộ nội dung văn bản


def compute_file_hash(path: Path) -> str:
    """Tính mã băm SHA-256 của file để phát hiện thay đổi nội dung."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_raw_files(raw_dir: Path) -> dict[str, tuple[Path, str]]:
    """Quét thư mục raw_dir và trả về map: {source_rel: (abs_path, sha256_hash)}."""
    file_map: dict[str, tuple[Path, str]] = {}
    if not raw_dir.exists():
        return file_map

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _TEXT_SUFFIXES or suffix == ".pdf":
            rel = str(path.relative_to(raw_dir)).replace("\\", "/")
            file_map[rel] = (path, compute_file_hash(path))
    return file_map


def _read_pdf(path: Path) -> str:
    """Đọc nội dung text từ file PDF (cần thư viện pypdf)."""
    from pypdf import PdfReader  # import cục bộ: chỉ cần khi thực sự gặp PDF

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def load_single_document(path: Path, source_rel: str) -> Document | None:
    """Đọc một file cụ thể và trả về Document (hoặc None nếu rỗng/lỗi)."""
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        return None

    text = text.strip()
    if not text:
        return None
    return Document(source=source_rel, text=text)


def load_documents(raw_dir: Path) -> list[Document]:
    """Duyệt đệ quy thư mục raw_dir, đọc mọi file được hỗ trợ."""
    documents: list[Document] = []
    file_map = scan_raw_files(raw_dir)
    for source_rel, (path, _) in file_map.items():
        doc = load_single_document(path, source_rel)
        if doc is not None:
            documents.append(doc)

    return documents
