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
# Các định dạng hình ảnh được hỗ trợ trong Hybrid Multimodal RAG
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class Document:
    """Một tài liệu đã đọc vào bộ nhớ (có thể là văn bản hoặc mô tả hình ảnh)."""

    source: str                 # Tên file nguồn (tương đối) để trích dẫn
    text: str                   # Toàn bộ nội dung văn bản
    image_path: str | None = None # Đường dẫn hình ảnh gốc (nếu là tài liệu ảnh)


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
        if suffix in _TEXT_SUFFIXES or suffix == ".pdf" or suffix in _IMAGE_SUFFIXES:
            rel = str(path.relative_to(raw_dir)).replace("\\", "/")
            file_map[rel] = (path, compute_file_hash(path))
    return file_map


def _read_pdf_text_and_images(
    path: Path,
    source_rel: str,
    multimodal_processor=None,
    extracted_images_dir: Path | None = None,
) -> tuple[str, list[Document]]:
    """Đọc văn bản từ PDF và bóc tách các hình ảnh nhúng bên trong."""
    from pypdf import PdfReader  # import cục bộ

    reader = PdfReader(str(path))
    text_parts = []
    image_docs: list[Document] = []

    pdf_stem = path.stem
    img_counter = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            text_parts.append(page_text)

        # Bóc tách hình ảnh trong trang PDF (nếu có thư mục lưu và processor)
        if multimodal_processor and extracted_images_dir:
            try:
                for img_file in page.images:
                    img_counter += 1
                    save_dir = extracted_images_dir / pdf_stem
                    save_dir.mkdir(parents=True, exist_ok=True)
                    out_img_name = f"page_{page_idx}_img_{img_counter}_{img_file.name}"
                    out_img_path = save_dir / out_img_name

                    # Lưu file ảnh trích xuất
                    with open(out_img_path, "wb") as f:
                        f.write(img_file.data)

                    # Phân tích ảnh bằng Hybrid OCR + Vision
                    rich_text = multimodal_processor.process_image(
                        out_img_path, f"{source_rel} (trang {page_idx}, hình {img_counter})"
                    )
                    image_docs.append(
                        Document(
                            source=f"{source_rel}#page{page_idx}_img{img_counter}",
                            text=rich_text,
                            image_path=str(out_img_path),
                        )
                    )
            except Exception:
                pass  # Bỏ qua nếu có ảnh định dạng đặc biệt không đọc được

    return "\n".join(text_parts), image_docs


def load_file_documents(
    path: Path,
    source_rel: str,
    multimodal_processor=None,
    extracted_images_dir: Path | None = None,
) -> list[Document]:
    """Đọc một file và trả về danh sách Document (hỗ trợ Text, PDF và Hình ảnh)."""
    suffix = path.suffix.lower()
    docs: list[Document] = []

    if suffix in _TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(source=source_rel, text=text))

    elif suffix == ".pdf":
        text, img_docs = _read_pdf_text_and_images(
            path, source_rel, multimodal_processor, extracted_images_dir
        )
        if text.strip():
            docs.append(Document(source=source_rel, text=text.strip()))
        docs.extend(img_docs)

    elif suffix in _IMAGE_SUFFIXES:
        if multimodal_processor:
            rich_text = multimodal_processor.process_image(path, source_rel)
        else:
            rich_text = f"[HÌNH ẢNH: {source_rel}]"
        docs.append(Document(source=source_rel, text=rich_text, image_path=str(path)))

    return docs


def load_single_document(path: Path, source_rel: str) -> Document | None:
    """Tương thích ngược: Đọc một file đơn lẻ."""
    docs = load_file_documents(path, source_rel)
    return docs[0] if docs else None


def load_documents(
    raw_dir: Path,
    multimodal_processor=None,
    extracted_images_dir: Path | None = None,
) -> list[Document]:
    """Duyệt đệ quy thư mục raw_dir, đọc mọi file được hỗ trợ (cả text lẫn ảnh)."""
    documents: list[Document] = []
    file_map = scan_raw_files(raw_dir)
    for source_rel, (path, _) in file_map.items():
        docs = load_file_documents(
            path,
            source_rel,
            multimodal_processor=multimodal_processor,
            extracted_images_dir=extracted_images_dir,
        )
        documents.extend(docs)

    return documents
