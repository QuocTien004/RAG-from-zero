"""CLI — Nạp tài liệu vào vector store (chạy giai đoạn OFFLINE của RAG).

Cách dùng:
    python scripts/ingest.py

Sau khi chạy xong, một vector store sẽ được lưu tại data/processed/.
Chạy lại lệnh này mỗi khi bạn thêm/sửa tài liệu trong data/raw/.
"""

import sys
from pathlib import Path

# Đảm bảo in được tiếng Việt + emoji trên terminal Windows (mặc định dùng cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Cho phép import package 'rag' ngay cả khi chưa 'pip install -e .'
# (thêm thư mục src/ vào đường dẫn tìm module của Python)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    pipeline = RAGPipeline()
    print(f"📥 Đang đọc tài liệu từ: {pipeline.settings.raw_dir}")

    num_chunks = pipeline.ingest()

    print(f"✅ Đã tạo {num_chunks} chunk.")
    print(f"💾 Vector store đã lưu tại: {pipeline.settings.store_path}")
    print('👉 Bước tiếp theo: python scripts/ask.py "câu hỏi của bạn"')


if __name__ == "__main__":
    main()
