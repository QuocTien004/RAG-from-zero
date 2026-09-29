"""CLI — Nạp tài liệu vào vector store (chạy giai đoạn OFFLINE của RAG).

Cách dùng:
    python scripts/ingest.py           # Nạp bù thông minh (chỉ nhúng file mới/sửa đổi)
    python scripts/ingest.py --force   # Ép buộc xây dựng lại toàn bộ từ đầu

Sau khi chạy xong, vector store và manifest sẽ được lưu tại data/processed/.
"""

import sys
from pathlib import Path

# Đảm bảo in được tiếng Việt + emoji trên terminal Windows (mặc định dùng cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Cho phép import package 'rag' ngay cả khi chưa 'pip install -e .'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.pipeline import RAGPipeline  # noqa: E402


def main() -> None:
    force = "--force" in sys.argv or "-f" in sys.argv
    pipeline = RAGPipeline()

    print(f"📥 Đang quét tài liệu từ: {pipeline.settings.raw_dir}")
    if force:
        print("⚠️  Chế độ --force: Xây dựng lại toàn bộ vector store...")

    num_chunks = pipeline.ingest(force=force)

    print(f"✅ Hoàn tất! Tổng số chunk trong kho: {num_chunks}")
    print(f"💾 Vector store: {pipeline.settings.store_path}")
    print('👉 Bước tiếp theo: python scripts/ask.py "câu hỏi của bạn"')


if __name__ == "__main__":
    main()
