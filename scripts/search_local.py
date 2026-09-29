"""CLI — Tìm kiếm ngữ nghĩa offline bằng vector store (Retrieval Local).

Cách dùng:
    python scripts/search_local.py "Muốn model biết dữ liệu mới mà không fine-tune thì làm gì?"

Chạy hoàn toàn cục bộ trên máy (100% offline, không tốn quota hay cần API key).
"""

import sys
from pathlib import Path

# Đảm bảo in được tiếng Việt + emoji trên terminal Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import load_settings  # noqa: E402
from rag.embeddings import LocalEmbedder  # noqa: E402
from rag.vector_store import VectorStore  # noqa: E402


def search_local(query: str, top_k: int = 3) -> None:
    settings = load_settings()
    store = VectorStore.load(settings.store_path)
    embedder = LocalEmbedder(settings.embed_model)

    print(f"\n🔍 Câu hỏi: \"{query}\"")
    print(f"⚙️  top_k = {top_k}")

    query_vector = embedder.embed_query(query)
    results = store.search(query_vector, top_k=top_k)

    print("\n📚 Kết quả truy xuất (Retrieval Local):")
    for i, s in enumerate(results, start=1):
        img_info = f" | ảnh={s.image_path}" if s.image_path else ""
        print(f"\n[{i}] score={s.score:.4f} | nguồn={s.source}{img_info}")
        print(f"    {s.text.strip()[:200]}...")


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Muốn model biết dữ liệu mới mà không fine-tune thì làm gì?"

    search_local(query)


if __name__ == "__main__":
    main()
