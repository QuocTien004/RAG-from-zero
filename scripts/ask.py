"""CLI — Đặt câu hỏi cho hệ thống RAG (chạy giai đoạn ONLINE của RAG).

Cách dùng:
    python scripts/ask.py "RAG là gì?"        # hỏi một câu rồi thoát
    python scripts/ask.py                       # chế độ hỏi–đáp liên tục

Yêu cầu: đã chạy `python scripts/ingest.py` trước đó.
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


def ask_once(pipeline: RAGPipeline, question: str) -> None:
    """Hỏi một câu và in ra câu trả lời kèm nguồn tham khảo."""
    result = pipeline.answer(question)

    print("\n🤖 Trả lời:")
    print(result.answer)

    print("\n📚 Nguồn tham khảo (đoạn liên quan nhất):")
    for i, s in enumerate(result.sources, start=1):
        img_info = f"\n     🖼️ Hình ảnh đính kèm: {s.image_path}" if s.image_path else ""
        print(f"  [{i}] {s.source}  (điểm tương đồng: {s.score:.3f}){img_info}")


def main() -> None:
    pipeline = RAGPipeline()

    # Nếu có câu hỏi truyền vào từ dòng lệnh -> trả lời một lần rồi thoát
    if len(sys.argv) > 1:
        ask_once(pipeline, " ".join(sys.argv[1:]))
        return

    # Ngược lại -> vào chế độ hỏi–đáp liên tục
    print("💬 Chế độ hỏi–đáp RAG (gõ 'exit' để thoát)")
    while True:
        try:
            question = input("\n❓ Câu hỏi: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit", "thoat", "q"}:
            break
        if question:
            ask_once(pipeline, question)


if __name__ == "__main__":
    main()
