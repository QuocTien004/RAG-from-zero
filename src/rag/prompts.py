"""Nơi tập trung các "prompt" — lời chỉ dẫn gửi cho LLM.

Tách prompt ra file riêng là thói quen tốt: dễ chỉnh sửa, thử nghiệm, so sánh
mà không phải đụng vào logic. Đây cũng là kỹ năng "prompt engineering".
"""

from __future__ import annotations

from .vector_store import SearchResult

# Chỉ dẫn vai trò (system) — ép model chỉ trả lời dựa trên ngữ cảnh, tránh "bịa".
SYSTEM_PROMPT = (
    "Bạn là trợ lý AI trả lời câu hỏi CHỈ dựa trên phần NGỮ CẢNH được cung cấp. "
    "Nếu ngữ cảnh không chứa đủ thông tin, hãy nói rõ là bạn không tìm thấy thông tin "
    "trong tài liệu, tuyệt đối không bịa. Trả lời ngắn gọn, chính xác, bằng tiếng Việt. "
    "Khi có thể, hãy trích dẫn nguồn theo dạng [nguồn: tên_file]."
)

# Khung câu hỏi (user) — ghép ngữ cảnh và câu hỏi lại với nhau.
RAG_PROMPT_TEMPLATE = """NGỮ CẢNH:
{context}

CÂU HỎI: {question}

Hãy trả lời câu hỏi dựa trên NGỮ CẢNH ở trên."""


def build_context(results: list[SearchResult]) -> str:
    """Ghép các đoạn tìm được thành một khối ngữ cảnh, có đánh số và ghi nguồn."""
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] (nguồn: {r.source})\n{r.text}")
    return "\n\n".join(blocks)
