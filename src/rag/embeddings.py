"""BƯỚC 3 — EMBED: biến văn bản thành vector số bằng MÔ HÌNH MIỄN PHÍ CHẠY TRÊN MÁY.

Khác với việc gọi API embedding của Gemini, ở đây ta dùng thư viện
`sentence-transformers` để tạo embedding NGAY TRÊN MÁY của bạn:
  - Miễn phí, KHÔNG tốn quota API, chạy được cả khi offline.
  - Model mặc định: intfloat/multilingual-e5-small — đa ngôn ngữ, hỗ trợ tiếng Việt.
  - Lần chạy ĐẦU TIÊN sẽ tự tải model về (~470MB) rồi cache lại cho các lần sau.

Gemini giờ CHỈ được dùng để chat (sinh câu trả lời), không dùng cho embedding nữa.

Embedding (vector nhúng) là một dãy số biểu diễn "ý nghĩa" của đoạn văn: hai đoạn gần
nghĩa nhau -> hai vector nằm gần nhau trong không gian.

Mẹo: dòng model họ E5 cần thêm tiền tố "query: " cho câu hỏi và "passage: " cho tài
liệu để đạt chất lượng tốt nhất. Ta xử lý sẵn bên trong, người dùng không phải bận tâm.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Tạo embedding cục bộ bằng sentence-transformers (miễn phí, không cần API)."""

    def __init__(self, model_name: str):
        # Tải model (hoặc lấy từ cache). Chạy trên CPU là đủ cho demo.
        self._model = SentenceTransformer(model_name)
        # Model họ E5 cần tiền tố query/passage; các model khác thì bỏ qua.
        self._is_e5 = "e5" in model_name.lower()

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # chuẩn hoá sẵn để cosine = tích vô hướng
        )
        return vectors.astype(np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Nhúng danh sách đoạn tài liệu. Trả về ma trận (N, D)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        prepared = [f"passage: {t}" for t in texts] if self._is_e5 else texts
        return self._encode(prepared)

    def embed_query(self, text: str) -> np.ndarray:
        """Nhúng một câu hỏi. Trả về vector 1 chiều (D,)."""
        prepared = f"query: {text}" if self._is_e5 else text
        return self._encode([prepared])[0]
