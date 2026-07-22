"""BƯỚC 6 — GENERATE: gọi Gemini để sinh câu trả lời dựa trên ngữ cảnh đã tìm được.

Tách riêng phần gọi LLM ra một lớp giúp:
  - Dễ đổi model (chỉ sửa một chỗ).
  - Dễ thêm tham số (nhiệt độ, giới hạn token...) sau này.
  - Ở bài về Agent, chính lớp này có thể được tái sử dụng để "suy nghĩ" và gọi tool.
"""

from __future__ import annotations

from google import genai
from google.genai import types


class GeminiLLM:
    """Bọc lời gọi Gemini generate_content."""

    def __init__(self, client: genai.Client, model: str):
        self._client = client
        self._model = model

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Sinh văn bản từ prompt. `system` là chỉ dẫn vai trò cho model."""
        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()
