"""BƯỚC 6 — GENERATE: gọi Gemini để sinh câu trả lời dựa trên ngữ cảnh đã tìm được.

Tách riêng phần gọi LLM ra một lớp giúp:
  - Dễ đổi model (chỉ sửa một chỗ).
  - Dễ thêm tham số (nhiệt độ, giới hạn token...) sau này.
  - Có thể tái sử dụng để làm module reasoning cho AI Agent hoặc mở rộng tool calling.
"""

from __future__ import annotations

from pathlib import Path

from google import genai
from google.genai import types


class GeminiLLM:
    """Bọc lời gọi Gemini generate_content."""

    def __init__(self, client: genai.Client, model: str):
        self._client = client
        self._model = model

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[Path] | None = None,
    ) -> str:
        """Sinh văn bản từ prompt (kèm hình ảnh trực quan nếu có). `system` là chỉ dẫn vai trò cho model."""
        config = None
        if system:
            config = types.GenerateContentConfig(system_instruction=system)

        contents = []
        if images:
            from PIL import Image

            for img_p in images:
                try:
                    contents.append(Image.open(img_p))
                except Exception:
                    pass

        contents.append(prompt)

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents if len(contents) > 1 else prompt,
            config=config,
        )
        return (response.text or "").strip()
