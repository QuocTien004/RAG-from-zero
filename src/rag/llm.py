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


class OllamaLLM:
    """Bọc lời gọi Ollama API cục bộ (100% offline, miễn phí, không tốn API key)."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        images: list[Path] | None = None,
    ) -> str:
        """Gọi Ollama sinh văn bản từ prompt (kèm hình ảnh trực quan nếu có)."""
        import base64
        import json
        import urllib.error
        import urllib.request

        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        if images:
            encoded_images = []
            for img_p in images:
                try:
                    with open(img_p, "rb") as f:
                        encoded_images.append(base64.b64encode(f.read()).decode("utf-8"))
                except Exception:
                    pass
            if encoded_images:
                payload["images"] = encoded_images

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return (result.get("response") or "").strip()
        except urllib.error.URLError as e:
            if "Connection refused" in str(e) or "10061" in str(e):
                raise RuntimeError(
                    f"Không thể kết nối đến Ollama tại {self._base_url}.\n"
                    "-> Hãy mở ứng dụng Ollama hoặc kiểm tra xem tiến trình Ollama có đang chạy không."
                ) from e
            raise RuntimeError(f"Lỗi khi gọi Ollama API: {e}") from e
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code == 404 or "not found" in err_body.lower():
                raise RuntimeError(
                    f"Mô hình '{self._model}' chưa được tải về Ollama.\n"
                    f"-> Hãy mở PowerShell và chạy lệnh: ollama pull {self._model}"
                ) from e
            raise RuntimeError(f"Lỗi Ollama HTTP {e.code}: {err_body}") from e
