"""BƯỚC ĐA PHƯƠNG THỨC — MULTIMODAL PROCESSOR:
Kết hợp 3 kỹ thuật:
  1. OCR cục bộ (EasyOCR): Trích xuất chữ, số, bảng biểu chính xác 100%.
  2. Vision AI (Google Gemini): Đọc hiểu ý nghĩa biểu đồ, sơ đồ, infographics.
  3. Rich Image Chunk: Tạo đoạn ngữ cảnh kết hợp và lưu vết đường dẫn ảnh gốc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class MultimodalProcessor:
    """Bộ xử lý hình ảnh đa phương thức kết hợp OCR và Vision AI."""

    def __init__(self, gemini_client=None, chat_model: str = "gemini-flash-latest"):
        self._client = gemini_client
        self._model = chat_model
        self._ocr_reader = None

    def _get_ocr_reader(self):
        """Khởi tạo EasyOCR lười (lazy-load) khi thực sự cần xử lý ảnh."""
        if self._ocr_reader is None:
            try:
                import easyocr

                # Khởi tạo mô hình nhận diện Tiếng Việt và Tiếng Anh
                self._ocr_reader = easyocr.Reader(["vi", "en"], gpu=False, verbose=False)
            except Exception as e:
                logger.warning(f"Không thể khởi tạo EasyOCR: {e}")
                self._ocr_reader = False
        return self._ocr_reader if self._ocr_reader is not False else None

    def extract_ocr_text(self, image_path: Path) -> str:
        """Trích xuất toàn bộ văn bản và số liệu trong ảnh bằng OCR cục bộ."""
        reader = self._get_ocr_reader()
        if not reader:
            return ""
        try:
            results = reader.readtext(str(image_path), detail=0)
            return " ".join(results).strip()
        except Exception as e:
            logger.warning(f"Lỗi khi OCR file {image_path}: {e}")
            return ""

    def describe_with_vision(self, image_path: Path) -> str:
        """Dùng Gemini Vision để hiểu ngữ nghĩa biểu đồ, hình vẽ và sơ đồ."""
        if not self._client:
            return ""
        try:
            pil_image = Image.open(image_path)
            prompt = (
                "Hãy phân tích chi tiết hình ảnh này:\n"
                "1. Xác định loại hình ảnh (biểu đồ cột/tròn, sơ đồ quy trình, kiến trúc, bảng biểu, ảnh chụp...).\n"
                "2. Tóm tắt nội dung chính và các số liệu hoặc kết luận quan trọng được thể hiện.\n"
                "Trả lời súc tích, ngắn gọn bằng tiếng Việt."
            )
            response = self._client.models.generate_content(
                model=self._model,
                contents=[pil_image, prompt],
            )
            return (response.text or "").strip()
        except Exception as e:
            logger.warning(f"Không thể gọi Gemini Vision cho {image_path}: {e}")
            return ""

    def process_image(self, image_path: Path, source_rel: str) -> str:
        """Tạo đoạn văn bản Rich Chunk kết hợp cả OCR và mô tả Vision AI."""
        # 1. Trích xuất text/số bằng OCR cục bộ
        ocr_text = self.extract_ocr_text(image_path)

        # 2. Phân tích ngữ nghĩa bằng Vision AI (nếu có client)
        vision_desc = self.describe_with_vision(image_path)

        # 3. Ghép thành Rich Image Chunk
        sections = [f"[HÌNH ẢNH: {source_rel}]"]
        if vision_desc:
            sections.append(f"• Mô tả trực quan (Vision AI): {vision_desc}")
        if ocr_text:
            sections.append(f"• Dữ liệu văn bản & số liệu (OCR): {ocr_text}")

        if not vision_desc and not ocr_text:
            sections.append("• Tệp hình ảnh đính kèm trong tài liệu.")

        return "\n".join(sections)
