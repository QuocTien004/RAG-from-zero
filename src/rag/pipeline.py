"""RAGPipeline — lớp ĐIỀU PHỐI ghép toàn bộ 6 bước lại thành một quy trình hoàn chỉnh.

Toàn bộ RAG được rút gọn về HAI hành động:

  1. ingest()  — GIAI ĐOẠN OFFLINE (chạy 1 lần mỗi khi có tài liệu mới):
       Load -> Chunk -> Embed -> Store

  2. answer()  — GIAI ĐOẠN ONLINE (chạy mỗi khi có câu hỏi):
       Embed câu hỏi -> Retrieve (tìm đoạn liên quan) -> Generate (LLM trả lời)

Interface gọn gàng này chính là "Tool/Skill" có thể tích hợp trực tiếp vào AI Agent:
Agent chỉ cần gọi `pipeline.answer("câu hỏi")` mà không quan tâm chi tiết bên trong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from pathlib import Path

from google import genai
from google.genai import types

from . import prompts
from .chunker import chunk_documents
from .config import Settings, load_settings
from .embeddings import LocalEmbedder
from .llm import GeminiLLM
from .loader import load_documents, load_file_documents, load_single_document, scan_raw_files
from .multimodal import MultimodalProcessor
from .vector_store import SearchResult, VectorStore


@dataclass
class RAGAnswer:
    """Kết quả trả về cho người dùng: câu trả lời + các nguồn đã dùng."""

    answer: str
    sources: list[SearchResult]


class RAGPipeline:
    def __init__(self, settings: Settings | None = None):
        # Nếu không truyền sẵn cấu hình thì tự đọc từ .env
        self.settings = settings or load_settings()

        # Cấu hình retry NGẮN & CÓ GIỚI HẠN cho các lỗi tạm thời của server
        # (429 hết quota tạm thời, 5xx quá tải). Mặc định SDK retry rất lâu (~2 phút)
        # gây cảm giác "treo"; ở đây ta giới hạn ~4 lần, tối đa ~10s mỗi lần.
        self._http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=4,
                initial_delay=1.0,
                max_delay=10.0,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        )
        self._llm: GeminiLLM | None = None
        self._multimodal_processor: MultimodalProcessor | None = None

        # Embedding chạy CỤC BỘ, miễn phí — không cần API, không tốn quota
        self.embedder = LocalEmbedder(self.settings.embed_model)

        self._store: VectorStore | None = None  # nạp lười (lazy) khi cần

    @property
    def multimodal_processor(self) -> MultimodalProcessor:
        """Bộ xử lý đa phương thức OCR + Vision AI (tự động fallback nếu không có API key)."""
        if self._multimodal_processor is None:
            client = None
            if self.settings.api_key:
                client = genai.Client(
                    api_key=self.settings.api_key, http_options=self._http_options
                )
            self._multimodal_processor = MultimodalProcessor(
                gemini_client=client,
                chat_model=self.settings.chat_model,
            )
        return self._multimodal_processor

    @property
    def llm(self) -> GeminiLLM:
        """Khởi tạo GeminiLLM khi thực sự cần gọi sinh câu trả lời."""
        if self._llm is None:
            if not self.settings.api_key:
                raise RuntimeError(
                    "Thiếu GEMINI_API_KEY.\n"
                    "-> Hãy mở file .env và điền API key lấy từ https://aistudio.google.com/apikey"
                )
            client = genai.Client(
                api_key=self.settings.api_key, http_options=self._http_options
            )
            self._llm = GeminiLLM(client, self.settings.chat_model)
        return self._llm

    # ---------- GIAI ĐOẠN 1: INGEST (offline) ----------
    def ingest(self, force: bool = False) -> int:
        """Nạp tài liệu vào vector store với cơ chế Nạp bù thông minh (Incremental Ingestion).

        - Nếu file chưa đổi hash: bỏ qua, không tốn tài nguyên nhúng lại.
        - Nếu có file mới / file sửa: chỉ nhúng các chunk của file đó và ghép vào store.
        - Nếu có file bị xóa: tự động gỡ các chunk liên quan khỏi store.
        - force=True: dựng lại toàn bộ vector store từ con số 0.
        """
        current_files = scan_raw_files(self.settings.raw_dir)
        if not current_files:
            raise RuntimeError(
                f"Không tìm thấy tài liệu nào trong {self.settings.raw_dir}.\n"
                "-> Hãy bỏ vài file .txt/.md/.pdf vào thư mục data/raw/ rồi thử lại."
            )

        manifest_path = self.settings.manifest_path
        store_exists = self.settings.store_path.exists()

        prev_hashes: dict[str, str] = {}
        if manifest_path.exists() and not force and store_exists:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                prev_hashes = manifest.get("file_hashes", {})
            except Exception:
                prev_hashes = {}

        # Nếu chưa có store hoặc yêu cầu force rebuild -> Xây mới toàn bộ
        if force or not store_exists or not prev_hashes:
            print(f"📦 Đang xây dựng mới vector store cho {len(current_files)} file...")
            documents = []
            for src, (path, _) in current_files.items():
                docs = load_file_documents(
                    path,
                    src,
                    multimodal_processor=self.multimodal_processor,
                    extracted_images_dir=self.settings.extracted_images_dir,
                )
                documents.extend(docs)

            chunks = chunk_documents(
                documents, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if not chunks:
                raise RuntimeError("Tài liệu không có nội dung văn bản để chia chunk.")

            vectors = self.embedder.embed_documents([c.text for c in chunks])
            metadatas = [
                {
                    "text": c.text,
                    "source": c.source,
                    "index": c.index,
                    "image_path": c.image_path,
                }
                for c in chunks
            ]
            store = VectorStore.build(vectors, metadatas)
            store.save(self.settings.store_path)

            manifest_data = {
                "updated_at": datetime.now().isoformat(),
                "total_chunks": len(store),
                "file_hashes": {src: h for src, (_, h) in current_files.items()},
            }
            manifest_path.write_text(
                json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._store = store
            return len(chunks)

        # Ngược lại: Nạp bù thông minh (Incremental Update)
        new_or_modified = {
            src: (path, h)
            for src, (path, h) in current_files.items()
            if prev_hashes.get(src) != h
        }
        deleted = set(prev_hashes.keys()) - set(current_files.keys())
        unchanged = {
            src
            for src, (_, h) in current_files.items()
            if prev_hashes.get(src) == h
        }

        # Nếu không có file nào thay đổi
        if not new_or_modified and not deleted:
            print(f"✨ Không có file nào thay đổi ({len(unchanged)} file đã nạp). Vector store đã ở trạng thái mới nhất.")
            store = self._get_store()
            return len(store)

        print("🔄 Phát hiện thay đổi trong thư mục data/raw/:")
        for src in new_or_modified:
            label = "cập nhật" if src in prev_hashes else "file mới"
            print(f"   • [{label}] {src}")
        for src in deleted:
            print(f"   • [xóa] {src}")
        if unchanged:
            print(f"   • [giữ nguyên] {len(unchanged)} file")

        # Nạp store hiện tại
        store = VectorStore.load(self.settings.store_path)

        # 1. Loại bỏ các chunk của file bị sửa đổi hoặc bị xóa
        to_remove = set(new_or_modified.keys()) | deleted
        store = store.remove_sources(to_remove)

        # 2. Đọc và chia chunk cho các file mới/sửa đổi
        new_docs = []
        for src, (path, _) in new_or_modified.items():
            docs = load_file_documents(
                path,
                src,
                multimodal_processor=self.multimodal_processor,
                extracted_images_dir=self.settings.extracted_images_dir,
            )
            new_docs.extend(docs)

        if new_docs:
            new_chunks = chunk_documents(
                new_docs, self.settings.chunk_size, self.settings.chunk_overlap
            )
            if new_chunks:
                print(f"⚡ Đang nhúng {len(new_chunks)} chunk mới từ {len(new_docs)} tài liệu...")
                new_vectors = self.embedder.embed_documents([c.text for c in new_chunks])
                new_metadatas = [
                    {
                        "text": c.text,
                        "source": c.source,
                        "index": c.index,
                        "image_path": c.image_path,
                    }
                    for c in new_chunks
                ]
                store = store.add_chunks(new_vectors, new_metadatas)

        # Lưu store và cập nhật manifest
        store.save(self.settings.store_path)
        manifest_data = {
            "updated_at": datetime.now().isoformat(),
            "total_chunks": len(store),
            "file_hashes": {src: h for src, (_, h) in current_files.items()},
        }
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._store = store
        return len(store)

    # ---------- GIAI ĐOẠN 2: QUERY (online) ----------
    def answer(self, question: str) -> RAGAnswer:
        """Trả lời một câu hỏi bằng quy trình Retrieve -> Generate (kèm hình ảnh đa phương thức)."""
        store = self._get_store()

        # (a) Nhúng câu hỏi thành vector rồi tìm các đoạn liên quan nhất
        query_vector = self.embedder.embed_query(question)
        results = store.search(query_vector, self.settings.top_k)

        # (b) Thu thập danh sách ảnh từ các chunk tìm kiếm được (nếu có)
        image_paths: list[Path] = []
        for r in results:
            if r.image_path:
                img_p = Path(r.image_path)
                if img_p.exists() and img_p not in image_paths:
                    image_paths.append(img_p)

        # (c) Ghép ngữ cảnh + câu hỏi thành prompt và nhờ LLM trả lời (kèm ảnh trực quan)
        context = prompts.build_context(results)
        prompt = prompts.RAG_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        answer_text = self.llm.generate(
            prompt,
            system=prompts.SYSTEM_PROMPT,
            images=image_paths if image_paths else None,
        )

        return RAGAnswer(answer=answer_text, sources=results)

    # ---------- Tiện ích nội bộ ----------
    def _get_store(self) -> VectorStore:
        """Nạp vector store từ đĩa một lần rồi giữ trong bộ nhớ (cache)."""
        if self._store is None:
            self._store = VectorStore.load(self.settings.store_path)
        return self._store
