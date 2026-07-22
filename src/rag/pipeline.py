"""RAGPipeline — lớp ĐIỀU PHỐI ghép toàn bộ 6 bước lại thành một quy trình hoàn chỉnh.

Toàn bộ RAG được rút gọn về HAI hành động:

  1. ingest()  — GIAI ĐOẠN OFFLINE (chạy 1 lần mỗi khi có tài liệu mới):
       Load -> Chunk -> Embed -> Store

  2. answer()  — GIAI ĐOẠN ONLINE (chạy mỗi khi có câu hỏi):
       Embed câu hỏi -> Retrieve (tìm đoạn liên quan) -> Generate (LLM trả lời)

Interface gọn gàng này chính là "SKILL" mà AI Agent sẽ gọi tới ở các bài sau:
Agent chỉ cần biết `pipeline.answer("câu hỏi")` mà không quan tâm bên trong ra sao.
"""

from __future__ import annotations

from dataclasses import dataclass

from google import genai
from google.genai import types

from . import prompts
from .chunker import chunk_documents
from .config import Settings, load_settings
from .embeddings import LocalEmbedder
from .llm import GeminiLLM
from .loader import load_documents
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
        http_options = types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=4,
                initial_delay=1.0,
                max_delay=10.0,
                http_status_codes=[429, 500, 502, 503, 504],
            )
        )

        # Client Gemini giờ CHỈ dùng cho chat (sinh câu trả lời)
        client = genai.Client(
            api_key=self.settings.api_key, http_options=http_options
        )
        self.llm = GeminiLLM(client, self.settings.chat_model)

        # Embedding chạy CỤC BỘ, miễn phí — không cần API, không tốn quota
        self.embedder = LocalEmbedder(self.settings.embed_model)

        self._store: VectorStore | None = None  # nạp lười (lazy) khi cần

    # ---------- GIAI ĐOẠN 1: INGEST (offline) ----------
    def ingest(self) -> int:
        """Đọc tài liệu, chia chunk, tạo embedding và lưu vector store.

        Trả về số chunk đã tạo.
        """
        documents = load_documents(self.settings.raw_dir)
        if not documents:
            raise RuntimeError(
                f"Không tìm thấy tài liệu nào trong {self.settings.raw_dir}.\n"
                "-> Hãy bỏ vài file .txt/.md/.pdf vào thư mục data/raw/ rồi thử lại."
            )

        chunks = chunk_documents(
            documents, self.settings.chunk_size, self.settings.chunk_overlap
        )
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        metadatas = [
            {"text": c.text, "source": c.source, "index": c.index} for c in chunks
        ]

        store = VectorStore.build(vectors, metadatas)
        store.save(self.settings.store_path)
        self._store = store  # dùng lại luôn nếu hỏi ngay sau khi ingest
        return len(chunks)

    # ---------- GIAI ĐOẠN 2: QUERY (online) ----------
    def answer(self, question: str) -> RAGAnswer:
        """Trả lời một câu hỏi bằng quy trình Retrieve -> Generate."""
        store = self._get_store()

        # (a) Nhúng câu hỏi thành vector rồi tìm các đoạn liên quan nhất
        query_vector = self.embedder.embed_query(question)
        results = store.search(query_vector, self.settings.top_k)

        # (b) Ghép ngữ cảnh + câu hỏi thành prompt và nhờ LLM trả lời
        context = prompts.build_context(results)
        prompt = prompts.RAG_PROMPT_TEMPLATE.format(
            context=context, question=question
        )
        answer_text = self.llm.generate(prompt, system=prompts.SYSTEM_PROMPT)

        return RAGAnswer(answer=answer_text, sources=results)

    # ---------- Tiện ích nội bộ ----------
    def _get_store(self) -> VectorStore:
        """Nạp vector store từ đĩa một lần rồi giữ trong bộ nhớ (cache)."""
        if self._store is None:
            self._store = VectorStore.load(self.settings.store_path)
        return self._store
