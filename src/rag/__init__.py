"""Package `rag` — một pipeline RAG (Retrieval-Augmented Generation) tối giản.

RAG = tìm kiếm thông tin liên quan trong tài liệu của bạn (Retrieval)
rồi đưa vào LLM để sinh câu trả lời có căn cứ (Augmented Generation).

Điểm truy cập chính là lớp `RAGPipeline` trong module `pipeline`.
Ở các bài sau, chính lớp này sẽ trở thành một "skill" mà AI Agent gọi tới.
"""

from .pipeline import RAGPipeline, RAGAnswer

__all__ = ["RAGPipeline", "RAGAnswer"]
__version__ = "0.1.0"
