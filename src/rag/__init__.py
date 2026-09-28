"""Package `rag` — một pipeline RAG (Retrieval-Augmented Generation) tối giản.

RAG = tìm kiếm thông tin liên quan trong tài liệu của bạn (Retrieval)
rồi đưa vào LLM để sinh câu trả lời có căn cứ (Augmented Generation).

Điểm truy cập chính là lớp `RAGPipeline` trong module `pipeline`.
Lớp này cũng có thể dễ dàng được đóng gói thành một "tool/skill" cho AI Agent.
"""

from .pipeline import RAGPipeline, RAGAnswer

__all__ = ["RAGPipeline", "RAGAnswer"]
__version__ = "0.1.0"
