# RAG là gì?

RAG (Retrieval-Augmented Generation — Sinh câu trả lời có tăng cường truy hồi) là
một kỹ thuật kết hợp giữa TÌM KIẾM thông tin và MÔ HÌNH NGÔN NGỮ LỚN (LLM).

Thay vì để LLM tự trả lời chỉ dựa trên kiến thức đã học (dễ bịa, dễ lỗi thời), RAG
thực hiện hai việc: trước tiên tìm những đoạn tài liệu liên quan tới câu hỏi, sau đó
đưa các đoạn này vào cho LLM làm ngữ cảnh để sinh câu trả lời có căn cứ.

## Vì sao cần RAG?

- Giảm hiện tượng "ảo giác" (hallucination): câu trả lời bám vào tài liệu thật.
- Cập nhật kiến thức mới mà không cần huấn luyện lại mô hình: chỉ cần thêm tài liệu.
- Trả lời được câu hỏi trên dữ liệu riêng (nội bộ doanh nghiệp, giáo trình, sổ tay...).
- Có thể trích dẫn nguồn, giúp người dùng kiểm chứng.

## Sáu bước cốt lõi của một pipeline RAG

1. Load: đọc tài liệu thô (txt, md, pdf...).
2. Chunk: chia tài liệu dài thành các đoạn nhỏ có phần chồng lấn.
3. Embed: biến mỗi đoạn thành một vector số bằng mô hình embedding.
4. Store: lưu các vector vào một "vector store" để tìm kiếm nhanh.
5. Retrieve: với mỗi câu hỏi, tìm các đoạn có vector gần nghĩa nhất (cosine).
6. Generate: đưa các đoạn tìm được cùng câu hỏi vào LLM để sinh câu trả lời.

Bốn bước đầu chạy MỘT LẦN (offline) khi có tài liệu mới. Hai bước cuối chạy mỗi khi
có câu hỏi (online).
