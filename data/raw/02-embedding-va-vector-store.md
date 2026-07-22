# Embedding và Vector Store

## Embedding là gì?

Embedding (vector nhúng) là cách biến một đoạn văn bản thành một dãy số thực, ví dụ
768 chiều. Dãy số này biểu diễn "ý nghĩa" của đoạn văn. Đặc điểm quan trọng: hai đoạn
văn có nội dung gần nghĩa nhau sẽ cho ra hai vector nằm gần nhau trong không gian.

Nhờ đó máy tính có thể so sánh NGỮ NGHĨA thay vì chỉ so khớp từ khoá. Ví dụ "xe hơi"
và "ô tô" tuy khác chữ nhưng vector của chúng rất gần nhau.

Trong project này, embedding được tạo bằng mô hình MIỄN PHÍ chạy cục bộ trên máy
(intfloat/multilingual-e5-small qua thư viện sentence-transformers), nên không tốn
quota API và chạy được cả khi offline. Model họ E5 thêm tiền tố "passage:" cho tài liệu
và "query:" cho câu hỏi để tối ưu chất lượng tìm kiếm. Google Gemini chỉ còn được dùng
ở bước cuối để sinh câu trả lời (chat).

## Độ tương đồng cosine

Để đo hai vector "gần nhau" tới mức nào, ta dùng độ tương đồng cosine — cosin của góc
giữa hai vector. Giá trị bằng 1 nghĩa là cùng hướng (rất giống nghĩa), bằng 0 nghĩa là
vuông góc (không liên quan). Nếu chuẩn hoá các vector về độ dài 1 thì cosine chính bằng
tích vô hướng, tính rất nhanh bằng một phép nhân ma trận.

## Vector Store

Vector store là nơi lưu trữ các vector kèm nội dung gốc, và cho phép tìm kiếm nhanh
những vector gần nhất với một câu hỏi. Trong bài cơ bản, chúng ta tự viết một vector
store tối giản bằng numpy để hiểu rõ cơ chế. Khi lên production, có thể thay bằng các
công cụ chuyên dụng như FAISS, Chroma, Qdrant hay pgvector mà không phải sửa nhiều.
