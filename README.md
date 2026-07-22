# HuyG RAG Series — Bài 1: RAG cơ bản với Google Gemini

> Một project RAG (Retrieval-Augmented Generation) tối giản nhưng có cấu trúc
> **chuyên nghiệp**, viết bằng Python + Google Gemini. Đây là nền móng để ở các bài
> sau chúng ta nâng cấp dần lên thành một **AI Agent** biết dùng RAG như một *skill*.

---

## 1. RAG là gì? (giải thích trong 30 giây)

Mô hình ngôn ngữ lớn (LLM) như Gemini rất giỏi diễn đạt, nhưng nó **không biết** về
tài liệu riêng của bạn (giáo trình, sổ tay công ty, ghi chú...) và đôi khi **bịa** ra
thông tin. **RAG** khắc phục điều đó bằng cách:

1. **Tìm** (Retrieval) những đoạn tài liệu liên quan nhất tới câu hỏi.
2. **Đưa** các đoạn đó cho LLM làm ngữ cảnh để **sinh** (Generation) câu trả lời có căn cứ.

```
                          ┌─────────────── GIAI ĐOẠN OFFLINE (chạy 1 lần) ───────────────┐
   data/raw/*.md   ─►  Load  ─►  Chunk  ─►  Embed  ─►  Vector Store (data/processed)
                          └──────────────────────────────────────────────────────────────┘

                          ┌─────────────── GIAI ĐOẠN ONLINE (mỗi câu hỏi) ───────────────┐
   "RAG là gì?"     ─►  Embed câu hỏi ─► Retrieve (top-k) ─► Ghép ngữ cảnh ─► Gemini ─► Trả lời
                          └──────────────────────────────────────────────────────────────┘
```

Sáu bước: **Load → Chunk → Embed → Store → Retrieve → Generate.**

---

## 2. Cấu trúc thư mục

```
HuyG-RAG-Series/
├── .env                      # API key thật (KHÔNG commit) — bạn tự tạo từ .env.example
├── .env.example              # Mẫu cấu hình
├── .gitignore
├── requirements.txt          # Danh sách thư viện
├── pyproject.toml            # Khai báo package (layout src/)
├── README.md                 # File bạn đang đọc
│
├── data/
│   ├── raw/                  # 📂 Tài liệu gốc của bạn (.txt, .md, .pdf)
│   └── processed/            # 💾 Vector store sinh ra tự động (đừng sửa tay)
│
├── src/rag/                  # 🧠 Package chính — mỗi file = 1 bước trong pipeline
│   ├── config.py             #   Đọc .env, gom cấu hình
│   ├── loader.py             #   Bước 1 — Load tài liệu
│   ├── chunker.py            #   Bước 2 — Chia nhỏ (chunk)
│   ├── embeddings.py         #   Bước 3 — Tạo vector (model MIỄN PHÍ, chạy cục bộ)
│   ├── vector_store.py       #   Bước 4 & 5 — Lưu trữ + tìm kiếm cosine
│   ├── llm.py                #   Bước 6 — Gọi Gemini sinh câu trả lời
│   ├── prompts.py            #   Các prompt template
│   └── pipeline.py           #   ⭐ Ghép tất cả lại — đây sẽ là "skill" cho Agent
│
├── scripts/
│   ├── ingest.py             # Chạy giai đoạn OFFLINE (nạp tài liệu)
│   └── ask.py                # Chạy giai đoạn ONLINE (hỏi–đáp)
│
├── tests/
│   └── test_smoke.py         # Test nhanh, không cần gọi API
│
└── notebooks/                # (để dành) minh hoạ trực quan từng bước
```

**Vì sao tách nhỏ như vậy?** Mỗi bước RAG nằm gọn trong một file → dễ đọc, dễ dạy, dễ
thay thế (ví dụ đổi vector store numpy sang FAISS chỉ cần sửa một file), và dễ tái sử
dụng khi lên Agent.

---

## 3. Cài đặt — làm theo đúng thứ tự

> Yêu cầu: Python 3.10+ (bạn đang có 3.12 ✔). Các lệnh dưới đây dùng **PowerShell (Windows)**.

### Bước 3.1 — Tạo môi trường ảo (venv)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> Dấu `(.venv)` xuất hiện đầu dòng nghĩa là đã kích hoạt thành công.
> Nếu PowerShell chặn script, chạy 1 lần:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Bước 3.2 — Cài thư viện

```powershell
pip install -r requirements.txt
```

### Bước 3.3 — Cấu hình API key

```powershell
Copy-Item .env.example .env
```

Mở file `.env`, dán API key lấy miễn phí tại **https://aistudio.google.com/apikey**:

```
GEMINI_API_KEY=dán_key_của_bạn_vào_đây
GEMINI_CHAT_MODEL=gemini-flash-latest
EMBED_MODEL=intfloat/multilingual-e5-small
```

> 💡 **Embedding chạy MIỄN PHÍ trên máy** (thư viện `sentence-transformers`), không cần
> API và không tốn quota. API key Gemini chỉ dùng cho bước **chat** (sinh câu trả lời).
> Lần chạy `ingest.py` **đầu tiên** sẽ tự tải model embedding (~470MB) rồi cache lại —
> lần sau chạy nhanh và không cần mạng để nhúng.

---

## 4. Chạy thử — 2 lệnh

### Bước 4.1 — Nạp tài liệu vào vector store (offline)

```powershell
python scripts/ingest.py
```

Kết quả mong đợi: `✅ Đã tạo N chunk` và một file xuất hiện trong `data/processed/`.

> Sẵn có 2 tài liệu mẫu về RAG trong `data/raw/`. Muốn dùng tài liệu của bạn:
> bỏ file `.txt/.md/.pdf` vào `data/raw/` rồi chạy lại `ingest.py`.

### Bước 4.2 — Đặt câu hỏi (online)

```powershell
python scripts/ask.py "RAG gồm những bước nào?"
```

Hoặc vào chế độ hỏi–đáp liên tục:

```powershell
python scripts/ask.py
```

Bạn sẽ nhận được câu trả lời **kèm nguồn** và điểm tương đồng của từng đoạn được dùng.

---

## 5. Luồng dữ liệu chạy qua code như thế nào?

Khi bạn gọi `pipeline.answer("RAG là gì?")`:

| # | File | Việc làm |
|---|------|----------|
| 1 | `embeddings.py` | Biến câu hỏi thành vector (`embed_query`) |
| 2 | `vector_store.py` | Nhân ma trận + sắp xếp → lấy `top_k` đoạn gần nghĩa nhất |
| 3 | `prompts.py` | Ghép các đoạn thành "NGỮ CẢNH" + câu hỏi thành 1 prompt |
| 4 | `llm.py` | Gửi prompt cho Gemini, nhận câu trả lời |
| 5 | `pipeline.py` | Trả về `RAGAnswer(answer, sources)` |

Đọc code theo đúng thứ tự file trong `src/rag/` (config → loader → chunker → embeddings
→ vector_store → llm → prompts → pipeline) là bạn hiểu trọn vẹn một hệ RAG.

---

## 6. Kiểm thử

```powershell
pytest
```

Các test trong `tests/test_smoke.py` kiểm tra logic chia chunk và tìm kiếm vector —
**không tốn quota API, không cần mạng**.

---

## 7. Bài tập mở rộng (để sinh viên tự luyện)

1. Thêm một file `.pdf` vào `data/raw/` rồi ingest lại — loader đã hỗ trợ sẵn PDF.
2. Đổi `RAG_TOP_K` trong `.env` (ví dụ 2 → 6) và quan sát chất lượng câu trả lời.
3. Thử chia chunk theo câu/đoạn văn thay vì theo số ký tự (sửa `chunker.py`).
4. Thay vector store numpy bằng **Chroma** hoặc **FAISS** (chỉ cần sửa `vector_store.py`).
5. Thêm bước "trả lời kèm mức độ tin cậy" dựa trên điểm cosine cao nhất.

---

## 8. Lộ trình: từ RAG → AI Agent (các bài sau)

Điểm mấu chốt của thiết kế này: lớp **`RAGPipeline`** phơi ra một interface cực gọn —
`ingest()` và `answer()`. Đó chính là hình hài của một **skill/tool**.

- **Bài 2 — Function Calling:** biến `pipeline.answer()` thành một *tool* mà Gemini có
  thể tự quyết định gọi (`search_knowledge_base`).
- **Bài 3 — Agent Loop:** cho model tự lập kế hoạch: khi nào cần tra cứu RAG, khi nào
  trả lời trực tiếp, khi nào hỏi lại người dùng (vòng lặp suy nghĩ → hành động → quan sát).
- **Bài 4 — Multi-tool Agent:** RAG chỉ là *một* trong nhiều skill (thêm máy tính, tìm
  web, gọi API...). Agent điều phối tất cả.

Nói cách khác: **RAG hôm nay = một kỹ năng của Agent ngày mai.** Vì thế ngay từ bài 1
chúng ta đã đóng gói nó thật sạch sẽ.

---

## 9. Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân & cách sửa |
|-----|------------------------|
| `Thiếu GEMINI_API_KEY` | Chưa tạo `.env` hoặc chưa dán key. Xem Bước 3.3. |
| `Chưa có vector store...` | Chưa chạy `ingest.py`. Chạy Bước 4.1 trước. |
| `Không tìm thấy tài liệu nào` | Thư mục `data/raw/` rỗng. Bỏ file .md/.txt/.pdf vào. |
| Lỗi liên quan `google.genai` / `sentence_transformers` | Chưa `pip install -r requirements.txt` trong venv đã kích hoạt. |
| Lần đầu `ingest.py` chạy hơi lâu | Bình thường — đang tải model embedding (~470MB) về cache. Lần sau sẽ nhanh. |
| `503 UNAVAILABLE` / `429 RESOURCE_EXHAUSTED` khi hỏi | Server Gemini quá tải hoặc hết quota free-tier TẠM THỜI (chỉ ở bước chat). Đợi chút rồi thử lại. |
| Câu trả lời "không tìm thấy trong tài liệu" | Đúng như thiết kế — model chỉ trả lời theo tài liệu, không bịa. Hãy thêm tài liệu liên quan. |
