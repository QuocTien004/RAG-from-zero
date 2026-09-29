"""BƯỚC 4 & 5 — STORE + RETRIEVE: lưu vector và tìm kiếm theo độ tương đồng cosine.

Đây là một "vector database" phiên bản tối giản viết bằng numpy, để bạn nhìn thấy
RÕ cơ chế bên trong: tìm kiếm ngữ nghĩa thực chất là phép nhân ma trận + sắp xếp.

Độ tương đồng cosine = cos(góc giữa 2 vector):
  - Bằng 1  -> cùng hướng (rất giống nghĩa)
  - Bằng 0  -> vuông góc (không liên quan)
Mẹo: nếu chuẩn hoá mọi vector về độ dài 1, thì cosine = tích vô hướng (dot product),
tính cực nhanh bằng một phép nhân ma trận.

Khi lên production, có thể thay lớp này bằng FAISS / Chroma / Qdrant / pgvector...
mà phần còn lại của pipeline gần như không đổi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SearchResult:
    """Một kết quả tìm kiếm: đoạn văn + nguồn + điểm tương đồng + đường dẫn ảnh (nếu có)."""

    text: str
    source: str
    score: float
    image_path: str | None = None


class VectorStore:
    def __init__(self, vectors: np.ndarray, metadatas: list[dict]):
        # vectors: ma trận (N, D) ĐÃ chuẩn hoá L2; metadatas: thông tin kèm theo
        self._vectors = vectors
        self._metadatas = metadatas

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """Chuẩn hoá từng hàng về độ dài 1 (để cosine = dot product)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12  # tránh chia cho 0
        return matrix / norms

    def __len__(self) -> int:
        return len(self._metadatas)

    @property
    def sources(self) -> set[str]:
        """Tập hợp tên các file nguồn đang có trong store."""
        return {m["source"] for m in self._metadatas}

    def remove_sources(self, sources_to_remove: set[str]) -> "VectorStore":
        """Loại bỏ các chunk thuộc về danh sách file chỉ định."""
        if not sources_to_remove:
            return self
        keep_indices = [
            i for i, m in enumerate(self._metadatas)
            if m["source"] not in sources_to_remove
        ]
        if not keep_indices:
            dim = self._vectors.shape[1] if self._vectors.ndim == 2 and self._vectors.shape[0] > 0 else 0
            return VectorStore(np.zeros((0, dim), dtype=np.float32), [])

        new_vectors = self._vectors[keep_indices]
        new_metadatas = [self._metadatas[i] for i in keep_indices]
        return VectorStore(new_vectors, new_metadatas)

    def add_chunks(self, vectors: np.ndarray, metadatas: list[dict]) -> "VectorStore":
        """Nối thêm vector và metadata mới vào store (tự chuẩn hoá vector mới)."""
        if len(metadatas) == 0:
            return self
        normed_new = self._normalize(vectors.astype(np.float32))
        if len(self._metadatas) == 0 or self._vectors.shape[0] == 0:
            return VectorStore(normed_new, metadatas)

        combined_vectors = np.vstack([self._vectors, normed_new])
        combined_metadatas = self._metadatas + metadatas
        return VectorStore(combined_vectors, combined_metadatas)

    @classmethod
    def build(cls, vectors: np.ndarray, metadatas: list[dict]) -> "VectorStore":
        """Tạo store mới từ vector thô (sẽ tự chuẩn hoá)."""
        return cls(cls._normalize(vectors.astype(np.float32)), metadatas)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Trả về top_k đoạn có cosine cao nhất so với câu hỏi."""
        if not self._metadatas:
            return []

        q = query_vector.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)      # chuẩn hoá câu hỏi

        scores = self._vectors @ q               # (N,) điểm cosine cho mọi đoạn
        top_idx = np.argsort(-scores)[:top_k]    # lấy chỉ số của top_k điểm cao nhất

        results: list[SearchResult] = []
        for i in top_idx:
            meta = self._metadatas[int(i)]
            results.append(
                SearchResult(
                    text=meta["text"],
                    source=meta["source"],
                    score=float(scores[i]),
                    image_path=meta.get("image_path"),
                )
            )
        return results

    # ----- Lưu / nạp từ đĩa -----

    def save(self, path: Path) -> None:
        """Lưu vector (.npz) và metadata (.meta.json) cạnh nhau."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, vectors=self._vectors)
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(self._metadatas, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Nạp lại store đã lưu. Báo lỗi thân thiện nếu chưa chạy ingest."""
        if not path.exists():
            raise FileNotFoundError(
                f"Chưa có vector store tại {path}.\n"
                "-> Hãy chạy trước: python scripts/ingest.py"
            )
        data = np.load(path)
        meta_path = path.with_suffix(".meta.json")
        metadatas = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(data["vectors"], metadatas)
