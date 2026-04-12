import os
import sys
import torch
import faiss

# Thêm gốc dự án vào path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR

def build_faiss_index():
    embeddings_path = os.path.join(DATA_DIR, "smart_embeddings.pt")
    index_path = os.path.join(DATA_DIR, "faiss.index")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Không tìm thấy {embeddings_path}. Hãy chạy model_gat.py trước.")

    print(f"Đang tải embeddings từ {embeddings_path}...")
    embeddings = torch.load(embeddings_path).float()
    
    # Chuẩn hóa L2 để Inner Product tương đương với Cosine Similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.numpy().astype("float32")

    d = embeddings_np.shape[1]
    print(f"Khởi tạo FAISS IndexFlatIP (chạy trên CPU) với chiều dữ liệu {d}...")
    
    index = faiss.IndexFlatIP(d)
    index.add(embeddings_np)

    faiss.write_index(index, index_path)
    print(f"Hoàn tất! Đã lưu FAISS index chứa {index.ntotal} vector tại: {index_path}")

if __name__ == "__main__":
    build_faiss_index()