import os
import sys
import json
import torch
from tqdm import tqdm

# Đảm bảo import được các module trong src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA_DIR, BASE_DIR
from src.ranking import _get_faiss_index, _get_embeddings_and_metadata

def run_precompute(top_k=100):
    print("Đang tải dữ liệu...")
    index = _get_faiss_index()
    embeddings, metadata_map, paper_to_row = _get_embeddings_and_metadata()
    
    # Tải quan hệ trích dẫn (edge_index) từ preprocessing
    edge_index_path = os.path.join(DATA_DIR, "edge_index.pt")
    if not os.path.exists(edge_index_path):
        print("⚠️ Không tìm thấy edge_index.pt, danh sách trích dẫn sẽ trống.")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.load(edge_index_path)

    # Mapping
    row_to_paper = {row: pidx for pidx, row in paper_to_row.items()}
    num_papers = len(paper_to_row)
    embeddings_np = embeddings.numpy().astype("float32")

    # 1. Xử lý danh sách trích dẫn (A trích dẫn B)
    print("Đang xử lý quan hệ trích dẫn...")
    citation_map = {}
    if edge_index.shape[1] > 0:
        src_rows = edge_index[0].tolist()
        dst_rows = edge_index[1].tolist()
        for s, d in zip(src_rows, dst_rows):
            src_pidx = str(row_to_paper[s])
            dst_pidx = row_to_paper[d]
            if src_pidx not in citation_map:
                citation_map[src_pidx] = []
            citation_map[src_pidx].append(dst_pidx)

    # 2. Xử lý gợi ý (Recommender)
    print(f"Bắt đầu pre-compute gợi ý cho {num_papers} bài báo...")
    batch_size = 1000
    recs_dict = {}

    for start_idx in tqdm(range(0, num_papers, batch_size), desc="Searching"):
        end_idx = min(start_idx + batch_size, num_papers)
        query_batch = embeddings_np[start_idx:end_idx]
        scores_batch, indices_batch = index.search(query_batch, top_k + 1)

        for i in range(end_idx - start_idx):
            current_row = start_idx + i
            source_pidx = row_to_paper[current_row]
            
            recs = []
            for score, faiss_row in zip(scores_batch[i], indices_batch[i]):
                if faiss_row < 0: continue
                rel_pidx = row_to_paper.get(int(faiss_row))
                if rel_pidx is None or rel_pidx == source_pidx: continue
                recs.append({"id": rel_pidx, "score": float(round(score, 4))})
                if len(recs) == top_k: break
            recs_dict[str(source_pidx)] = recs

    # Lưu tất cả vào một file JSON duy nhất
    final_data = {
        "recommendations": recs_dict,
        "citations": citation_map
    }

    out_path = os.path.join(DATA_DIR, "precomputed_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False)

    print(f"\nHoàn tất! Đã lưu dữ liệu tại: {out_path}")

if __name__ == "__main__":
    run_precompute()