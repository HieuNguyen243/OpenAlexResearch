
# Tiền xử lý dữ liệu: lấy dữ liệu từ SQL, tạo embedding tiêu đề, tạo ma trận cạnh trích dẫn, xuất metadata
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import pyodbc
import torch
from sentence_transformers import SentenceTransformer

import sys

# Đảm bảo import đúng config và đường dẫn dữ liệu
try:
    from src.config import DATA_DIR, get_conn_str
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR, get_conn_str



# Kết nối tới SQL Server
def _connect() -> pyodbc.Connection:
    return pyodbc.connect(get_conn_str())



# Lấy toàn bộ thông tin bài báo từ SQL Server, gộp tác giả, loại bỏ bản ghi thiếu title/năm
def load_papers_from_sql() -> pd.DataFrame:
    conn = _connect()
    try:
        query = """
        SELECT
            p.PaperIndex,
            p.OpenAlex_ID AS OpenAlexID,
            CAST(p.Title AS NVARCHAR(MAX)) AS Title,
            COALESCE(a.Authors, '') AS Authors,
            p.PublicationYear AS [Year],
            ISNULL(p.Citations, 0) AS Citations,
            CAST(p.DOI AS NVARCHAR(512)) AS URL
        FROM dbo.Papers p
        OUTER APPLY (
            SELECT STUFF((
                SELECT ', ' + pa.AuthorName
                FROM dbo.PaperAuthors pa
                WHERE pa.PaperIndex = p.PaperIndex
                FOR XML PATH(''), TYPE
            ).value('.', 'NVARCHAR(MAX)'), 1, 2, '') AS Authors
        ) a
        WHERE p.Title IS NOT NULL AND LTRIM(RTRIM(CAST(p.Title AS NVARCHAR(MAX)))) <> '' AND p.PublicationYear IS NOT NULL
        ORDER BY p.PaperIndex ASC;
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df



# Lấy thông tin trích dẫn (cạnh) giữa các bài báo từ SQL Server
def load_references_from_sql() -> pd.DataFrame:
    conn = _connect()
    try:
        query = """
        SELECT
            PaperIndex,
            Referenced_OpenAlex_ID
        FROM dbo.PaperReferences
        WHERE Referenced_OpenAlex_ID IS NOT NULL AND LTRIM(RTRIM(Referenced_OpenAlex_ID)) <> '';
        """
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df



# Tạo embedding vector cho tiêu đề bài báo bằng mô hình SentenceTransformer
def build_node_features(df_papers: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> torch.Tensor:
    model = SentenceTransformer(model_name)
    titles = df_papers["Title"].astype(str).tolist()
    return model.encode(titles, convert_to_tensor=True)



# Xây dựng ma trận cạnh (edge_index) cho graph, chỉ giữ cạnh mà cả source và target đều tồn tại trong tập dữ liệu
def build_edge_index(df_papers: pd.DataFrame, df_refs: pd.DataFrame) -> torch.Tensor:
    # Map OpenAlexID và PaperIndex sang chỉ số dòng
    id_to_row: Dict[str, int] = dict(zip(df_papers["OpenAlexID"], range(len(df_papers))))
    paperindex_to_row: Dict[int, int] = {
        int(pidx): row_idx for row_idx, pidx in enumerate(df_papers["PaperIndex"].tolist())
    }

    edges: List[Tuple[int, int]] = []
    for _, ref_row in df_refs.iterrows():
        source_paper_index = int(ref_row["PaperIndex"])
        source_row = paperindex_to_row.get(source_paper_index)
        if source_row is None:
            continue  # Bỏ qua nếu source không tồn tại

        target_id = str(ref_row["Referenced_OpenAlex_ID"]).strip()
        target_row = id_to_row.get(target_id)
        if target_row is not None:
            edges.append((source_row, target_row))  # Chỉ thêm cạnh hợp lệ

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)  # Không có cạnh nào

    # Trả về tensor 2xN (source, target)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()



# Hàm chính: thực hiện toàn bộ pipeline tiền xử lý
def run_preprocessing() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)  # Đảm bảo thư mục data tồn tại

    print("Loading papers from SQL Server...")
    df_papers = load_papers_from_sql()  # Lấy dữ liệu bài báo
    if df_papers.empty:
        raise RuntimeError("No valid rows found in dbo.Papers. Run data_loader.py first.")

    print("Building title vectors with all-MiniLM-L6-v2...")
    node_features = build_node_features(df_papers)  # Tạo embedding tiêu đề
    node_features_path = os.path.join(DATA_DIR, "node_features.pt")
    torch.save(node_features, node_features_path)

    print("Building edge_index with in-range references only...")
    df_refs = load_references_from_sql()  # Lấy dữ liệu trích dẫn
    edge_index = build_edge_index(df_papers, df_refs)  # Tạo ma trận cạnh
    edge_index_path = os.path.join(DATA_DIR, "edge_index.pt")
    torch.save(edge_index, edge_index_path)

    # Lưu metadata cho các bài báo (dùng cho app và ranking)
    metadata = df_papers[["PaperIndex", "Title", "Authors", "Year", "Citations", "URL"]].copy()
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print(f"Saved node features: {node_features_path} shape={tuple(node_features.shape)}")
    print(f"Saved edge index: {edge_index_path} edges={edge_index.shape[1]}")
    print(f"Saved metadata: {metadata_path} rows={len(metadata)}")



# Chạy script trực tiếp sẽ thực hiện pipeline tiền xử lý
if __name__ == "__main__":
    run_preprocessing()