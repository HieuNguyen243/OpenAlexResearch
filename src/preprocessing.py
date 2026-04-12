import json
import os
from typing import Dict, List, Tuple

import pandas as pd
import pyodbc
import torch
from sentence_transformers import SentenceTransformer

import sys

try:
    from src.config import DATA_DIR, get_conn_str
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR, get_conn_str

from src.data_loader import load_papers_from_sql


def _connect() -> pyodbc.Connection:
    return pyodbc.connect(get_conn_str())



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


def build_node_features(df_papers: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> torch.Tensor:
    model = SentenceTransformer(model_name)
    titles = df_papers["Title"].astype(str).tolist()
    return model.encode(titles, convert_to_tensor=True)


def build_edge_index(df_papers: pd.DataFrame, df_refs: pd.DataFrame) -> torch.Tensor:
    id_to_row: Dict[str, int] = dict(zip(df_papers["OpenAlexID"], range(len(df_papers))))
    paperindex_to_row: Dict[int, int] = {
        int(pidx): row_idx for row_idx, pidx in enumerate(df_papers["PaperIndex"].tolist())
    }

    edges: List[Tuple[int, int]] = []
    for _, ref_row in df_refs.iterrows():
        source_paper_index = int(ref_row["PaperIndex"])
        source_row = paperindex_to_row.get(source_paper_index)
        if source_row is None:
            continue

        target_id = str(ref_row["Referenced_OpenAlex_ID"]).strip()
        target_row = id_to_row.get(target_id)
        if target_row is not None:
            edges.append((source_row, target_row))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def run_preprocessing() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading papers from SQL Server...")
    df_papers = load_papers_from_sql()
    if df_papers.empty:
        raise RuntimeError("No valid rows found in dbo.Papers. Run data_loader.py first.")

    print("Building title vectors with all-MiniLM-L6-v2...")
    node_features = build_node_features(df_papers)
    node_features_path = os.path.join(DATA_DIR, "node_features.pt")
    torch.save(node_features, node_features_path)

    print("Building edge_index with in-range references only...")
    df_refs = load_references_from_sql()
    edge_index = build_edge_index(df_papers, df_refs)
    edge_index_path = os.path.join(DATA_DIR, "edge_index.pt")
    torch.save(edge_index, edge_index_path)

    # Phát hiện và báo cáo isolated nodes
    num_nodes = len(df_papers)
    node_degrees = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.shape[1] > 0:
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        ones = torch.ones(edge_index.shape[1], dtype=torch.long)
        node_degrees.scatter_add_(0, src_nodes, ones)
        node_degrees.scatter_add_(0, dst_nodes, ones)

    isolated_mask = (node_degrees == 0)
    n_isolated = isolated_mask.sum().item()
    print(f"Isolated nodes (không có citation edge): {n_isolated}/{num_nodes} "
          f"({100*n_isolated/num_nodes:.1f}%)")

    # Lưu isolated mask để các bước sau có thể xử lý riêng nếu cần
    isolated_path = os.path.join(DATA_DIR, "isolated_mask.pt")
    torch.save(isolated_mask, isolated_path)
    print(f"Saved isolated mask: {isolated_path}")

    metadata = df_papers[["PaperIndex", "Title", "Authors", "Year", "Citations", "URL"]].copy()
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    print(f"Saved node features: {node_features_path} shape={tuple(node_features.shape)}")
    print(f"Saved edge index: {edge_index_path} edges={edge_index.shape[1]}")
    print(f"Saved metadata: {metadata_path} rows={len(metadata)}")


if __name__ == "__main__":
    run_preprocessing()