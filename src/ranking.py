import torch
import os
import pandas as pd
import sys

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR


SIMILARITY_PATH = os.path.join(DATA_DIR, "similarity_matrix.pt")


def _load_embeddings_and_metadata():
    embeddings_path = os.path.join(DATA_DIR, "smart_embeddings.pt")
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("smart_embeddings.pt not found. Run model_gat.py first.")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.csv not found. Run preprocessing.py first.")

    embeddings = torch.load(embeddings_path).float()
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    metadata = pd.read_csv(metadata_path)
    return embeddings, metadata


def build_similarity_index(force_recompute=False):
    if os.path.exists(SIMILARITY_PATH) and not force_recompute:
        return torch.load(SIMILARITY_PATH)

    embeddings, _ = _load_embeddings_and_metadata()
    similarity_matrix = embeddings @ embeddings.T
    torch.save(similarity_matrix, SIMILARITY_PATH)
    return similarity_matrix


def get_recommendations(target_paper_idx, top_n=5):
    similarity_matrix = build_similarity_index(force_recompute=False)
    _, metadata = _load_embeddings_and_metadata()

    if "PaperIndex" not in metadata.columns:
        raise ValueError("metadata.csv must include PaperIndex column.")

    paper_to_row = {int(v): i for i, v in enumerate(metadata["PaperIndex"].tolist())}
    target_row = paper_to_row.get(int(target_paper_idx))
    if target_row is None:
        raise ValueError(f"PaperIndex {target_paper_idx} does not exist.")

    similarities = similarity_matrix[target_row].cpu().numpy()

    candidates = metadata.copy()
    candidates["Score"] = similarities
    candidates = candidates[candidates["PaperIndex"] != int(target_paper_idx)]

    # Ranking logic: group by Year (newer first), then cosine score desc.
    candidates = candidates.sort_values(by=["Score"], ascending=[False])
    
    top_df = candidates.head(top_n)

    if "Year" in top_df.columns:
        top_df = top_df.sort_values(by=["Year", "Score"], ascending=[False, False])
    results = []
    for _, row in top_df.iterrows():
        results.append(
            {
                "PaperIndex": int(row["PaperIndex"]),
                "Title": row.get("Title", ""),
                "Authors": row.get("Authors", ""),
                "Year": int(row["Year"]) if not pd.isna(row.get("Year")) else None,
                "Citations": int(row["Citations"]) if "Citations" in row and not pd.isna(row.get("Citations")) else 0,
                "URL": row.get("URL", ""),
                "Score": float(round(row["Score"], 4)),
            }
        )

    return results