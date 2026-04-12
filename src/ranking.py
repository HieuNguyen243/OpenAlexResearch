import torch
import faiss
import numpy as np
import os
import pandas as pd
import sys

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR


FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

# ── Candidate expansion when year filter is active ────────────────────────────
# Fetching more candidates from FAISS ensures that after year-filtering we
# still have enough high-score results to fill top_n.
YEAR_FILTER_CANDIDATE_MULTIPLIER = 100  # query this many from FAISS when year filter is used


# ── Module-level singletons (loaded once, reused on every call) ───────────────
_faiss_index: faiss.Index | None = None
_embeddings: torch.Tensor | None = None
_metadata_map: dict[int, dict] | None = None  # PaperIndex → metadata row


def _get_faiss_index() -> faiss.Index:
    """Load FAISS index from disk once and cache it in memory."""
    global _faiss_index
    if _faiss_index is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m src.build_index` (or model_gat.py) first."
            )
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    return _faiss_index


def _get_embeddings_and_metadata() -> tuple[torch.Tensor, dict[int, dict], dict[int, int]]:
    """
    Load embeddings and build a fast metadata hash-map once, then cache both.

    Returns
    -------
    embeddings : torch.Tensor  shape (N, D)
    metadata_map : dict  PaperIndex → {Title, Authors, Year, Citations, URL}
    paper_to_row : dict  PaperIndex → embedding row index
    """
    global _embeddings, _metadata_map

    if _embeddings is None or _metadata_map is None:
        embeddings_path = os.path.join(DATA_DIR, "smart_embeddings.pt")
        metadata_path = os.path.join(DATA_DIR, "metadata.csv")

        if not os.path.exists(embeddings_path):
            raise FileNotFoundError("smart_embeddings.pt not found. Run model_gat.py first.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError("metadata.csv not found. Run preprocessing.py first.")

        embeddings = torch.load(embeddings_path).float()
        # Embeddings should already be L2-normalised by model_gat.py, but normalise
        # defensively to guarantee valid inner-product similarity scores.
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        _embeddings = embeddings

        metadata_df = pd.read_csv(metadata_path)
        # Build hash-map for O(1) metadata lookup during result assembly
        _metadata_map = {}
        for _, row in metadata_df.iterrows():
            pidx = int(row["PaperIndex"])
            _metadata_map[pidx] = {
                "Title": row.get("Title", ""),
                "Authors": row.get("Authors", ""),
                "Year": int(row["Year"]) if "Year" in row and not pd.isna(row["Year"]) else None,
                "Citations": int(row["Citations"]) if "Citations" in row and not pd.isna(row["Citations"]) else 0,
                "URL": row.get("URL", ""),
            }

    # Rebuild paper_to_row from the metadata_map keys (preserves insertion order)
    paper_to_row = {pidx: i for i, pidx in enumerate(_metadata_map.keys())}
    return _embeddings, _metadata_map, paper_to_row


# ── Public API ────────────────────────────────────────────────────────────────

def build_similarity_index(force_recompute: bool = False) -> faiss.Index:
    """
    Load the pre-built FAISS index from disk.

    `force_recompute` is kept for API compatibility but is ignored — index
    building is handled offline by `src/build_index.py`.
    """
    return _get_faiss_index()


def get_recommendations(
    target_paper_idx: int,
    top_n: int = 5,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict]:
    """
    Return the top-n most similar papers to `target_paper_idx`.

    Year filtering is optional:
    - No filter  → query exactly top_n+1 from FAISS (fast path).
    - With filter → query YEAR_FILTER_CANDIDATE_MULTIPLIER candidates, apply
                    year constraints, then slice top_n (ensures non-empty results).
    """
    index = _get_faiss_index()
    embeddings, metadata_map, paper_to_row = _get_embeddings_and_metadata()

    target_row = paper_to_row.get(int(target_paper_idx))
    if target_row is None:
        raise ValueError(f"PaperIndex {target_paper_idx} does not exist in metadata.")

    # ── Decide candidate pool size ────────────────────────────────────────────
    year_filter_active = year_from is not None or year_to is not None
    if year_filter_active:
        # Fetch a large candidate set so year filtering still leaves enough results
        k = min(YEAR_FILTER_CANDIDATE_MULTIPLIER + 1, index.ntotal)
    else:
        # Fast path: fetch exactly what we need
        k = min(top_n + 1, index.ntotal)  # +1 to exclude target itself

    # ── FAISS vector search ───────────────────────────────────────────────────
    query = embeddings[target_row].numpy().astype("float32").reshape(1, -1)
    scores, faiss_indices = index.search(query, k)
    scores, faiss_indices = scores[0], faiss_indices[0]

    # index.ntotal rows map 1-to-1 with the embedding matrix / metadata_map
    row_to_paper = {row: pidx for pidx, row in paper_to_row.items()}

    # ── Assemble result list via O(1) dict lookup ─────────────────────────────
    results: list[dict] = []
    for score, faiss_row in zip(scores, faiss_indices):
        if faiss_row < 0:  # FAISS pads with -1 when k > ntotal
            continue
        pidx = row_to_paper.get(int(faiss_row))
        if pidx is None or pidx == int(target_paper_idx):
            continue  # skip self
        meta = metadata_map[pidx]
        results.append(
            {
                "PaperIndex": pidx,
                "Title": meta["Title"],
                "Authors": meta["Authors"],
                "Year": meta["Year"],
                "Citations": meta["Citations"],
                "URL": meta["URL"],
                "Score": float(round(score, 4)),
            }
        )

    # ── Year post-filter (only when requested) ────────────────────────────────
    if year_filter_active:
        if year_from is not None:
            results = [r for r in results if r["Year"] is not None and r["Year"] >= year_from]
        if year_to is not None:
            results = [r for r in results if r["Year"] is not None and r["Year"] <= year_to]

    # Results are already sorted by FAISS score (descending); just slice top_n
    return results[:top_n]