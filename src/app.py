import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path so `src.*` imports resolve correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking import get_recommendations, build_similarity_index
from src.config import DATA_DIR

st.set_page_config(page_title="OpenAlex Citation Map", layout="wide")
st.title("OpenAlex Citation Recommender")


# ── Singletons (loaded once per app lifetime, shared across all sessions) ─────

@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss_index():
    """Load and cache the FAISS index as a process-level singleton."""
    return build_similarity_index(force_recompute=False)


@st.cache_resource(show_spinner="Loading metadata…")
def load_metadata() -> tuple[pd.DataFrame, dict[int, dict]]:
    """
    Load metadata.csv once and return:
    - papers_df  : DataFrame used for the selectbox / display
    - paper_map  : dict[PaperIndex → row dict] for O(1) title/info lookup
    """
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    df = pd.read_csv(
        metadata_path,
        usecols=["PaperIndex", "Title", "Authors", "Year", "Citations", "URL"],
        dtype={"PaperIndex": int},
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Authors"] = df["Authors"].fillna("").astype(str)
    df["URL"] = df["URL"].fillna("").astype(str)
    df.sort_values(by=["Year", "PaperIndex"], ascending=[False, True], inplace=True)

    paper_map: dict[int, dict] = {
        int(row["PaperIndex"]): row.to_dict()
        for _, row in df.iterrows()
    }
    return df, paper_map


# ── Startup checks ────────────────────────────────────────────────────────────

metadata_path = os.path.join(DATA_DIR, "metadata.csv")
faiss_path = os.path.join(DATA_DIR, "faiss.index")

if not os.path.exists(metadata_path):
    st.error("❌ `data/metadata.csv` not found. Run `preprocessing.py` first.")
    st.stop()

if not os.path.exists(faiss_path):
    st.error("❌ `data/faiss.index` not found. Run `python -m src.build_index` first.")
    st.stop()

# Load singletons (cached after first call)
try:
    load_faiss_index()
except Exception as exc:
    st.error(f"Failed to load FAISS index: {exc}")
    st.stop()

try:
    papers_df, paper_map = load_metadata()
except Exception as exc:
    st.error(f"Failed to load metadata.csv: {exc}")
    st.stop()

if papers_df.empty:
    st.warning("No papers found in metadata.csv.")
    st.stop()

# ── UI ────────────────────────────────────────────────────────────────────────

options = papers_df["PaperIndex"].tolist()

selected_index = st.selectbox(
    "Select a paper:",
    options=options,
    format_func=lambda pidx: f"[{pidx}] {paper_map[pidx]['Title']}",
)

# Display info for the selected paper
info = paper_map[selected_index]
year_display = int(info["Year"]) if not pd.isna(info.get("Year", float("nan"))) else "N/A"
st.caption(
    f"**{info['Title']}** | Year: {year_display} | Citations: {info['Citations']}"
)

st.markdown("**Lọc theo năm xuất bản (tùy chọn)**")
col1, col2 = st.columns(2)
with col1:
    use_year_from = st.checkbox("Từ năm", value=False)
    year_from = st.number_input(
        "Năm bắt đầu", min_value=1900, max_value=2025, value=2000, step=1,
        disabled=not use_year_from, label_visibility="collapsed",
    ) if use_year_from else None

with col2:
    use_year_to = st.checkbox("Đến năm", value=False)
    year_to = st.number_input(
        "Năm kết thúc", min_value=1900, max_value=2025, value=2025, step=1,
        disabled=not use_year_to, label_visibility="collapsed",
    ) if use_year_to else None

top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=8, step=1)

if st.button("🔍 Find related papers"):
    with st.spinner("Searching for similar papers…"):
        try:
            recs = get_recommendations(
                selected_index,
                top_n=top_n,
                year_from=int(year_from) if year_from is not None else None,
                year_to=int(year_to) if year_to is not None else None,
            )
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            st.stop()

    if not recs:
        st.info("No recommendations found. Try relaxing the year filter or selecting a different paper.")
    else:
        st.subheader(f"Related Papers ({len(recs)} results)")
        current_year = None
        for rec in recs:
            year = rec.get("Year")
            if year != current_year:
                current_year = year
                st.markdown(f"### 📅 {year if year else 'Unknown Year'}")

            st.markdown(f"**[{rec['PaperIndex']}] {rec['Title']}**")
            st.write(f"Authors: {rec.get('Authors') or 'N/A'}")
            st.write(f"Citations: {rec.get('Citations', 0)} | Cosine score: {rec['Score']:.4f}")
            url = (rec.get("URL") or "").strip()
            if url:
                st.link_button("🔗 Open DOI", url)
            st.divider()