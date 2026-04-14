import streamlit as st
import pandas as pd
import os
import sys
from streamlit_agraph import agraph, Node, Edge, Config

# Thêm gốc dự án vào path để import src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking import get_recommendations, build_similarity_index
from src.config import DATA_DIR

# Cấu hình trang - Loại bỏ icon, giữ layout rộng
st.set_page_config(page_title="OpenAlex Citation Analysis", layout="wide")

# CSS: Phong cách tối giản, học thuật, sử dụng tone màu trung tính
st.markdown("""
    <style>
    /* Nút bấm mặc định: Xám tối giản, không bo góc quá tròn */
    div.stButton > button {
        border-radius: 4px;
        font-weight: 600;
        border: 1px solid #cbd5e1;
        background-color: #f8fafc;
        color: #334155;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #94a3b8;
        background-color: #e2e8f0;
        color: #0f172a;
    }
    
    /* Thẻ thông tin bài báo mục tiêu: Viền mảnh, điểm nhấn bên trái */
    .target-card {
        padding: 24px;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #334155;
        margin-bottom: 24px;
        border-radius: 4px;
    }
    .target-label {
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
        display: block;
    }
    .target-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 0;
        margin-bottom: 12px;
    }
    .target-meta {
        font-size: 0.95rem;
        color: #475569;
        margin: 4px 0;
    }
    
    /* Hỗ trợ giao diện tối (Dark Mode) */
    @media (prefers-color-scheme: dark) {
        .target-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-left: 4px solid #94a3b8;
        }
        .target-label { color: #94a3b8; }
        .target-title { color: #f8fafc; }
        .target-meta { color: #cbd5e1; }
        div.stButton > button {
            background-color: #1e293b;
            border-color: #334155;
            color: #e2e8f0;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index():
    return build_similarity_index(force_recompute=False)

@st.cache_resource(show_spinner="Loading metadata...")
def load_metadata() -> tuple[pd.DataFrame, dict[int, dict]]:
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
    st.error("System Error: 'data/metadata.csv' not found. Please run preprocessing script.")
    st.stop()

if not os.path.exists(faiss_path):
    st.error("System Error: 'data/faiss.index' not found. Please run build_index module.")
    st.stop()

try:
    load_faiss_index()
    papers_df, paper_map = load_metadata()
except Exception as exc:
    st.error(f"Initialization failed: {exc}")
    st.stop()

if papers_df.empty:
    st.warning("Database is empty. No metadata found.")
    st.stop()

# ── UI SIDEBAR ────────────────────────────────────────────────────────────────
options = papers_df["PaperIndex"].tolist()

with st.sidebar:
    st.title("Configuration")
    st.markdown("Set parameters for citation network analysis.")
    st.markdown("---")
    
    st.markdown("**Target Document**")
    selected_index = st.selectbox(
        "Select Paper",
        options=options,
        format_func=lambda pidx: f"[{pidx}] {paper_map[pidx]['Title'][:60]}...",
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Filter Criteria**")
    with st.expander("Publication Year", expanded=True):
        col_from, col_to = st.columns(2)
        with col_from:
            use_year_from = st.checkbox("From", value=False)
            year_from = st.number_input(
                "Start Year", min_value=1900, max_value=2025, value=2000, step=1,
                disabled=not use_year_from, label_visibility="collapsed",
            ) if use_year_from else None

        with col_to:
            use_year_to = st.checkbox("To", value=False)
            year_to = st.number_input(
                "End Year", min_value=1900, max_value=2025, value=2025, step=1,
                disabled=not use_year_to, label_visibility="collapsed",
            ) if use_year_to else None

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Output Settings**")
    top_n = st.slider("Result Count (Top N)", min_value=3, max_value=50, value=10, step=1)
    
    st.markdown("---")
    search_clicked = st.button("Run Analysis", use_container_width=True)

# ── MAIN PANEL ────────────────────────────────────────────────────────────────
st.title("OpenAlex Citation Analysis")
st.markdown("Vector-based similarity search for academic literature.")

# Hiển thị Target Paper Card
info = paper_map[selected_index]
year_display = int(info["Year"]) if not pd.isna(info.get("Year", float("nan"))) else "N/A"
url = (info.get("URL") or "").strip()

st.markdown(f"""
<div class="target-card">
    <span class="target-label">Target Document</span>
    <h3 class="target-title">{info['Title']}</h3>
    <p class="target-meta"><b>Authors:</b> {info['Authors'] or 'Unknown'}</p>
    <p class="target-meta"><b>Year:</b> {year_display} &nbsp;|&nbsp; <b>Citations:</b> {info['Citations']}</p>
</div>
""", unsafe_allow_html=True)

if url:
    st.link_button("View Original Source", url)

st.markdown("---")

# Xử lý Kết quả
if search_clicked:
    with st.spinner("Computing vector similarities..."):
        try:
            recs = get_recommendations(
                selected_index,
                top_n=top_n,
                year_from=int(year_from) if year_from is not None else None,
                year_to=int(year_to) if year_to is not None else None,
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    if not recs:
        st.info("No matching results found. Consider relaxing the year filters.")
    else:
        st.subheader(f"Analysis Results ({len(recs)} documents)")
        
        tab1, tab2 = st.tabs(["Network Graph", "Detailed List"])
        
        # --- TAB 1: TRỰC QUAN HÓA ĐỒ THỊ ---
        with tab1:
            nodes = []
            edges = []
            
            def truncate_label(text, length=35):
                return text[:length] + "..." if len(text) > length else text
            
            # Node Trung tâm (Màu Xám đậm - Charcoal)
            nodes.append(Node(
                id=str(selected_index),
                label=truncate_label(info['Title']),
                size=25,
                color="#334155", 
                shape="square",
                title=f"TARGET: {info['Title']} ({year_display})"
            ))
            
            # Các Node vệ tinh (Màu Bạc/Xám nhạt - Silver)
            for rec in recs:
                rec_year = int(rec['Year']) if rec.get('Year') and not pd.isna(rec['Year']) else 'N/A'
                nodes.append(Node(
                    id=str(rec['PaperIndex']),
                    label=truncate_label(rec['Title']),
                    size=15,
                    color="#94a3b8",
                    title=f"{rec['Title']}\nYear: {rec_year} | Score: {rec['Score']:.4f}"
                ))
                
                edges.append(Edge(
                    source=str(selected_index),
                    target=str(rec['PaperIndex']),
                    color="#e2e8f0"
                ))
                
            config = Config(
                width="100%", 
                height=600, 
                directed=True, 
                physics=True, 
                hierarchical=False,
                interaction={'hover': True, 'zoomView': True}
            )
            
            agraph(nodes=nodes, edges=edges, config=config)
            
        # --- TAB 2: DANH SÁCH CHI TIẾT ---
        with tab2:
            for i, rec in enumerate(recs):
                rec_year = int(rec['Year']) if rec.get('Year') and not pd.isna(rec['Year']) else 'N/A'
                
                # Sử dụng đường viền tối giản thay vì thẻ nổi
                st.markdown(f"#### {i+1}. {rec['Title']}")
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.markdown(f"<span style='color: #475569;'><b>Authors:</b> {rec.get('Authors') or 'Unknown'}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color: #64748b; font-size: 0.9em;'>ID: {rec['PaperIndex']} &nbsp;|&nbsp; Year: {rec_year}</span>", unsafe_allow_html=True)
                    
                    rec_url = (rec.get("URL") or "").strip()
                    if rec_url:
                        st.markdown(f"<a href='{rec_url}' target='_blank' style='font-size: 0.9em; text-decoration: none; color: #3b82f6;'>View Source</a>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Similarity", value=f"{rec['Score']:.3f}")
                
                st.markdown("<hr style='margin-top: 10px; margin-bottom: 20px; border-top: 1px solid #f1f5f9;'>", unsafe_allow_html=True)