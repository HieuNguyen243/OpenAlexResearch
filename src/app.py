import streamlit as st
import pandas as pd
import os
import sys
from streamlit_agraph import agraph, Node, Edge, Config

# Thêm gốc dự án vào path để import src.*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ranking import get_recommendations, build_similarity_index
from src.config import DATA_DIR

st.set_page_config(page_title="OpenAlex Citation Map", page_icon="🔗", layout="wide")

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss_index():
    return build_similarity_index(force_recompute=False)

@st.cache_resource(show_spinner="Loading metadata…")
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
    st.error("❌ `data/metadata.csv` not found. Run `preprocessing.py` first.")
    st.stop()

if not os.path.exists(faiss_path):
    st.error("❌ `data/faiss.index` not found. Run `python -m src.build_index` first.")
    st.stop()

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

# ── UI SIDEBAR (Bảng điều khiển) ──────────────────────────────────────────────
options = papers_df["PaperIndex"].tolist()

with st.sidebar:
    st.header("⚙️ Bảng điều khiển")
    
    selected_index = st.selectbox(
        "Chọn một bài báo:",
        options=options,
        format_func=lambda pidx: f"[{pidx}] {paper_map[pidx]['Title'][:60]}...",
    )
    
    st.markdown("---")
    st.markdown("**Lọc theo năm xuất bản**")
    
    col_from, col_to = st.columns(2)
    with col_from:
        use_year_from = st.checkbox("Từ năm", value=False)
        year_from = st.number_input(
            "Năm bắt đầu", min_value=1900, max_value=2025, value=2000, step=1,
            disabled=not use_year_from, label_visibility="collapsed",
        ) if use_year_from else None

    with col_to:
        use_year_to = st.checkbox("Đến năm", value=False)
        year_to = st.number_input(
            "Năm kết thúc", min_value=1900, max_value=2025, value=2025, step=1,
            disabled=not use_year_to, label_visibility="collapsed",
        ) if use_year_to else None

    top_n = st.slider("Số lượng gợi ý:", min_value=3, max_value=50, value=10, step=1)
    
    search_clicked = st.button("🔍 Find related papers", use_container_width=True, type="primary")

# ── MAIN PANEL (Không gian chính) ─────────────────────────────────────────────
st.title("🚀 OpenAlex Citation Recommender")

# Lấy thông tin bài báo đang chọn
info = paper_map[selected_index]
year_display = int(info["Year"]) if not pd.isna(info.get("Year", float("nan"))) else "N/A"

with st.container(border=True):
    st.markdown("### 🎯 Target Paper")
    st.markdown(f"**{info['Title']}**")
    st.caption(f"✍️ **Tác giả:** {info['Authors'] or 'N/A'} | 📅 **Năm:** {year_display} | 📊 **Trích dẫn:** {info['Citations']}")
    
    url = (info.get("URL") or "").strip()
    if url:
        st.link_button("🔗 DOI", url)

if search_clicked:
    with st.spinner("Đang tìm kiếm mạng lưới tương đồng…"):
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
        st.info("Không tìm thấy kết quả. Thử nới lỏng bộ lọc năm hoặc chọn bài báo khác.")
    else:
        st.subheader(f"Kết quả phân tích ({len(recs)} bài báo)")
        
        # Hệ thống Tabs
        tab1, tab2 = st.tabs(["🌐 Network Graph", "📋 List View"])
        
        # --- TAB 1: TRỰC QUAN HÓA ĐỒ THỊ ---
        with tab1:
            nodes = []
            edges = []
            
            # Hàm rút gọn nhãn để tránh node bị tràn chữ
            def truncate_label(text, length=30):
                return text[:length] + "..." if len(text) > length else text
            
            # Node Trung tâm (Màu đỏ nổi bật, kích thước lớn)
            nodes.append(Node(
                id=str(selected_index),
                label=truncate_label(info['Title']),
                size=25,
                color="#FF4B4B",
                title=f"TARGET: {info['Title']} ({year_display})"
            ))
            
            # Các Node vệ tinh (Kết quả gợi ý)
            for rec in recs:
                rec_year = int(rec['Year']) if rec.get('Year') and not pd.isna(rec['Year']) else 'N/A'
                nodes.append(Node(
                    id=str(rec['PaperIndex']),
                    label=truncate_label(rec['Title']),
                    size=15,
                    color="#0068C9",
                    title=f"{rec['Title']}\nNăm: {rec_year} | Điểm Score: {rec['Score']:.4f}"
                ))
                
                # Tạo cạnh nối từ Target đến Vệ tinh
                edges.append(Edge(
                    source=str(selected_index),
                    target=str(rec['PaperIndex'])
                ))
                
            # Cấu hình đồ thị (Bật vật lý physics để các node tự đẩy nhau đẹp mắt)
            config = Config(width=800, height=500, directed=True, physics=True, hierarchical=False)
            
            # Render Agraph
            agraph(nodes=nodes, edges=edges, config=config)
            
        # --- TAB 2: DANH SÁCH CHI TIẾT ---
        with tab2:
            for rec in recs:
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        rec_year = int(rec['Year']) if rec.get('Year') and not pd.isna(rec['Year']) else 'N/A'
                        st.markdown(f"**[{rec['PaperIndex']}] {rec['Title']}** ({rec_year})")
                        st.caption(f"Tác giả: {rec.get('Authors') or 'N/A'}")
                        
                    with col2:
                        # Hiển thị độ tương đồng với Metric Component
                        st.metric("Cosine Score", value=f"{rec['Score']:.4f}")
                        
                        url = (rec.get("URL") or "").strip()
                        if url:
                            st.link_button("🔗 DOI", url)