
# Giao diện Streamlit cho hệ thống gợi ý bài báo dựa trên embedding và trích dẫn
import streamlit as st
import pandas as pd
import os
import sys
import pyodbc

# Thêm thư mục gốc vào sys.path để import module src dễ dàng
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các hàm chính từ các module khác
from src.ranking import get_recommendations
from src.ranking import build_similarity_index
from src.config import DATA_DIR, get_conn_str

# Thiết lập cấu hình giao diện Streamlit
st.set_page_config(page_title="OpenAlex Citation Map", layout="wide")
st.title("OpenAlex Citation Recommender")

# Hàm cache để tải dữ liệu bài báo từ SQL Server (giảm thời gian tải lại)
@st.cache_data(show_spinner=False)
def load_papers_from_sql():
    conn = pyodbc.connect(get_conn_str())
    try:
        query = """
        SELECT
            p.PaperIndex,
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
        ORDER BY p.PublicationYear DESC, p.PaperIndex ASC;
        """
        return pd.read_sql(query, conn)
    finally:
        conn.close()

# Đường dẫn tới các file dữ liệu cần thiết
metadata_path = os.path.join(DATA_DIR, "metadata.csv")
emb_path = os.path.join(DATA_DIR, "smart_embeddings.pt")

# Kiểm tra dữ liệu đầu vào, nếu thiếu báo lỗi
if not os.path.exists(metadata_path) or not os.path.exists(emb_path):
    st.error("Data artifacts missing. Run preprocessing.py then model_gat.py.")
else:
    try:
        papers_df = load_papers_from_sql()  # Lấy dữ liệu bài báo từ SQL
    except Exception as exc:
        st.error(f"SQL connection failed: {exc}")
        st.stop()

    if papers_df.empty:
        st.warning("No papers found in SQL table dbo.Papers.")
        st.stop()

    metadata_df = pd.read_csv(metadata_path)
    valid_idx_set = set(metadata_df["PaperIndex"].astype(int).tolist()) if "PaperIndex" in metadata_df.columns else set()

    if not valid_idx_set:
        st.error("metadata.csv is invalid. Please run preprocessing.py again.")
        st.stop()

    # Tạo/cập nhật ma trận similarity nếu cần
    try:
        build_similarity_index(force_recompute=False)
    except Exception as exc:
        st.error(f"Failed to build/load similarity index: {exc}")
        st.stop()

    # Chỉ cho phép chọn các PaperIndex hợp lệ
    options = [int(v) for v in papers_df["PaperIndex"].tolist() if int(v) in valid_idx_set]
    if not options:
        st.error("No overlapping PaperIndex between SQL table and metadata.csv. Re-run preprocessing.py.")
        st.stop()

    # Giao diện chọn bài báo để gợi ý
    selected_index = st.selectbox(
        "Select a paper:",
        options=options,
        format_func=lambda pidx: f"[{pidx}] {papers_df.loc[papers_df['PaperIndex'] == pidx, 'Title'].iloc[0]}",
    )

    # Hiển thị thông tin bài báo đã chọn
    selected_row = papers_df[papers_df["PaperIndex"] == selected_index].iloc[0]
    st.caption(
        f"Selected: {selected_row['Title']} | Year: {int(selected_row['Year']) if not pd.isna(selected_row['Year']) else 'N/A'} | Citations: {int(selected_row['Citations']) if not pd.isna(selected_row['Citations']) else 0}"
    )

    # Chọn số lượng bài báo gợi ý
    top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=8, step=1)

    # Khi nhấn nút, thực hiện gợi ý bài báo liên quan
    if st.button("Find related papers"):
        try:
            recs = get_recommendations(selected_index, top_n=top_n)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            st.stop()

        if not recs:
            st.info("No recommendations found.")
        else:
            st.subheader("Related Papers")
            current_year = None
            for rec in recs:
                year = rec.get("Year")
                if year != current_year:
                    current_year = year
                    st.markdown(f"### Year {year}")

                st.markdown(f"**[{rec['PaperIndex']}] {rec['Title']}**")
                st.write(f"Authors: {rec.get('Authors') or 'N/A'}")
                st.write(f"Citations: {rec.get('Citations', 0)}")
                st.write(f"Cosine score: {rec['Score']:.4f}")
                url = (rec.get("URL") or "").strip()
                if url:
                    st.link_button("Open DOI", url)
                st.divider()


metadata_path = os.path.join(DATA_DIR, "metadata.csv")
emb_path = os.path.join(DATA_DIR, "smart_embeddings.pt")

if not os.path.exists(metadata_path) or not os.path.exists(emb_path):
    st.error("Data artifacts missing. Run preprocessing.py then model_gat.py.")
else:
    try:
        papers_df = load_papers_from_sql()
    except Exception as exc:
        st.error(f"SQL connection failed: {exc}")
        st.stop()

    if papers_df.empty:
        st.warning("No papers found in SQL table dbo.Papers.")
        st.stop()

    metadata_df = pd.read_csv(metadata_path)
    valid_idx_set = set(metadata_df["PaperIndex"].astype(int).tolist()) if "PaperIndex" in metadata_df.columns else set()

    if not valid_idx_set:
        st.error("metadata.csv is invalid. Please run preprocessing.py again.")
        st.stop()

    try:
        build_similarity_index(force_recompute=False)
    except Exception as exc:
        st.error(f"Failed to build/load similarity index: {exc}")
        st.stop()

    options = [int(v) for v in papers_df["PaperIndex"].tolist() if int(v) in valid_idx_set]
    if not options:
        st.error("No overlapping PaperIndex between SQL table and metadata.csv. Re-run preprocessing.py.")
        st.stop()

    selected_index = st.selectbox(
        "Select a paper:",
        options=options,
        format_func=lambda pidx: f"[{pidx}] {papers_df.loc[papers_df['PaperIndex'] == pidx, 'Title'].iloc[0]}",
    )

    selected_row = papers_df[papers_df["PaperIndex"] == selected_index].iloc[0]
    st.caption(
        f"Selected: {selected_row['Title']} | Year: {int(selected_row['Year']) if not pd.isna(selected_row['Year']) else 'N/A'} | Citations: {int(selected_row['Citations']) if not pd.isna(selected_row['Citations']) else 0}"
    )

    top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=8, step=1)

    if st.button("Find related papers"):
        try:
            recs = get_recommendations(selected_index, top_n=top_n)
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
            st.stop()

        if not recs:
            st.info("No recommendations found.")
        else:
            st.subheader("Related Papers")
            current_year = None
            for rec in recs:
                year = rec.get("Year")
                if year != current_year:
                    current_year = year
                    st.markdown(f"### Year {year}")

                st.markdown(f"**[{rec['PaperIndex']}] {rec['Title']}**")
                st.write(f"Authors: {rec.get('Authors') or 'N/A'}")
                st.write(f"Citations: {rec.get('Citations', 0)}")
                st.write(f"Cosine score: {rec['Score']:.4f}")
                url = (rec.get("URL") or "").strip()
                if url:
                    st.link_button("Open DOI", url)
                st.divider()