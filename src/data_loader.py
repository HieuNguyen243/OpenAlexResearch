"""
Script lấy dữ liệu từ SQL Server, làm sạch, chuẩn hóa và xuất metadata cho pipeline.
"""
import argparse
import os
import sys
import pandas as pd
import pyodbc

# Đảm bảo import đúng config và đường dẫn dữ liệu
try:
    from src.config import DATA_DIR, get_conn_str
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR, get_conn_str

# Các cột bắt buộc phải có trong metadata
REQUIRED_COLUMNS = ["PaperIndex", "OpenAlexID", "Title", "Authors", "Year", "Citations", "URL"]

# Kết nối tới SQL Server
def _connect() -> pyodbc.Connection:
    return pyodbc.connect(get_conn_str())

# Lấy dữ liệu bài báo từ SQL, cho phép giới hạn số lượng (limit)
def load_papers_from_sql(limit: int | None = None) -> pd.DataFrame:
    top_clause = f"TOP ({limit})" if limit is not None else ""
    query = f"""
    SELECT {top_clause}
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
    ORDER BY p.PaperIndex ASC;
    """

    conn = _connect()
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    return df

# Làm sạch dữ liệu: loại bỏ bản ghi thiếu title/năm, chuẩn hóa citations về số nguyên
def clean_papers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    cleaned = df.copy()
    cleaned["Title"] = cleaned["Title"].fillna("").astype(str).str.strip()
    cleaned = cleaned[(cleaned["Title"] != "") & (cleaned["Year"].notna())]
    cleaned["Citations"] = cleaned["Citations"].fillna(0).astype(int)
    return cleaned

# Kiểm tra schema dataframe có đủ các cột cần thiết không
def validate_schema(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in dbo.Papers: {missing}")

def run_sql_loader(limit: int | None = None, export_csv: bool = False) -> pd.DataFrame:
    print("Loading papers from SQL Server dbo.Papers...")
    raw_df = load_papers_from_sql(limit=limit)
    validate_schema(raw_df)

    cleaned_df = clean_papers(raw_df)

    dropped = len(raw_df) - len(cleaned_df)
    print(f"Loaded rows: {len(raw_df)} | Valid rows: {len(cleaned_df)} | Dropped rows: {dropped}")

    if export_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        out_path = os.path.join(DATA_DIR, "sql_papers_snapshot.csv")
        cleaned_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Exported cleaned snapshot: {out_path}")

    return cleaned_df

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and validate papers from SQL Server only.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick checks.")
    parser.add_argument("--export-csv", action="store_true", help="Export cleaned SQL snapshot to data/sql_papers_snapshot.csv.")
    return parser

if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_sql_loader(limit=args.limit, export_csv=args.export_csv)
