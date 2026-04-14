import os
import sys
import time
import re
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ── Khởi tạo Session chống Block IP (Exponential Backoff) ──
def get_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,                # Thử lại tối đa 5 lần
        backoff_factor=2.0,     # Tăng dần thời gian chờ: 2s, 4s, 8s, 16s...
        status_forcelist=[429, 500, 502, 503, 504], # Trigger retry khi bị Timeout hoặc Rate Limited
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

http_client = get_robust_session()

# Thêm path dự án
try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR
    
from src.ranking import get_recommendations

def normalize_title(title):
    if pd.isna(title) or not title:
        return ""
    # Lowercase, remove punctuation, strip whitespaces
    title = str(title).lower().strip()
    title = re.sub(r'[^\w\s]', '', title)
    return " ".join(title.split())

def calculate_overlap(local_titles, api_titles):
    local_norm = set([normalize_title(t) for t in local_titles if t])
    api_norm = set([normalize_title(t) for t in api_titles if t])
    
    if not local_norm or not api_norm:
        return 0.0, []
        
    intersect = local_norm.intersection(api_norm)
    overlap_pct = (len(intersect) / len(local_norm)) * 100.0
    return overlap_pct, list(intersect)

# 1. Local System
def get_local_recs(paper_index, top_n=20):
    try:
        recs = get_recommendations(paper_index, top_n=top_n)
        return [r['Title'] for r in recs]
    except Exception as e:
        print(f"  [Local Error] {e}")
        return []

# 2. Semantic Scholar API
def get_s2_recs(paper_title, top_n=20):
    try:
        # Step 1: Search paper ID
        encoded_query = urllib.parse.quote(paper_title)
        search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit=1"
        res = http_client.get(search_url, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if not data.get("data"):
            return []
            
        paper_id = data["data"][0]["paperId"]
        time.sleep(1) # Rate limit
        
        # Step 2: Get recommendations
        rec_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}?limit={top_n}&fields=title"
        res2 = http_client.get(rec_url, timeout=15)
        res2.raise_for_status()
        rec_data = res2.json()
        
        if not rec_data.get("recommendedPapers"):
            return []
            
        titles = [p.get("title", "") for p in rec_data["recommendedPapers"]]
        return titles
    except Exception as e:
        print(f"  [S2 Warning] {e}")
        return []

# 3. OpenAlex API
def get_openalex_recs(paper_title, top_n=20):
    try:
        headers = {"User-Agent": "mailto:research@example.com"}
        encoded_title = urllib.parse.quote(paper_title)
        
        # Step 1: Search by exact title snippet
        search_url = f"https://api.openalex.org/works?filter=display_name.search:{encoded_title}"
        res = http_client.get(search_url, headers=headers, timeout=15)
        res.raise_for_status()
        data = res.json()
        
        if not data.get("results"):
            return []
            
        first_work = data["results"][0]
        related_works_urls = first_work.get("related_works", [])
        
        if not related_works_urls:
            return []
            
        # Step 2: Grab up to top_n
        related_ids = [w.replace("https://openalex.org/", "") for w in related_works_urls[:top_n]]
        
        time.sleep(1) # Bổ sung thêm hard-sleep chống bạo lực Request
        query_ids = "|".join(related_ids)
        details_url = f"https://api.openalex.org/works?filter=openalex:{query_ids}&select=display_name&per_page={top_n}"
        
        res2 = http_client.get(details_url, headers=headers, timeout=15)
        res2.raise_for_status()
        details_data = res2.json()
        
        titles = [r.get("display_name", "") for r in details_data.get("results", [])]
        return titles
    except Exception as e:
        print(f"  [OpenAlex Warning] {e}")
        return []

def main():
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Missing {metadata_path}. Please run preprocessing first.")
        return
        
    df = pd.read_csv(metadata_path)
    if len(df) < 3:
        target_indices = df["PaperIndex"].tolist()
    else:
        # Lấy 3 bài báo tiêu biểu (nhiều trích dẫn) để đảm bảo có kết quả từ các API
        target_indices = df.sort_values("Citations", ascending=False)["PaperIndex"].head(3).tolist()

    top_n = 20
    
    print("="*75)
    print("   COMPARISON: LOCAL FAISS vs SEMANTIC SCHOLAR vs OPENALEX")
    print("="*75)
    
    for pidx in target_indices:
        row = df[df["PaperIndex"] == pidx].iloc[0]
        target_title = row["Title"]
        
        print(f"\n🎯 [Target Paper {pidx}]: {target_title}")
        print("-" * 50)
        
        print("Fetching Local Recommendations...")
        local_titles = get_local_recs(pidx, top_n)
        
        print("Fetching Semantic Scholar Recommendations...")
        s2_titles = get_s2_recs(target_title, top_n)
        
        print("Fetching OpenAlex Recommendations...")
        oa_titles = get_openalex_recs(target_title, top_n)
        
        s2_overlap_pct, s2_matches = calculate_overlap(local_titles, s2_titles)
        oa_overlap_pct, oa_matches = calculate_overlap(local_titles, oa_titles)
        
        print("\n📊 Results:")
        print(f"  - Local vs S2 Overlap:       {s2_overlap_pct:.2f}% ({len(s2_matches)}/{len(local_titles)})")
        print(f"  - Local vs OpenAlex Overlap: {oa_overlap_pct:.2f}% ({len(oa_matches)}/{len(local_titles)})")
        
        if s2_matches:
            print("\n  🔍 Intersecting Titles (Local ∩ S2):")
            for t in s2_matches:
                print(f"    * {t}")
                
        if oa_matches:
            print("\n  🔍 Intersecting Titles (Local ∩ OpenAlex):")
            for t in oa_matches:
                print(f"    * {t}")

if __name__ == "__main__":
    main()
