# CẤU TRÚC DỰ ÁN

File: config.py: Kết nối đến sql server để lấy dữ liệu

    data_loader.py: Lấy dữ liệu từ sql, loại bỏ những bản ghi thiếu title hoặc thiếu năm, chuẩn hóa ciation về số nguyên và trả về file metadata.csv

    preprrocessing.py: là node_feaures.pt (lưu trữ vector embedding từ tiêu đề) và edge_index.pt (mỗi cột là một cạnh có hướng source -> target giữa 2 bài báo.)

# CẤU TRÚC DỰ ÁN

Thư mục `src/` gồm các file chính:

- **config.py**: Cấu hình kết nối SQL Server, xác định thư mục dữ liệu.
- **data_loader.py**: Lấy dữ liệu từ SQL, loại bỏ bản ghi thiếu `title` hoặc `năm`, chuẩn hóa `citations` về số nguyên, xuất file `metadata.csv`.
- **preprocessing.py**: Tiền xử lý, tạo vector embedding từ tiêu đề (`node_features.pt`), tạo ma trận cạnh trích dẫn (`edge_index.pt`), xuất `metadata.csv`.
- **model_gat.py**: Định nghĩa và huấn luyện mô hình GATv2, sinh embedding chuẩn hóa (`smart_embeddings.pt`), lưu checkpoint mô hình (`gat_autoencoder.pt`).
- **ranking.py**: Tính ma trận cosine similarity, xếp hạng bài báo theo năm và độ tương đồng embedding, cung cấp hàm gợi ý bài liên quan.
- **app.py**: Giao diện Streamlit, hiển thị kết quả, cho phép tìm kiếm và xem gợi ý bài báo.

Thư mục `data/` chứa các file dữ liệu trung gian:

- `node_features.pt`, `edge_index.pt`, `metadata.csv`, `smart_embeddings.pt`, `gat_autoencoder.pt`, ...

## Mục tiêu

Pipeline gồm 5 giai đoạn:

1. Đọc và kiểm tra dữ liệu từ SQL Server.
2. Tiền xử lý: tạo vector tiêu đề (all-MiniLM-L6-v2), tạo edge_index từ bảng trích dẫn.
3. Huấn luyện GATv2 và xuất embedding đã chuẩn hóa.
4. Tính cosine similarity và xếp hạng theo năm + độ tương đồng.
5. Hiển thị kết quả trên Streamlit.

## Nguồn dữ liệu SQL

Dự án đang dùng đúng schema hiện có trong SQL Server:

- Bảng `dbo.Papers`
  - `PaperIndex` (int identity)
  - `OpenAlex_ID` (varchar)
  - `Title` (ntext)
  - `PublicationYear` (int)
  - `Citations` (int)
  - `DOI` (text)
- Bảng `dbo.PaperAuthors`
  - `PaperIndex` (int)
  - `AuthorName` (nvarchar)
- Bảng `dbo.PaperReferences`
  - `PaperIndex` (int)
  - `Referenced_OpenAlex_ID` (varchar)

## Cấu hình kết nối SQL

Cấu hình ở file `src/config.py`:

- Server: `LAPTOP-TEU72OKK\\SQLEXPRESS`
- Database: `OpenAlexDB`
- Trusted_Connection: `yes`

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## Cách chạy toàn bộ pipeline

### Bước 1: Kiểm tra đọc dữ liệu từ SQL

```bash
python src/data_loader.py --limit 20
```

Tác dụng:

- Kiểm tra kết nối SQL.
- Đọc dữ liệu từ `dbo.Papers` và gộp tác giả từ `dbo.PaperAuthors`.
- Lọc dữ liệu rỗng (`Title`, `Year`) để báo số dòng hợp lệ.

### Bước 2: Tiền xử lý

```bash
python src/preprocessing.py
```

Đầu ra trong thư mục `data/`:

- `node_features.pt`
- `edge_index.pt`
- `metadata.csv`

### Bước 3: Huấn luyện GAT và sinh embedding

```bash
python src/model_gat.py
```

Đầu ra:

- `data/smart_embeddings.pt`
- `data/gat_autoencoder.pt`
- Đọc dữ liệu đã tiền xử lý (`node_features.pt`, `edge_index.pt`).
- Huấn luyện mô hình GATv2 tự động (GATAutoEncoder).
- Xuất embedding đã chuẩn hóa L2 ra file `data/smart_embeddings.pt`.
- Lưu checkpoint mô hình vào `data/gat_autoencoder.pt`.

### Bước 4: (Tùy chọn) kiểm tra ranking bằng script

- `data/smart_embeddings.pt`: embedding vector cho từng bài báo.
- `data/gat_autoencoder.pt`: trọng số mô hình đã huấn luyện.

```bash
python - <<'PY'
from src.ranking import build_similarity_index, get_recommendations

sim = build_similarity_index(force_recompute=True)
print('similarity shape:', tuple(sim.shape))
recs = get_recommendations(0, top_n=5)
print('recommendations:', len(recs))
print('first:', recs[0] if recs else 'none')
PY
```

### Bước 5: Chạy giao diện Streamlit

```bash
streamlit run src/app.py
```

## Logic quan trọng

- Preprocessing chỉ giữ cạnh trích dẫn mà bài được trích dẫn có tồn tại trong tập dữ liệu SQL hiện tại.
- Mô hình GAT xuất embedding đã normalize L2.
- Ranking ưu tiên `Year` mới hơn, sau đó mới đến điểm cosine giảm dần.
- App hiển thị đầy đủ: tiêu đề, tác giả, năm, số trích dẫn, điểm tương đồng và link DOI.

## Trạng thái chạy thử

Đã chạy thử đủ các giai đoạn trên dữ liệu SQL hiện tại:

- Đọc SQL thành công.
- Preprocessing thành công.
- Huấn luyện GAT thành công.
- Ranking thành công.
- Streamlit khởi động thành công.

Nếu bạn thay đổi schema SQL, hãy cập nhật lại câu SQL trong:

- `src/data_loader.py`
- `src/preprocessing.py`
- `src/app.py`
