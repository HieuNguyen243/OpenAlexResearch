
# Cấu hình toàn cục cho dự án: đường dẫn dữ liệu, thông tin kết nối SQL Server
import os

# Xác định thư mục gốc của dự án dựa trên vị trí file config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Đường dẫn tới thư mục lưu trữ dữ liệu trung gian (node_features.pt, edge_index.pt, ...)
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Đảm bảo thư mục data luôn tồn tại (tạo nếu chưa có)
os.makedirs(DATA_DIR, exist_ok=True)

# Thông tin cấu hình kết nối SQL Server
SQL_CONFIG = {
    'driver': '{SQL Server}',  # Tên driver ODBC cho SQL Server
    'server': r'LAPTOP-TEU72OKK\\SQLEXPRESS',  # Tên server và instance
    'database': 'OpenAlexDB',  # Tên database
    'trusted_connection': 'yes'  # Sử dụng xác thực Windows
}

# Hàm sinh chuỗi kết nối ODBC từ cấu hình trên
def get_conn_str():
    return f"Driver={SQL_CONFIG['driver']};Server={SQL_CONFIG['server']};Database={SQL_CONFIG['database']};Trusted_Connection={SQL_CONFIG['trusted_connection']};"