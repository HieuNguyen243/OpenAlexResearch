import os

# Tự động xác định gốc dự án từ file config này
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Đảm bảo thư mục data luôn tồn tại
os.makedirs(DATA_DIR, exist_ok=True)

SQL_CONFIG = {
    'driver': '{SQL Server}',
    'server': r'LAPTOP-TEU72OKK\SQLEXPRESS',
    'database': 'OpenAlexDB',
    'trusted_connection': 'yes'
}

def get_conn_str():
    return f"Driver={SQL_CONFIG['driver']};Server={SQL_CONFIG['server']};Database={SQL_CONFIG['database']};Trusted_Connection={SQL_CONFIG['trusted_connection']};"