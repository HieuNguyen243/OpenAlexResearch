import torch
import os
from src.config import DATA_DIR

# Đường dẫn đến file
features_path = os.path.join(DATA_DIR, "node_features.pt")

# Tải dữ liệu
node_features = torch.load(features_path)

print(f"Kiểu dữ liệu: {type(node_features)}")
print(f"Kích thước (Shape): {node_features.shape}") # Ví dụ: [Số_bài_báo, 384]
print(f"Kiểu phần tử (Dtype): {node_features.dtype}")

# Xem 5 hàng đầu tiên
print("5 hàng đầu tiên:")
print(node_features[:5])

# Xem vector của một bài báo cụ thể (ví dụ bài báo đầu tiên)
print("Vector của bài báo đầu tiên:")
print(node_features[0])