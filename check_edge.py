import torch
import os
from src.config import DATA_DIR

edge_path = os.path.join(DATA_DIR, "edge_index.pt")
edge_index = torch.load(edge_path)

print(f"Kích thước ma trận cạnh: {edge_index.shape}") 
print(f"Tổng số liên kết trích dẫn: {edge_index.shape[1]}")
print(f"Kiểu dữ liệu: {edge_index.dtype}") # Thường là torch.int64 (long)


# Xem 5 liên kết đầu tiên
# Kết quả dạng: tensor([[src1, src2...], [tgt1, tgt2...]])
print("5 liên kết đầu tiên:")
print(edge_index[:, :5]) 

# Chuyển sang dạng cặp (Source -> Target) cho dễ nhìn
for i in range(min(5, edge_index.shape[1])):
    source = edge_index[0, i].item()
    target = edge_index[1, i].item()
    print(f"Bài báo tại dòng {source} trích dẫn bài báo tại dòng {target}")
    
    
    
meta_path = os.path.join(DATA_DIR, "metadata.csv")
import pandas as pd
df = pd.read_csv(meta_path)
num_papers = len(df)

max_idx = edge_index.max().item() if edge_index.numel() > 0 else 0
min_idx = edge_index.min().item() if edge_index.numel() > 0 else 0

print(f"Chỉ số lớn nhất trong cạnh: {max_idx}")
print(f"Tổng số bài báo trong metadata: {num_papers}")

if max_idx >= num_papers:
    print("❌ LỖI: Có chỉ số cạnh vượt quá số lượng bài báo!")
if min_idx < 0:
    print("❌ LỖI: Chỉ số cạnh không được là số âm!")
    
self_loops = (edge_index[0] == edge_index[1]).sum().item()
print(f"Số lượng bài báo tự trích dẫn chính mình: {self_loops}")