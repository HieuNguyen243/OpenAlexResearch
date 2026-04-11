import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# 1. KIẾN TRÚC GATv2 
# ==========================================
class OptimizedGATv2(nn.Module):
    def __init__(self, in_channels=384, hidden_channels=64, out_channels=128, heads=4, dropout=0.2):
        super(OptimizedGATv2, self).__init__()
        self.dropout = dropout
        
        # Layer 1: Chú ý đa đầu
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout, add_self_loops=True)
        
        # Layer 2: Gom lại thành vector đầu ra
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout, add_self_loops=True)
        
        # Lớp Linear giúp biến đổi SBERT (384) -> Kích thước Output (128) để làm Skip Connection
        self.skip_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 1. Lưu lại đặc trưng gốc của SBERT đã được ép về 128 chiều
        x_residual = self.skip_proj(x)
        
        # 2. Đi qua mạng đồ thị
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = F.elu(self.conv1(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        
        # 3.Giữ lại bản sắc ngữ nghĩa văn bản, kết hợp cấu trúc trích dẫn
        out = h + x_residual
        return out

# ==========================================
# 2. HÀM TÍNH LOSS CHO LINK PREDICTION
# ==========================================
def link_prediction_loss(edge_embeddings_src, edge_embeddings_dst, edge_labels):
    """
    Tính Loss so sánh giữa Cạnh Thật (label=1) và Cạnh Ảo (label=0)
    """
    # Tích vô hướng giữa 2 node của 1 cạnh
    out = (edge_embeddings_src * edge_embeddings_dst).sum(dim=-1)
    # Dùng BCEWithLogitsLoss cho ổn định
    return F.binary_cross_entropy_with_logits(out, edge_labels)

# ==========================================
# 3. PIPELINE TRAIN BẰNG MINI-BATCH
# ==========================================
def train_model(data, epochs=10, device='cuda'):
    model = OptimizedGATv2(in_channels=384, hidden_channels=64, out_channels=128, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # DataLoader: Chia nhỏ đồ thị
    # num_neighbors=[10, 10]: Tối đa lấy 10 hàng xóm ở Layer 1, 10 ở Layer 2 để tránh tràn RAM
    # neg_sampling_ratio=1.0: Tự động sinh ra 1 cạnh ảo cho mỗi 1 cạnh thật
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=2048, # Thay đổi VRAM của gpu
        edge_label_index=data.edge_index,
        neg_sampling_ratio=1.0,
        shuffle=True,
    )
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Đẩy batch qua GNN để sinh vector
            z = model(batch.x, batch.edge_index)
            
            # Lấy vector của cặp node theo cạnh (cả cạnh thật và cạnh ảo được sinh ra trong batch.edge_label_index)
            src_idx = batch.edge_label_index[0]
            dst_idx = batch.edge_label_index[1]
            
            z_src = z[src_idx]
            z_dst = z[dst_idx]
            
            # Tính Loss
            loss = link_prediction_loss(z_src, z_dst, batch.edge_label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
            
        print(f"-> Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
    return model

# ==========================================
# 4. XUẤT FILE
# ==========================================
@torch.no_grad()
def extract_and_save_embeddings(model, data, save_path="final_embeddings.npy", device='cuda'):
    print("\nĐang trích xuất Vector Embedding toàn đồ thị...")
    model.eval()
    
    # Tạo NodeLoader
    from torch_geometric.loader import NeighborLoader
    node_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=4096,
        input_nodes=None,
        shuffle=False
    )
    
    all_embeddings = []
    
    for batch in tqdm(node_loader, desc="Inference"):
        batch = batch.to(device)
        # Chỉ lấy embedding của các "tâm" (batch_size), không lấy embedding của hàng xóm phụ trợ
        z = model(batch.x, batch.edge_index)[:batch.batch_size] 
        all_embeddings.append(z.cpu().numpy())
        
    # Ghép tất cả mini-batch lại thành 1 ma trận duy nhất
    final_matrix = np.concatenate(all_embeddings, axis=0)
    
    # Lưu ra file cho Hiếu
    np.save(save_path, final_matrix)
    print(f" Đã xuất thành công ma trận {final_matrix.shape} ra file: {save_path}")

# ==========================================
# KHU VỰC CHẠY THỬ (MAIN)
# ==========================================
if __name__ == "__main__":
    # --- MÔ PHỎNG DỮ LIỆU ---
    print("Đang load dữ liệu.")
    NUM_NODES = 100000 
    SBERT_DIM = 384
    
    dummy_x = torch.randn((NUM_NODES, SBERT_DIM), dtype=torch.float)
    # Tạo ngẫu nhiên 500k lượt trích dẫn
    dummy_edge_index = torch.randint(0, NUM_NODES, (2, 500000), dtype=torch.long) 
    
    graph_data = Data(x=dummy_x, edge_index=dummy_edge_index)
    # ------------------------------------------------------------------
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    
    # 1. Huấn luyện mô hình
    trained_model = train_model(graph_data, epochs=3, device=DEVICE)
    
    # 2. Xuất file cho Hiếu
    extract_and_save_embeddings(trained_model, graph_data, save_path="citation_embeddings_128d.npy", device=DEVICE)