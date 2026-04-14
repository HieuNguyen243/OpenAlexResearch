import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR
    
from src.model_gat import GATModel, citation_contrastive_loss

# 1. Baseline 2: Standard GCN
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 2. Reusable Training Function
def train_model(model, train_data, val_data, epochs=50, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    model = model.to(device)
    
    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[10, 10],
        batch_size=1024,
        edge_label_index=train_data.edge_label_index,
        neg_sampling_ratio=1.0,
        shuffle=True,
    )
    
    print(f"--- Training {model.__class__.__name__} ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index)
            # Use same contrastive loss
            loss = citation_contrastive_loss(z, batch.edge_label_index, margin=0.5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} - Loss: {avg_loss:.6f}")
            
    return model

# 3. Embedding Generation
def get_node_embeddings(model, data, device):
    node_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=2048,
        input_nodes=torch.arange(data.num_nodes),
        shuffle=False,
    )
    model.eval()
    
    # Figure out out_channels dynamically
    z_temp = model(data.x[:2].to(device), data.edge_index[:, :1].to(device))
    out_channels = z_temp.shape[1]
    
    all_embeddings = torch.zeros(data.num_nodes, out_channels)
    ptr = 0
    with torch.no_grad():
        for batch in node_loader:
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index)[:batch.batch_size]
            z = F.normalize(z, p=2, dim=1) # L2 Norm
            n = batch.batch_size
            all_embeddings[ptr:ptr+n] = z.cpu()
            ptr += n
            
    return all_embeddings.to(device)

# 4. Evaluation Metrics
def evaluate_link_prediction(z, edge_label_index, edge_label):
    src, dst = edge_label_index[0], edge_label_index[1]
    z_src = z[src]
    z_dst = z[dst]
    
    # Cosine Similarity
    scores = F.cosine_similarity(z_src, z_dst).cpu().numpy()
    labels = edge_label.cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_ranking_metrics(z, test_data, num_samples=500):
    device = z.device
    num_nodes = z.shape[0]
    
    # Positives only
    pos_mask = test_data.edge_label == 1
    pos_edge_index = test_data.edge_label_index[:, pos_mask]
    
    num_pos = pos_edge_index.shape[1]
    if num_pos > num_samples:
        indices = torch.randperm(num_pos)[:num_samples]
        pos_edge_index = pos_edge_index[:, indices]
    else:
        num_samples = num_pos
        
    mrr_sum = 0.0
    hit10_count = 0
    
    src_nodes = pos_edge_index[0]
    dst_nodes = pos_edge_index[1]
    
    print("Computing MRR and Hit@10...")
    for i in tqdm(range(num_samples), leave=False, desc="Ranking Nodes"):
        src = src_nodes[i]
        true_dst = dst_nodes[i]
        
        src_emb = z[src].unsqueeze(0)
        sims = F.cosine_similarity(src_emb, z)
        
        # Sort descending
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == true_dst).nonzero(as_tuple=True)[0].item() + 1
        
        mrr_sum += 1.0 / rank
        if rank <= 10:
            hit10_count += 1
            
    mrr = mrr_sum / num_samples
    hit10 = hit10_count / num_samples
    return mrr, hit10

# 5. Pipeline Runner
def run_evaluation_pipeline():
    features_path = os.path.join(DATA_DIR, "node_features.pt")
    edges_path = os.path.join(DATA_DIR, "edge_index.pt")
    
    print("Loading graph data...")
    if not os.path.exists(features_path) or not os.path.exists(edges_path):
        print(f"Error: Could not find {features_path} or {edges_path}. Run preprocessing first.")
        return
        
    x = torch.load(features_path).to(dtype=torch.float32)
    edge_index = torch.load(edges_path)
    
    graph_data = Data(x=x, edge_index=edge_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=False,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(graph_data)
    print(f"Train/Val/Test Split Done. Nodes: {x.shape[0]}, Train Edges: {train_data.edge_index.shape[1]}")
    
    # Store results
    results = []
    
    # ==========================================
    # Baseline 1: Raw Features (No-GNN)
    # ==========================================
    print("\n[Baseline 1] Evaluating Raw Features (No-GNN)...")
    z_raw = F.normalize(x, p=2, dim=1).to(device)
    raw_auc = evaluate_link_prediction(z_raw, test_data.edge_label_index, test_data.edge_label)
    raw_mrr, raw_hit10 = compute_ranking_metrics(z_raw, test_data)
    
    results.append({
        "Model": "Raw SBERT Features (No-GNN)",
        "ROC-AUC": raw_auc,
        "MRR": raw_mrr,
        "Hit@10": raw_hit10
    })
    
    # Model configs
    in_channels = graph_data.num_node_features
    hidden_channels = 128
    out_channels = 128
    epochs = 60  
    
    # ==========================================
    # Baseline 2: GCN Model
    # ==========================================
    print("\n[Baseline 2] Training & Evaluating GCN...")
    gcn = GCNModel(in_channels, hidden_channels, out_channels)
    gcn = train_model(gcn, train_data, val_data, epochs=epochs, device=device)
    z_gcn = get_node_embeddings(gcn, test_data, device)
    
    gcn_auc = evaluate_link_prediction(z_gcn, test_data.edge_label_index, test_data.edge_label)
    gcn_mrr, gcn_hit10 = compute_ranking_metrics(z_gcn, test_data)
    
    results.append({
        "Model": "GCN (Baseline)",
        "ROC-AUC": gcn_auc,
        "MRR": gcn_mrr,
        "Hit@10": gcn_hit10
    })
    
    # ==========================================
    # Proposed Model: GATv2
    # ==========================================
    print("\n[Proposed] Training & Evaluating GATv2...")
    gat = GATModel(in_channels, hidden_channels, out_channels, heads=4, dropout=0.2)
    gat = train_model(gat, train_data, val_data, epochs=epochs, device=device)
    z_gat = get_node_embeddings(gat, test_data, device)
    
    gat_auc = evaluate_link_prediction(z_gat, test_data.edge_label_index, test_data.edge_label)
    gat_mrr, gat_hit10 = compute_ranking_metrics(z_gat, test_data)
    
    results.append({
        "Model": "GATv2 (Proposed)",
        "ROC-AUC": gat_auc,
        "MRR": gat_mrr,
        "Hit@10": gat_hit10
    })
    
    # ==========================================
    # Display Results
    # ==========================================
    print("\n" + "="*65)
    print("📈  MODEL PERFORMANCE COMPARISON ON LINK PREDICTION  📈")
    print("="*65)
    df_results = pd.DataFrame(results)
    
    df_results["ROC-AUC"] = df_results["ROC-AUC"].apply(lambda x: f"{x:.4f}")
    df_results["MRR"] = df_results["MRR"].apply(lambda x: f"{x:.4f}")
    df_results["Hit@10"] = df_results["Hit@10"].apply(lambda x: f"{x:.4f}")
    
    # Format and print
    try:
        from tabulate import tabulate
        print(tabulate(df_results, headers='keys', tablefmt='outline', showindex=False))
    except ImportError:
        print(df_results.to_markdown(index=False))
    print("="*65)

if __name__ == "__main__":
    run_evaluation_pipeline()
