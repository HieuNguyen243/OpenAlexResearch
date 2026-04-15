import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.nn import Node2Vec
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR
    
from src.model_gat import GATModel, citation_contrastive_loss


def train_gat_model(model, train_data, val_data, epochs=50, lr=1e-3, device='cpu'):
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
    gat_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model(batch.x, batch.edge_index)
            loss = citation_contrastive_loss(z, batch.edge_label_index, margin=0.5)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        gat_losses.append(avg_loss)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{epochs} - Loss: {avg_loss:.6f}")

    return model, gat_losses


def train_node2vec(edge_index, num_nodes, device, embedding_dim=128):
    model = Node2Vec(
        edge_index, 
        embedding_dim=embedding_dim, 
        walk_length=20,
        context_size=10, 
        walks_per_node=10,
        num_negative_samples=1, 
        p=1, q=1, 
        sparse=True,
        num_nodes=num_nodes  # Đã thêm num_nodes để fix lỗi IndexError của Node2Vec
    ).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    
    print(f"--- Training Node2Vec ---")
    node2vec_losses = []
    for epoch in range(1, 11): 
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        node2vec_losses.append(avg_loss)
        if epoch == 1 or epoch % 2 == 0 or epoch == 10:
            print(f"Epoch {epoch:02d}/10 - Loss: {avg_loss:.4f}")
            
    model.eval()
    with torch.no_grad():
        z = model()
    
    return F.normalize(z, p=2, dim=1), node2vec_losses


def moving_average(values, window=5):
    if len(values) == 0:
        return []
    window = max(1, int(window))
    if len(values) < window:
        window = len(values)
    return np.convolve(values, np.ones(window) / window, mode='valid')

# 3. Embedding Generation for GAT
def get_node_embeddings(model, data, device):
    node_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=2048,
        input_nodes=torch.arange(data.num_nodes),
        shuffle=False,
    )
    model.eval()
    
    # [ĐÃ FIX] Sử dụng cạnh giả an toàn để test dimension, tránh lỗi Out-Of-Bounds
    dummy_x = data.x[:2].to(device)
    dummy_edge = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    z_temp = model(dummy_x, dummy_edge)
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
    
    # [ĐÃ FIX] Thêm .detach() để tránh lỗi RuntimeError với Numpy
    scores = F.cosine_similarity(z_src, z_dst).detach().cpu().numpy()
    labels = edge_label.detach().cpu().numpy()
    return roc_auc_score(labels, scores)

def compute_ranking_metrics(z, test_data, num_samples=500):
    device = z.device
    
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
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    
    print("Loading graph data...")
    if not os.path.exists(features_path) or not os.path.exists(edges_path):
        print(f"Error: Could not find {features_path} or {edges_path}. Run preprocessing first.")
        return
        
    # [ĐÃ FIX] Thêm weights_only=True để tắt cảnh báo FutureWarning
    x = torch.load(features_path, weights_only=True).to(dtype=torch.float32)
    edge_index = torch.load(edges_path, weights_only=True)
    
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
    # Baseline 1: TF-IDF (Pure Lexical, No-Graph, No-Semantics)
    # ==========================================
    print("\n[Baseline 1] Evaluating TF-IDF (Pure Lexical)...")
    if os.path.exists(metadata_path):
        # Read Original Titles
        df_meta = pd.read_csv(metadata_path)
        titles = df_meta["Title"].fillna("").astype(str).tolist()
        
        vectorizer = TfidfVectorizer(max_features=512)
        tfidf_matrix = vectorizer.fit_transform(titles).toarray()
        
        z_tfidf = torch.tensor(tfidf_matrix, dtype=torch.float32).to(device)
        z_tfidf = F.normalize(z_tfidf, p=2, dim=1)
        
        tfidf_auc = evaluate_link_prediction(z_tfidf, test_data.edge_label_index, test_data.edge_label)
        tfidf_mrr, tfidf_hit10 = compute_ranking_metrics(z_tfidf, test_data)
        
        results.append({
            "Model": "TF-IDF (Pure Lexical)",
            "ROC-AUC": tfidf_auc,
            "MRR": tfidf_mrr,
            "Hit@10": tfidf_hit10
        })
    else:
        print(f"Warning: {metadata_path} not found. Skipping TF-IDF Baseline.")

    # ==========================================
    # Baseline 2: Node2Vec (Pure Graph, No-Text)
    # ==========================================
    print("\n[Baseline 2] Training & Evaluating Node2Vec (Pure Graph)...")
    # Make edges undirected for Node2Vec to discover structural communities better
    undirected_train_edges = torch.cat([train_data.edge_index, train_data.edge_index[[1, 0]]], dim=1)
    
    z_n2v, node2vec_losses = train_node2vec(undirected_train_edges, graph_data.num_nodes, device, embedding_dim=128)
    
    n2v_auc = evaluate_link_prediction(z_n2v, test_data.edge_label_index, test_data.edge_label)
    n2v_mrr, n2v_hit10 = compute_ranking_metrics(z_n2v, test_data)
    
    results.append({
        "Model": "Node2Vec (Pure Graph Structure)",
        "ROC-AUC": n2v_auc,
        "MRR": n2v_mrr,
        "Hit@10": n2v_hit10
    })
    
    # Model configs
    in_channels = graph_data.num_node_features
    hidden_channels = 128
    out_channels = 128
    epochs = 60  
    
    # ==========================================
    # Proposed System: GATv2 + SBERT
    # ==========================================
    print("\n[Proposed] Training & Evaluating GATv2 (Hybrid GNN + Semantics)...")
    gat = GATModel(in_channels, hidden_channels, out_channels, heads=4, dropout=0.2)
    gat, gat_losses = train_gat_model(gat, train_data, val_data, epochs=epochs, device=device)
    z_gat = get_node_embeddings(gat, test_data, device)
    
    gat_auc = evaluate_link_prediction(z_gat, test_data.edge_label_index, test_data.edge_label)
    gat_mrr, gat_hit10 = compute_ranking_metrics(z_gat, test_data)
    
    results.append({
        "Model": "GATv2 + SBERT (Proposed Hybrid)",
        "ROC-AUC": gat_auc,
        "MRR": gat_mrr,
        "Hit@10": gat_hit10
    })
    
    # ==========================================
    # Display Results
    # ==========================================
    print("\n" + "="*70)
    print("📈  MODEL PERFORMANCE COMPARISON (HYBRID SYSTEM EVALUATION)  📈")
    print("="*70)
    # -------------------------
    # Plot Loss Comparison
    # -------------------------
    losses = {}
    try:
        losses["Node2Vec"] = node2vec_losses
    except NameError:
        losses["Node2Vec"] = []
    try:
        losses["GATv2"] = gat_losses
    except NameError:
        losses["GATv2"] = []

    plt.figure(figsize=(8, 6))
    for name, vals in losses.items():
        if not vals:
            continue
        epochs_x = list(range(1, len(vals) + 1))
        plt.plot(epochs_x, vals, linestyle='--', alpha=0.4, label=f"{name} raw")
        if len(vals) >= 3:
            w = min(5, len(vals))
            sm = moving_average(vals, window=w)
            plt.plot(list(range(w, len(vals) + 1)), sm, linewidth=2, label=f"{name} smoothed")
        else:
            plt.plot(epochs_x, vals, linewidth=2, label=f"{name}")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch Comparison")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(DATA_DIR, "loss_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved loss comparison plot: {out_path}")

    # Print final loss values
    print("Final losses:")
    for name, vals in losses.items():
        if vals:
            print(f" - {name}: {vals[-1]:.6f}")
    df_results = pd.DataFrame(results)
    
    df_results["ROC-AUC"] = df_results["ROC-AUC"].apply(lambda x: f"{x:.4f}")
    df_results["MRR"] = df_results["MRR"].apply(lambda x: f"{x:.4f}")
    df_results["Hit@10"] = df_results["Hit@10"].apply(lambda x: f"{x:.4f}")
    
    try:
        from tabulate import tabulate
        print(tabulate(df_results, headers='keys', tablefmt='outline', showindex=False))
    except ImportError:
        print(df_results.to_markdown(index=False))
    print("="*70)

if __name__ == "__main__":
    run_evaluation_pipeline()