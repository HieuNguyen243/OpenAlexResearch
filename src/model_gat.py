import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score
import os
import sys

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR

def citation_contrastive_loss(z, edge_index, margin=0.5):
    if edge_index.shape[1] == 0:
        return torch.tensor(0.0, requires_grad=True, device=z.device)
    src, dst = edge_index[0], edge_index[1]
    pos_sim = F.cosine_similarity(z[src], z[dst])
    neg_dst = torch.randint(0, z.shape[0], (src.shape[0],), device=z.device)
    neg_sim = F.cosine_similarity(z[src], z[neg_dst])
    loss = torch.clamp(margin - pos_sim + neg_sim, min=0).mean()
    return loss


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x




def train_and_get_embeddings(
    epochs=120,
    lr=1e-3,
    hidden_channels=128,
    out_channels=128,
    heads=4,
    dropout=0.2,
):
    features_path = os.path.join(DATA_DIR, "node_features.pt")
    edges_path = os.path.join(DATA_DIR, "edge_index.pt")

    x = torch.load(features_path)
    edge_index = torch.load(edges_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(dtype=torch.float32)
    graph_data = Data(x=x, edge_index=edge_index)

    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.0,
        is_undirected=False,
        add_negative_train_samples=False,
    )
    train_data, val_data, _ = transform(graph_data)

    model = GATModel(
        in_channels=graph_data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[10, 10],
        batch_size=1024,
        edge_label_index=train_data.edge_label_index,
        neg_sampling_ratio=1.0,
        shuffle=True,
    )

    def validate(model, val_data, device):
        model.eval()
        with torch.no_grad():
            val_data = val_data.to(device)
            z = model(val_data.x, val_data.edge_index)
            src = val_data.edge_label_index[0]
            dst = val_data.edge_label_index[1]
            scores = (z[src] * z[dst]).sum(dim=-1).sigmoid().cpu().numpy()
            labels = val_data.edge_label.cpu().numpy()
        return roc_auc_score(labels, scores)

    try:
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
            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                auc = validate(model, val_data, device)
                print(f"Epoch {epoch:03d}/{epochs} - Loss: {avg_loss:.6f} - Val AUC: {auc:.4f}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "OOM detected while training GAT. Reduce heads/hidden size or switch to CPU batching."
            ) from exc
        raise

    node_loader = NeighborLoader(
        graph_data,
        num_neighbors=[10, 10],
        batch_size=2048,
        input_nodes=torch.arange(graph_data.num_nodes),
        shuffle=False,
    )
    model.eval()
    all_embeddings = torch.zeros(graph_data.num_nodes, out_channels)
    ptr = 0
    with torch.no_grad():
        for batch in node_loader:
            batch = batch.to(device)
            z = model(batch.x, batch.edge_index)[:batch.batch_size]
            z = F.normalize(z, p=2, dim=1)
            n = batch.batch_size
            all_embeddings[ptr:ptr+n] = z.cpu()
            ptr += n
    assert ptr == graph_data.num_nodes
    embeddings = all_embeddings

    output_path = os.path.join(DATA_DIR, "smart_embeddings.pt")
    model_path = os.path.join(DATA_DIR, "gat_autoencoder.pt")
    torch.save(embeddings, output_path)
    torch.save(model.state_dict(), model_path)
    print(f"Saved normalized embeddings: {output_path} shape={tuple(embeddings.shape)}")
    print(f"Saved model checkpoint: {model_path}")
    try:
        print(f"Final training loss (last epoch): {gat_losses[-1]:.6f}")
    except Exception:
        pass
    return embeddings, gat_losses

if __name__ == "__main__":
    train_and_get_embeddings()