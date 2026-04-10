import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import os
import sys

try:
    from src.config import DATA_DIR
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import DATA_DIR

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


class GATAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=128, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = GATModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )
        self.decoder = torch.nn.Linear(out_channels, in_channels)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        reconstructed = self.decoder(z)
        return reconstructed, z


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
    x = x.to(device=device, dtype=torch.float32)
    edge_index = edge_index.to(device)

    model = GATAutoEncoder(
        in_channels=x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            reconstructed, _ = model(x, edge_index)
            loss = F.mse_loss(reconstructed, x)
            loss.backward()
            optimizer.step()

            if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
                print(f"Epoch {epoch:03d}/{epochs} - Loss: {loss.item():.6f}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "OOM detected while training GAT. Reduce heads/hidden size or switch to CPU batching."
            ) from exc
        raise

    model.eval()
    with torch.no_grad():
        _, embeddings = model(x, edge_index)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    embeddings = embeddings.detach().cpu()

    output_path = os.path.join(DATA_DIR, "smart_embeddings.pt")
    model_path = os.path.join(DATA_DIR, "gat_autoencoder.pt")
    torch.save(embeddings, output_path)
    torch.save(model.state_dict(), model_path)
    print(f"Saved normalized embeddings: {output_path} shape={tuple(embeddings.shape)}")
    print(f"Saved model checkpoint: {model_path}")

if __name__ == "__main__":
    train_and_get_embeddings()