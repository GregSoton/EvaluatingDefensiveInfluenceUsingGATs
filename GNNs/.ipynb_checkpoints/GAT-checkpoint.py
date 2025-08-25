import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import GNNs.convert_data as cd
from GNNs.GNN import TeamShotGNN,ReceptionPredictionGNN,GATReceptionPredictor
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv

class GATReceptionPredictor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, edge_hidden_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads

        # Initial node feature transformation
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = torch.nn.Linear(num_edge_features, edge_hidden_channels)

        # GAT layers
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, 
                             edge_dim=edge_hidden_channels, add_self_loops=True)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, 
                             edge_dim=edge_hidden_channels, add_self_loops=True)

        # Final prediction layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Transform node features
        x = self.node_encoder(x)
        x_original = x  # Save for skip connection
        edge_features = self.edge_encoder(edge_attr)

        # First GAT layer
        x, attention_weights1 = self.gat1(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        x, attention_weights2 = self.gat2(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x_combined = torch.cat([x_original, x], dim=1)
        
        reception_logits = self.mlp(x_combined)

        return torch.sigmoid(reception_logits).squeeze(-1), (attention_weights1,attention_weights2)
    
def train_reception_prediction_AT_model(graphs, num_epochs=100, batch_size=32, hidden_channels=64, edge_hidden_channels=8, lr=0.001):
    dataset = cd.prepare_dataset_reception(graphs)
    #dataset,scaler = scale_selected_features(dataset)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATReceptionPredictor(
      num_node_features=dataset[0].x.size(1),
      num_edge_features=dataset[0].edge_attr.size(1),
      hidden_channels=hidden_channels,
      edge_hidden_channels=edge_hidden_channels
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            out,_ = model(data.x, data.edge_index, data.edge_attr, data.batch)

            # Only compute loss for attacking players
            masked_out = out[data.attacking_player_mask]
            masked_y = data.y[data.attacking_player_mask]

            loss = criterion(masked_out, masked_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_loss = 0
            predictions = []
            true_values = []

            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out,_ = model(data.x, data.edge_index, data.edge_attr, data.batch)

                    masked_out = out[data.attacking_player_mask]
                    masked_y = data.y[data.attacking_player_mask]

                    val_loss += criterion(masked_out, masked_y).item()
                    predictions.extend(masked_out.cpu().numpy())
                    true_values.extend(masked_y.cpu().numpy())

            auc = roc_auc_score(true_values, predictions)
            print(f'Epoch {epoch}: Train Loss: {total_loss/len(train_loader):.4f}, '
                f'Val Loss: {val_loss/len(test_loader):.4f}, AUC: {auc:.4f}')

    return model, train_loader, test_loader