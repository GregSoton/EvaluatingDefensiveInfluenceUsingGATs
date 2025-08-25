import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv
import numpy as np

class TeamShotGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super(TeamShotGNN, self).__init__()

        # Graph Convolution layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # MLP for final prediction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),  # *2 because we'll concatenate mean and max pooling
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # If batch is None (single graph), create appropriate batch tensor
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Global pooling - combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Final prediction
        x = self.mlp(x)
        return torch.sigmoid(x)
    
class ReceptionPredictionGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels):
        super().__init__()

        # Add linear transformations to match dimensions
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = torch.nn.Linear(num_edge_features, hidden_channels)

        # Modified convolution layers
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)  # Using SAGEConv instead of GENConv
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        self.mlp = torch.nn.Sequential(
          torch.nn.Linear(hidden_channels, hidden_channels),
          torch.nn.ReLU(),
          torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Transform features to same dimension
        x = self.node_encoder(x)

        # Message passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        #x = self.conv3(x, edge_index)

        # Predict reception probability for each node
        reception_logits = self.mlp(x)
        return torch.sigmoid(reception_logits).squeeze(-1)
    
class GATReceptionPredictor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        # Initial node feature transformation
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)

        # GAT layers
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, add_self_loops=True)
        self.gat2 = GATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads, add_self_loops=True)

        # Final prediction layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # Transform node features
        x = self.node_encoder(x)

        # First GAT layer
        #x, attention_weights1 = self.gat1(x, edge_index, return_attention_weights=True)
        #x = F.elu(x)
        #x = F.dropout(x, p=0.1, training=self.training)

        # Second GAT layer
        #x, attention_weights2 = self.gat2(x, edge_index, return_attention_weights=True)
        #x = F.elu(x)

        # Final prediction
        reception_logits = self.mlp(x)
        return torch.sigmoid(reception_logits).squeeze(-1), (attention_weights1, attention_weights2)