import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import GNNs.convert_data as cd
from GNNs.custom_GAT import myGATv2Conv
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv

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

        # Predict reception probability for each node
        reception_logits = self.mlp(x)
        return torch.sigmoid(reception_logits).squeeze(-1)
