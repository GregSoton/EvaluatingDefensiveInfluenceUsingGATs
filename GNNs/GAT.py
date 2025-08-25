import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import GNNs.convert_data as cd
from GNNs.GNN import ReceptionPredictionGNN
from GNNs.custom_GAT import myGATv2Conv
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv

class GATReceptionPredictor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_channels, edge_hidden_channels, num_heads=16):
        super(GATReceptionPredictor, self).__init__()
        self.num_heads = num_heads
        self.test_mode = False
        self.mask_node = None
        self.target_node = None
        self.hidden_channels = hidden_channels

        # Initial node feature transformation
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = torch.nn.Linear(num_edge_features, edge_hidden_channels)

        # GAT layers
        self.gat1 = myGATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                             edge_dim=edge_hidden_channels, add_self_loops=True)
        self.gat2 = myGATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                             edge_dim=edge_hidden_channels, add_self_loops=True)

        # Final prediction layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels*2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def renormalize_attention(self, edge_index, attention_weights, target_node):
        # Find all edges pointing to the target node
        target_mask = (edge_index[1] == target_node)

        # For each head, renormalize the attention weights for edges pointing to target_node
        for head in range(self.num_heads):
            head_attn = attention_weights[:, head]
            target_attn = head_attn[target_mask]

            # Renormalize only if sum is not zero
            if target_attn.sum() > 0:
                head_attn[target_mask] = target_attn / target_attn.sum()
                attention_weights[:, head] = head_attn

        return attention_weights

    def forward(self, x, edge_index, edge_attr, batch, test_mode=False, mask_node_name=None, target_node_name=None, graph=None):
        # Transform node features
        x = self.node_encoder(x)
        x_original = x
        edge_features = self.edge_encoder(edge_attr)

        if test_mode == False:
            x, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
        else:
            graph_names = [s[0] for s in graph.nodes(data=True)]
            mask_node = graph_names.index(mask_node_name)
            target_node = graph_names.index(target_node_name)
            x_temp, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
            edge_index_1, attn_weights_1 = attention_weights1

            # Find edges where target_node is the destination and mask_node is the source
            target_mask = (edge_index_1[1] == target_node)
            source_mask = (edge_index_1[0] == mask_node)
            modify_mask = target_mask & source_mask

            # Modify attention weights
            attn_weights_1[modify_mask] = 0
            x, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, return_attention_weights=True, test=True, alpha_updated=attn_weights_1)
            edge_index_1, attn_weights_1 = attention_weights1

        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GAT layer
        if test_mode == False:
            x, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
        else:
            graph_names = [s[0] for s in graph.nodes(data=True)]
            mask_node = graph_names.index(mask_node_name)
            target_node = graph_names.index(target_node_name)
            x_temp, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, return_attention_weights=True)
            edge_index_2, attn_weights_2 = attention_weights2

            # Find edges where target_node is the destination and mask_node is the source
            target_mask = (edge_index_1[1] == target_node)
            source_mask = (edge_index_1[0] == mask_node)
            modify_mask = target_mask & source_mask

            # Modify attention weights
            attn_weights_2[modify_mask] = 0
            x, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, return_attention_weights=True, test=True, alpha_updated=attn_weights_2)
            
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x_combined = torch.cat([x_original, x], dim=1)
        reception_logits = self.mlp(x_combined)

        return torch.sigmoid(reception_logits).squeeze(-1), (attention_weights1, attention_weights2)