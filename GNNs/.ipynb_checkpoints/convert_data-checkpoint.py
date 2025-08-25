import torch
from torch_geometric.data import Data, DataLoader
import numpy as np

def convert_graph_to_pytorch_geometric(G):
    """Convert networkx graph to PyTorch Geometric Data object."""
    # Get number of node features from first node
    num_node_features = len(G.nodes['ball']['features'])

    # Create node feature matrix
    x = []
    y = []  # Team shot target
    node_mapping = {}  # Map node names to indices

    for idx, (node, data) in enumerate(G.nodes(data=True)):
        node_mapping[node] = idx
        x.append(data['features'])
        if node == 'ball':
            y.append(data['team_shot_target'])

    x = torch.tensor(np.array(x), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.float)

    # Create edge index and edge features
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]])  # Add reverse edge

        # Edge features
        edge_features = list(data['features'])#[data['dx'], data['dy'], data['same_team']]
        edge_attr.extend([edge_features, edge_features])  # Add for both directions

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

def convert_graph_to_pytorch_geometric_reception(G):
    """Convert a directed NetworkX graph to a PyTorch Geometric Data object."""
    # Get number of node features from the first node
    num_node_features = len(G.nodes['ball']['features'])

    # Create node feature matrix
    x = []
    y = []  # Team shot target
    node_mapping = {}  # Map node names to indices
    attacking_player_mask = []  # To mask out defending players and ball

    for idx, (node, data) in enumerate(G.nodes(data=True)):
        node_mapping[node] = idx
        x.append(data['features'])
        if node != 'ball':
            y.append(data.get('reception_target', 0))
            is_attacking = (data['features'][7] == 1)  # Feature 7 indicates attacking team
            attacking_player_mask.append(is_attacking)
        else:
            y.append(0)
            attacking_player_mask.append(False)

    x = torch.tensor(np.array(x), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.float)
    attacking_player_mask = torch.tensor(attacking_player_mask, dtype=torch.bool)

    # Create edge index and edge features
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        # Add directed edge (u -> v)
        edge_index.append([node_mapping[u], node_mapping[v]])

        # Edge features
        edge_features = list(data['features'])  # Example: [dx, dy, same_team]
        edge_attr.append(edge_features)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Transpose to shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, attacking_player_mask=attacking_player_mask)

def prepare_dataset_reception(graphs):
    """Convert list of networkx graphs to PyTorch Geometric dataset."""
    data_list = []
    for G in graphs:
        data = convert_graph_to_pytorch_geometric_reception(G)
        data_list.append(data)
    return data_list

def prepare_dataset(graphs):
    """Convert list of networkx graphs to PyTorch Geometric dataset."""
    data_list = []
    for G in graphs:
        data = convert_graph_to_pytorch_geometric(G)
        data_list.append(data)
    return data_list