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

def scale_selected_features(data_list, scaler=None):
    cols_to_scale = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]
  
    if scaler is None:
        scaler = MinMaxScaler()
        # Fit on all training data
        all_x = np.concatenate([data.x[:, cols_to_scale].numpy() for data in data_list])
        scaler.fit(all_x)
  
    # Transform each data object
    for data in data_list:
        x_numpy = data.x.numpy()
        x_numpy[:, cols_to_scale] = scaler.transform(x_numpy[:, cols_to_scale])
        data.x = torch.FloatTensor(x_numpy)
  
    return data_list, scaler

def train_model(model, train_loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.binary_cross_entropy(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate_model(model, loader, device):
    """Evaluate the model."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            predictions.extend(out.cpu().numpy())
            targets.extend(data.y.cpu().numpy())

    return roc_auc_score(targets, predictions)

# Main training function
def train_shot_prediction_model(graphs, num_epochs=50, batch_size=4, hidden_channels=64, lr=0.001):
    """Train the shot prediction model."""
    # Prepare dataset
    dataset = cd.prepare_dataset(graphs)

    # Split dataset
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TeamShotGNN(
      num_node_features=dataset[0].x.size(1),
      num_edge_features=dataset[0].edge_attr.size(1),
      hidden_channels=hidden_channels
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_val_auc = 0
    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, train_loader, optimizer, device)

        # Evaluate
        val_auc = evaluate_model(model, val_loader, device)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_shot_prediction_model.pt')

        if epoch % 5 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}')

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_shot_prediction_model.pt'))
    test_auc = evaluate_model(model, test_loader, device)
    print(f'Final Test AUC: {test_auc:.4f}')

    return model

def train_reception_prediction_model(graphs, num_epochs=100, batch_size=32, hidden_channels=64, lr=0.001):
    dataset = cd.prepare_dataset_reception(graphs)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReceptionPredictionGNN(
      num_node_features=dataset[0].x.size(1),
      num_edge_features=dataset[0].edge_attr.size(1),
      hidden_channels=hidden_channels
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
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

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
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)

                    masked_out = out[data.attacking_player_mask]
                    masked_y = data.y[data.attacking_player_mask]

                    val_loss += criterion(masked_out, masked_y).item()
                    predictions.extend(masked_out.cpu().numpy())
                    true_values.extend(masked_y.cpu().numpy())

            auc = roc_auc_score(true_values, predictions)
            print(f'Epoch {epoch}: Train Loss: {total_loss/len(train_loader):.4f}, '
                f'Val Loss: {val_loss/len(test_loader):.4f}, AUC: {auc:.4f}')

    return model, train_loader, test_loader

def train_reception_prediction_AT_model(graphs, num_epochs=100, batch_size=32, hidden_channels=64, lr=0.001):
    dataset = cd.prepare_dataset_reception(graphs)
    #dataset,scaler = scale_selected_features(dataset)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATReceptionPredictor(
      num_node_features=dataset[0].x.size(1),
      num_edge_features=dataset[0].edge_attr.size(1),
      hidden_channels=hidden_channels
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
        if epoch % 2 == 0:
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