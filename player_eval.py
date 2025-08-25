"""
Defender Analysis for Football Reception Prediction

This module provides functionality to analyze defender influence on reception prediction
using Graph Attention Networks (GAT). It evaluates how removing different defenders
affects the model's predictions for attacking players.

Key Features:
- Defender influence calculation based on attention weights
- Defender removal experiments (top/bottom/random defenders)
- Performance metrics and threat analysis
- Match-specific analysis with data export
"""

import os
import pickle
import pandas as pd
import numpy as np
import random
import sys
from typing import List, Dict, Tuple, Any

# PyTorch and PyTorch Geometric imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv
from torch_geometric.data import Data, DataLoader

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# Custom imports
import GNNs.convert_data as cd
from custom_GAT import myGATv2Conv
from GNNs.GNN import ReceptionPredictionGNN
from GNNs.GAT import GATReceptionPredictor


# ============================================================================
# DATA CONVERSION FUNCTIONS
# ============================================================================

def convert_graph_to_pytorch_geometric_reception(G):
    """
    Convert a directed NetworkX graph to a PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph with node and edge features
        
    Returns:
        Data: PyTorch Geometric Data object with node features, edge features, and labels
    """
    # Get number of node features from the first node
    num_node_features = len(G.nodes['ball']['features'])

    # Initialize lists for node data
    x = []  # Node features
    y = []  # Reception targets
    node_mapping = {}  # Map node names to indices
    attacking_player_mask = []  # Mask for attacking players

    # Process each node
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

    # Convert to tensors
    x = torch.tensor(np.array(x), dtype=torch.float)
    y = torch.tensor(np.array(y), dtype=torch.float)
    attacking_player_mask = torch.tensor(attacking_player_mask, dtype=torch.bool)

    # Process edges
    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        # Add directed edge (u -> v)
        edge_index.append([node_mapping[u], node_mapping[v]])
        
        # Extract edge features
        edge_features = list(data['features'])
        edge_attr.append(edge_features)

    # Convert edge data to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()  # Shape: [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                attacking_player_mask=attacking_player_mask)


def prepare_dataset_reception(graphs):
    """
    Convert list of NetworkX graphs to PyTorch Geometric dataset.
    
    Args:
        graphs: List of NetworkX graphs
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    data_list = []
    for G in graphs:
        data = convert_graph_to_pytorch_geometric_reception(G)
        data_list.append(data)
    return data_list


# ============================================================================
# GAT MODEL DEFINITION
# ============================================================================

class GATReceptionPredictor(torch.nn.Module):
    """
    Graph Attention Network for reception prediction in football.
    
    This model uses two GAT layers with attention mechanisms to predict
    which player is most likely to receive a pass.
    """
    
    def __init__(self, num_node_features: int, num_edge_features: int, 
                 hidden_channels: int, edge_hidden_channels: int, num_heads: int = 16):
        """
        Initialize the GAT model.
        
        Args:
            num_node_features: Number of input node features
            num_edge_features: Number of input edge features
            hidden_channels: Hidden layer dimensions
            edge_hidden_channels: Edge feature dimensions
            num_heads: Number of attention heads
        """
        super(GATReceptionPredictor, self).__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Feature encoders
        self.node_encoder = torch.nn.Linear(num_node_features, hidden_channels)
        self.edge_encoder = torch.nn.Linear(num_edge_features, edge_hidden_channels)

        # GAT layers
        self.gat1 = myGATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                               edge_dim=edge_hidden_channels, add_self_loops=True)
        self.gat2 = myGATv2Conv(hidden_channels, hidden_channels // num_heads, heads=num_heads,
                               edge_dim=edge_hidden_channels, add_self_loops=True)

        # Final prediction layer
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1)
        )

    def renormalize_attention(self, edge_index, attention_weights, target_node):
        """
        Renormalize attention weights for a specific target node.
        
        Args:
            edge_index: Edge connectivity
            attention_weights: Current attention weights
            target_node: Node to renormalize attention for
            
        Returns:
            Renormalized attention weights
        """
        # Find all edges pointing to the target node
        target_mask = (edge_index[1] == target_node)

        # Renormalize attention weights for each head
        for head in range(self.num_heads):
            head_attn = attention_weights[:, head]
            target_attn = head_attn[target_mask]

            # Renormalize only if sum is not zero
            if target_attn.sum() > 0:
                head_attn[target_mask] = target_attn / target_attn.sum()
                attention_weights[:, head] = head_attn

        return attention_weights

    def forward(self, x, edge_index, edge_attr, batch, test_mode=False, 
                mask_node_name=None, target_node_name=None, graph=None):
        """
        Forward pass of the GAT model.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_attr: Edge features
            batch: Batch information
            test_mode: Whether to run in test mode with node masking
            mask_node_name: Name of node to mask (test mode only)
            target_node_name: Name of target node (test mode only)
            graph: Original NetworkX graph (test mode only)
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Encode features
        x = self.node_encoder(x)
        x_original = x
        edge_features = self.edge_encoder(edge_attr)

        # First GAT layer
        if not test_mode:
            x, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, 
                                                      return_attention_weights=True)
        else:
            # Test mode: mask specific nodes
            graph_names = [s[0] for s in graph.nodes(data=True)]
            mask_node = graph_names.index(mask_node_name)
            target_node = graph_names.index(target_node_name)
            
            x_temp, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, 
                                                           return_attention_weights=True)
            edge_index_1, attn_weights_1 = attention_weights1

            # Modify attention weights by masking specific edges
            target_mask = (edge_index_1[1] == target_node)
            source_mask = (edge_index_1[0] == mask_node)
            modify_mask = target_mask & source_mask
            attn_weights_1[modify_mask] = 0
            
            x, attention_weights1, pre_sm1 = self.gat1(x, edge_index, edge_attr=edge_features, 
                                                      return_attention_weights=True, test=True, 
                                                      alpha_updated=attn_weights_1)

        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GAT layer
        if not test_mode:
            x, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, 
                                                      return_attention_weights=True)
        else:
            # Test mode: mask specific nodes
            x_temp, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, 
                                                           return_attention_weights=True)
            edge_index_2, attn_weights_2 = attention_weights2

            # Modify attention weights by masking specific edges
            target_mask = (edge_index_1[1] == target_node)
            source_mask = (edge_index_1[0] == mask_node)
            modify_mask = target_mask & source_mask
            attn_weights_2[modify_mask] = 0
            
            x, attention_weights2, pre_sm2 = self.gat2(x, edge_index, edge_attr=edge_features, 
                                                      return_attention_weights=True, test=True, 
                                                      alpha_updated=attn_weights_2)
            
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Final prediction
        x_combined = torch.cat([x_original, x], dim=1)
        reception_logits = self.mlp(x_combined)

        return torch.sigmoid(reception_logits).squeeze(-1), (attention_weights1, attention_weights2)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_player_team(game_id: int, player_name: str, team_rosters: pd.DataFrame) -> pd.Series:
    """
    Get the team name for a specific player in a given game.
    
    Args:
        game_id: Game identifier
        player_name: Player's nickname
        team_rosters: DataFrame containing team roster information
        
    Returns:
        Team name for the player
    """
    return team_rosters.loc[(team_rosters['game_id'] == game_id) & 
                           (team_rosters['player_nickname'] == player_name), 'team_name']


def get_pitch_value(data: np.ndarray, x: float, y: float, 
                   pitch_length: float = 105, pitch_width: float = 68) -> float:
    """
    Get the Expected Threat (xT) value for a specific pitch location.
    
    Args:
        data: xT grid data
        x: X coordinate on pitch
        y: Y coordinate on pitch
        pitch_length: Length of the pitch
        pitch_width: Width of the pitch
        
    Returns:
        xT value at the specified location
    """
    # Clamp coordinates to pitch boundaries
    if not (0 <= x <= pitch_length and 0 <= y <= pitch_width):
        x = max(1, min(x, 104))
        y = max(1, min(y, 67))

    # Calculate grid indices
    rows, cols = data.shape[0], data.shape[1]
    cell_length = pitch_length / cols
    cell_width = pitch_width / rows
    
    col_index = min(int(x / cell_length), cols - 1)
    row_index = min(int(y / cell_width), rows - 1)
    
    return data[col_index][row_index]


# ============================================================================
# DEFENDER ANALYSIS FUNCTIONS
# ============================================================================

def calculate_defender_influence(scaled_graph, model, defender: str, attacker_names: List[str], 
                               all_names: List[str], loaded_scaler) -> Tuple[List[float], ...]:
    """
    Calculate the influence of a specific defender on all attacking players.
    
    This function measures how much a defender's presence affects the reception
    probabilities of attacking players by comparing predictions with and without
    the defender.
    
    Args:
        scaled_graph: NetworkX graph representing the game state
        model: Trained GAT model
        defender: Name of the defender to analyze
        attacker_names: List of attacking player names
        all_names: List of all player names
        loaded_scaler: Scaler for coordinate transformation
        
    Returns:
        Tuple containing distances, attentions, influences, relative influences, 
        threats, and total performance
    """
    # Convert graph to PyTorch Geometric format
    data = cd.convert_graph_to_pytorch_geometric_reception(scaled_graph)
    model.eval()
    
    # Initialize result lists
    distances = []
    attentions = []
    influences = []
    rel_influences = []
    threats = []
    total_performance = 0
    
    # Get graph node names and defender position
    graph_names = [s[0] for s in scaled_graph.nodes(data=True)]
    defender_data = scaled_graph.nodes[defender]
    defender_x, defender_y = loaded_scaler.inverse_transform(
        [[defender_data['features'][0], defender_data['features'][1]]])[0]
    
    # Get original predictions and attention weights
    with torch.no_grad():
        probs, attention_weights = model(data.x, data.edge_index, data.edge_attr, None)
    
    attention_weights_1, attention_weights_2 = attention_weights
    edge_index_1, attn_weights_1 = attention_weights_1
    edge_index_2, attn_weights_2 = attention_weights_2
    
    # Analyze influence on each attacker
    for attacker in attacker_names:
        with torch.no_grad():
            # Get attacker position and threat value
            attacker_data = scaled_graph.nodes[attacker]
            x, y = loaded_scaler.inverse_transform(
                [[attacker_data['features'][0], attacker_data['features'][1]]])[0]
            distance = np.sqrt((defender_x - x)**2 + (defender_y - y)**2)
            xt_value = get_pitch_value(xT_grid, x, y)
            
            # Calculate attention weights between defender and attacker
            mask_node = graph_names.index(defender)
            target_node = graph_names.index(attacker)
            target_mask = (edge_index_1[1] == target_node)
            source_mask = (edge_index_1[0] == mask_node)
            modify_mask = target_mask & source_mask
            
            player_attn1 = attn_weights_1[modify_mask].mean()
            player_attn2 = attn_weights_2[modify_mask].mean()
            defender_attention = float((player_attn1 + player_attn2) / 2)
            
            # Get predictions without defender
            probs_v2, _ = model(data.x, data.edge_index, data.edge_attr, None,
                               test_mode=True, mask_node_name=[defender],
                               target_node_name=attacker, graph=scaled_graph)

        # Calculate influence metrics
        attacker_index = all_names.index(attacker)
        prob_before = float(probs[attacker_index])
        prob_after = float(probs_v2[attacker_index])
        influence = prob_after - prob_before
        rel_influence = 100 * ((abs(prob_after - prob_before)) / prob_after)
        
        # Store results
        distances.append(distance)
        attentions.append(defender_attention)
        influences.append(influence)
        rel_influences.append(rel_influence)
        threats.append(xt_value)
        total_performance += influence * xt_value * 100

    return distances, attentions, influences, rel_influences, threats, total_performance


def evaluate_model_with_defender_removal(scaled_graph, model, defender_names: List[str], 
                                        attacker_names: List[str], all_names: List[str], 
                                        graph_example_metadata: Dict[str, Any], 
                                        team_rosters: pd.DataFrame, n_defenders: int = 3) -> pd.DataFrame:
    """
    Evaluate model performance with different defender removal strategies.
    
    This function tests three strategies:
    1. Remove top N defenders by attention
    2. Remove bottom N defenders by attention  
    3. Remove random N defenders
    
    Args:
        scaled_graph: NetworkX graph representing the game state
        model: Trained GAT model
        defender_names: List of defender names
        attacker_names: List of attacker names to evaluate
        all_names: List of all player names
        graph_example_metadata: Metadata about the current graph
        team_rosters: DataFrame containing team information
        n_defenders: Number of defenders to remove
        
    Returns:
        DataFrame containing results for each removal strategy
    """
    # Convert graph and get initial predictions
    data = cd.convert_graph_to_pytorch_geometric_reception(scaled_graph)
    model.eval()
    results = {}
    graph_names = [s[0] for s in scaled_graph.nodes(data=True)]

    with torch.no_grad():
        original_probs, attention_weights = model(data.x, data.edge_index, data.edge_attr, None)
        attention_weights_1, attention_weights_2 = attention_weights
        edge_index_1, attn_weights_1 = attention_weights_1
        edge_index_2, attn_weights_2 = attention_weights_2

        # Analyze each attacker
        for attacker in attacker_names:
            attacker_index = all_names.index(attacker)
            target_node = graph_names.index(attacker)
            original_prob = float(original_probs[attacker_index])

            # Calculate attention scores for all defenders
            defender_attentions = []
            for defender in defender_names:
                mask_node = graph_names.index(defender)
                target_mask = (edge_index_1[1] == target_node)
                source_mask = (edge_index_1[0] == mask_node)
                modify_mask = target_mask & source_mask
                
                if modify_mask.any():
                    player_attn1 = attn_weights_1[modify_mask].mean()
                    player_attn2 = attn_weights_2[modify_mask].mean()
                    avg_attention = float((player_attn1 + player_attn2) / 2)
                    defender_attentions.append((defender, avg_attention))

            # Sort defenders by attention
            sorted_defenders = sorted(defender_attentions, key=lambda x: x[1], reverse=True)
            top_defenders = sorted_defenders[:n_defenders]
            bottom_defenders = sorted_defenders[-n_defenders:]
            
            # Strategy 1: Remove top N defenders
            top_defender_names = [name for name, _ in top_defenders]
            probs_without_top, _ = model(data.x, data.edge_index, data.edge_attr, None,
                                       test_mode=True, mask_node_name=top_defender_names,
                                       target_node_name=attacker, graph=scaled_graph)
            prob_without_top = float(probs_without_top[attacker_index])

            # Strategy 2: Remove bottom N defenders
            bottom_defender_names = [name for name, _ in bottom_defenders]
            probs_without_bottom, _ = model(data.x, data.edge_index, data.edge_attr, None,
                                          test_mode=True, mask_node_name=bottom_defender_names,
                                          target_node_name=attacker, graph=scaled_graph)
            prob_without_bottom = float(probs_without_bottom[attacker_index])

            # Strategy 3: Remove random N defenders
            random_defenders = random.sample([d for d, _ in defender_attentions], n_defenders)
            probs_without_random, _ = model(data.x, data.edge_index, data.edge_attr, None,
                                            test_mode=True, mask_node_name=random_defenders,
                                            target_node_name=attacker, graph=scaled_graph)
            prob_without_random = float(probs_without_random[attacker_index])
            
            # Store results
            results[attacker] = {
                'Att_team': get_player_team(graph_example_metadata['gameId'], attacker, team_rosters).values[0],
                'Def_team': get_player_team(graph_example_metadata['gameId'], defender, team_rosters).values[0],
                'match_id': graph_example_metadata['gameId'],
                'match_frame': graph_example_metadata['frameNum'],
                'original_prob': original_prob,
                'is_receiver': scaled_graph.nodes[attacker]['reception_target'],
                
                # Top defenders results
                'top_defenders': top_defenders,
                'prob_without_top': prob_without_top,
                'top_diff': prob_without_top - original_prob,
                
                # Bottom defenders results
                'bottom_defenders': bottom_defenders,
                'prob_without_bottom': prob_without_bottom,
                'bottom_diff': prob_without_bottom - original_prob,
                
                # Random defenders results
                'random_defenders': random_defenders,
                'prob_without_random': prob_without_random,
                'random_diff': prob_without_random - original_prob
            }

    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.reset_index().rename(columns={'index': 'attacker'})
    return results_df


def get_graph_defender_data(graph_example, loaded_model, defender_names: List[str], 
                           attacker_names: List[str], all_names: List[str], 
                           team_rosters: pd.DataFrame, graph_example_metadata: Dict[str, Any], 
                           loaded_scaler) -> pd.DataFrame:
    """
    Get comprehensive defender analysis data for a single graph.
    
    Args:
        graph_example: NetworkX graph representing the game state
        loaded_model: Trained GAT model
        defender_names: List of defender names
        attacker_names: List of attacker names
        all_names: List of all player names
        team_rosters: DataFrame containing team information
        graph_example_metadata: Metadata about the current graph
        loaded_scaler: Scaler for coordinate transformation
        
    Returns:
        DataFrame containing defender analysis results
    """
    all_defender_dataframes = []

    # Analyze each defender
    for defender in defender_names:
        # Calculate influences for current defender
        distances, attentions, influences, rel_influences, threats, performance = calculate_defender_influence(
            graph_example, loaded_model, defender, attacker_names, all_names, loaded_scaler
        )

        # Create dataframe for current defender
        defender_df = pd.DataFrame({
            'Attacker': attacker_names,
            'Defender': defender,
            'Att_team': [get_player_team(graph_example_metadata['gameId'], a, team_rosters).values[0] 
                        for a in attacker_names],
            'Def_team': get_player_team(graph_example_metadata['gameId'], defender, team_rosters).values[0],
            'match_id': graph_example_metadata['gameId'],
            'match_frame': graph_example_metadata['frameNum'],
            'Def_Distance': distances,
            'Def_Attention': attentions,
            'Def_Influence': influences,
            'Def_Rel_Influence (%)': rel_influences,
            'Att_Threat': threats,
            'Total_Perf': performance
        })

        all_defender_dataframes.append(defender_df)

    # Combine all defender dataframes
    combined_defender_dataframe = pd.concat(all_defender_dataframes, ignore_index=True)
    return combined_defender_dataframe


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main(match_id: int):
    """
    Run defender analysis for a specific match ID and save results.
    
    This function processes all graphs for a given match, calculates defender
    influences and removal experiments, and saves the results to CSV files.
    
    Args:
        match_id: The ID of the match to analyze
    """
    try:
        # Set working directory and load global data
        os.chdir('PFF_2023-24_Data')
        global xT_grid
        xT_grid = pd.read_csv('xT_grid.csv', header=None)
        
        # Create match-specific output directory
        output_dir = f'match_{match_id}_analysis'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        # Load necessary data files
        print("Loading data files...")
        with open('position_scaler.pkl', 'rb') as file:
            loaded_scaler = pickle.load(file)
            
        team_rosters = pd.read_csv('rosters_updated.csv')

        with open(f'graphs_scaled_version/{match_id}_graphs.pkl', 'rb') as file:
            scaled_graphs = pickle.load(file)

        scaled_graphs_metadata = pd.read_csv(f'graphs_scaled_version_metadata/{match_id}_metadata.csv')

        # Initialize and load the GAT model
        print("Initializing model...")
        graph_example = scaled_graphs[0]
        example_data = [convert_graph_to_pytorch_geometric_reception(graph_example)]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loaded_model = GATReceptionPredictor(
            num_node_features=example_data[0].x.size(1),
            num_edge_features=example_data[0].edge_attr.size(1),
            hidden_channels=32,
            edge_hidden_channels=16
        ).to(device)
        loaded_model.load_state_dict(torch.load('graphs/gat_model2.pth', 
                                               map_location=device))

        # Process all graphs in the match
        print(f"Processing {len(scaled_graphs)} graphs...")
        all_graph_dataframes = []
        defender_1_dataframes = []
        defender_2_dataframes = []
        defender_3_dataframes = []

        for i, graph_example in enumerate(scaled_graphs):
            if i % 100 == 0:  # Progress indicator
                print(f"Processing graph {i+1}/{len(scaled_graphs)}")
                
            # Get metadata and player lists
            graph_example_metadata = scaled_graphs_metadata.iloc[i]
            all_names = [s[0] for s in graph_example.nodes(data=True)]
            attacker_names = [s[0] for s in graph_example.nodes(data=True) 
                            if (s[1]['features'][7] == 1) & (s[0] != 'ball')]
            defender_names = [s[0] for s in graph_example.nodes(data=True) 
                            if (s[1]['features'][7] == 0) & (s[0] != 'ball')]

            # Run all analyses
            graph_defender_data = get_graph_defender_data(
                graph_example, loaded_model, defender_names, attacker_names, 
                all_names, team_rosters, graph_example_metadata, loaded_scaler
            )
            
            defender_removal_1 = evaluate_model_with_defender_removal(
                graph_example, loaded_model, defender_names, attacker_names, 
                all_names, graph_example_metadata, team_rosters, n_defenders=1
            )
            
            defender_removal_2 = evaluate_model_with_defender_removal(
                graph_example, loaded_model, defender_names, attacker_names, 
                all_names, graph_example_metadata, team_rosters, n_defenders=2
            )
            
            defender_removal_3 = evaluate_model_with_defender_removal(
                graph_example, loaded_model, defender_names, attacker_names, 
                all_names, graph_example_metadata, team_rosters, n_defenders=3
            )

            # Collect results
            all_graph_dataframes.append(graph_defender_data)
            defender_1_dataframes.append(defender_removal_1)
            defender_2_dataframes.append(defender_removal_2)
            defender_3_dataframes.append(defender_removal_3)

        # Combine all results
        print("Combining results...")
        combined_defender_dataframe = pd.concat(all_graph_dataframes, ignore_index=True)
        combined_defender_1_dataframe = pd.concat(defender_1_dataframes, ignore_index=True)
        combined_defender_2_dataframe = pd.concat(defender_2_dataframes, ignore_index=True)
        combined_defender_3_dataframe = pd.concat(defender_3_dataframes, ignore_index=True)

        # Save results to CSV files
        print("Saving results...")
        combined_defender_dataframe.to_csv(f'{match_id}_defender_performance_dataframe.csv', index=False)
        combined_defender_1_dataframe.to_csv(f'{match_id}_defender_model1_dataframe.csv', index=False)
        combined_defender_2_dataframe.to_csv(f'{match_id}_defender_model2_dataframe.csv', index=False)
        combined_defender_3_dataframe.to_csv(f'{match_id}_defender_model3_dataframe.csv', index=False)

        print(f"Analysis completed successfully for match {match_id}!")
        print(f"Generated {len(combined_defender_dataframe)} defender-attacker pairs")
        print("Files saved:")
        print(f"  - {match_id}_defender_performance_dataframe.csv")
        print(f"  - {match_id}_defender_model1_dataframe.csv")
        print(f"  - {match_id}_defender_model2_dataframe.csv")
        print(f"  - {match_id}_defender_model3_dataframe.csv")

    except Exception as e:
        print(f"Error processing match {match_id}: {str(e)}")
        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python defender_analysis.py <match_id>")
        print("Example: python defender_analysis.py 12345")
        sys.exit(1)

    try:
        match_id = int(sys.argv[1])
        print(f"Starting defender analysis for match {match_id}...")
        main(match_id)
    except ValueError:
        print("Error: match_id must be an integer")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)