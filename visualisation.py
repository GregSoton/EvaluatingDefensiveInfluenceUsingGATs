"""
Interactive Football Graph Visualization Tool

This module provides an interactive visualization system for analyzing football
reception prediction models using Graph Attention Networks (GAT). It allows users
to explore defender influences, attention weights, and player interactions.

Key Features:
- Real-time defender influence analysis
- Attention weight visualization between players
- Reception probability calculations
- Performance metrics display
- Graph navigation and comparison tools

Dependencies:
- matplotlib, mplsoccer for pitch visualization
- ipywidgets for interactive controls
- torch, torch_geometric for neural network operations
- networkx for graph operations
- Custom modules for data conversion and model operations
"""

import os
import pickle
import gc
from typing import Dict, List, Tuple, Any, Optional

# Visualization and UI imports
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import ipywidgets as widgets
from IPython.display import clear_output, display

# Scientific computing imports
import numpy as np
import pandas as pd
import networkx as nx

# PyTorch and PyTorch Geometric imports
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GENConv, SAGEConv, GATv2Conv
from torch_geometric.data import Data, DataLoader

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

# Custom module imports
import convert_tracking as ct
import plot_functions as pf
import create_graph as cg
import scale_graph as sg
import GNNs.model_training as mt
import GNNs.convert_data as cd
from GNNs.custom_GAT import myGATv2Conv
from GNNs.GNN import ReceptionPredictionGNN
from GNNs.GAT import GATReceptionPredictor


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_pitch_value(data: np.ndarray, x: float, y: float, 
                   pitch_length: float = 105, pitch_width: float = 68) -> float:
    """
    Get the Expected Threat (xT) value for a specific pitch location.
    
    Args:
        data: xT grid data array
        x: X coordinate on pitch (0-105m)
        y: Y coordinate on pitch (0-68m)
        pitch_length: Length of the pitch in meters
        pitch_width: Width of the pitch in meters
        
    Returns:
        xT value at the specified location
        
    Raises:
        ValueError: If coordinates are outside pitch boundaries
    """
    if not (0 <= x <= pitch_length and 0 <= y <= pitch_width):
        raise ValueError(f"Coordinates must be within pitch dimensions: "
                        f"0-{pitch_length}m (length) and 0-{pitch_width}m (width)")

    # Calculate grid dimensions and cell sizes
    rows, cols = data.shape[0], data.shape[1]
    cell_length = pitch_length / cols
    cell_width = pitch_width / rows
    
    # Convert coordinates to grid indices
    col_index = min(int(x / cell_length), cols - 1)
    row_index = min(int(y / cell_width), rows - 1)
    
    return data[col_index][row_index]


def predict_reception_probabilities(model_AT, graph, head_indexes: range = range(16), 
                                   return_player_info: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Predict reception probabilities and analyze attention weights for all players.
    
    This function runs the GAT model on a graph to get reception predictions and
    extracts attention weights to understand which players the model focuses on
    when making predictions.
    
    Args:
        model_AT: Trained GAT model for reception prediction
        graph: NetworkX graph representing the game state
        head_indexes: Which attention heads to average over
        return_player_info: Whether to return detailed player information
        
    Returns:
        Tuple containing:
        - DataFrame with player reception probabilities (attacking players only)
        - Dictionary with attention analysis for each player
    """
    device = next(model_AT.parameters()).device
    
    # Convert graph to PyTorch Geometric format
    data = cd.convert_graph_to_pytorch_geometric_reception(graph).to(device)
    
    # Get model predictions and attention weights
    model_AT.eval()
    with torch.no_grad():
        probs_AT, attention_weights = model_AT(data.x, data.edge_index, data.edge_attr, None)
        
    # Extract attention weights from first GAT layer
    edge_index, attention_weights = attention_weights[0][0], (attention_weights[0][1][:, head_indexes])
    
    # Process attention weights
    attention_weights = attention_weights.detach().cpu().numpy()
    avg_attention = attention_weights.mean(axis=1)

    # Extract player information
    player_names = []
    is_attackings = []
    for node, node_data in graph.nodes(data=True):
        player_names.append(node)
        if node != 'ball':
            is_attackings.append(bool(int(node_data['features'][7])))
        else:
            is_attackings.append(False)
    
    # Create results DataFrame for attacking players only
    results_df = pd.DataFrame({
        'player_name': player_names,
        'is_attacking': is_attackings,
        'reception_probability_AT': probs_AT
    })
    results_df = results_df[results_df['is_attacking'] == True].reset_index(drop=True)
    results_df = results_df.sort_values('reception_probability_AT', ascending=False)
    
    # Analyze attention weights for each player
    attention_analysis = {}
    node_list = list(graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(node_list)}
  
    for node in graph.nodes():
        node_idx = node_mapping[node]

        # Find all edges where this node is the target (incoming attention)
        target_mask = edge_index[1] == node_idx
        source_nodes = edge_index[0][target_mask]
        weights = avg_attention[target_mask]

        # Create sorted list of attention weights from other players
        attention_info = []
        for src_idx, weight in zip(source_nodes, weights):
            if src_idx < len(node_list):  # Ensure valid index
                source_name = node_list[src_idx]
                attention_info.append({
                    'source_player': source_name,
                    'attention_weight': weight
                })

        # Sort by attention weight (highest first)
        attention_info = sorted(attention_info, key=lambda x: x['attention_weight'], reverse=True)
        attention_analysis[node] = attention_info
        
    return results_df, attention_analysis


def update_graph_after_position_change(graph: nx.Graph, player_id: str, 
                                      new_x: float, new_y: float) -> nx.Graph:
    """
    Update graph with new player position and recalculate dependent features.
    
    This function updates a player's position in the graph and recalculates
    all position-dependent features like distances and relative positions.
    
    Args:
        graph: Original NetworkX graph
        player_id: ID of the player to move
        new_x: New X coordinate
        new_y: New Y coordinate
        
    Returns:
        Updated NetworkX graph with recalculated features
    """
    # Create a copy of the graph to avoid modifying the original
    updated_graph = graph.copy()
    
    # Update player position in node features
    if player_id in updated_graph.nodes:
        node_data = updated_graph.nodes[player_id]
        node_data['features'][0] = new_x  # X position
        node_data['features'][1] = new_y  # Y position
        
        # Recalculate edge features that depend on positions
        for u, v, edge_data in updated_graph.edges(data=True):
            if u == player_id or v == player_id:
                # Get positions of both nodes
                u_pos = updated_graph.nodes[u]['features'][:2]
                v_pos = updated_graph.nodes[v]['features'][:2]
                
                # Recalculate distance-based edge features
                dx = v_pos[0] - u_pos[0]
                dy = v_pos[1] - u_pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Update edge features (assuming first two are dx, dy)
                edge_data['features'][0] = dx
                edge_data['features'][1] = dy
                # Add distance if it's part of edge features
                if len(edge_data['features']) > 2:
                    edge_data['features'][2] = distance
    
    return updated_graph


# ============================================================================
# INTERACTIVE PLAYER CLASS
# ============================================================================

class DraggablePlayer:
    """
    A draggable player representation for interactive visualization.
    """
    
    def __init__(self, circle, text: str, player_id: str, G: nx.Graph, 
                 graphs: List[nx.Graph], graph_index: int, update_plot_callback):
        """
        Initialize a player.
        
        Args:
            circle: Matplotlib circle object representing the player
            text: Text label for the player
            player_id: Unique identifier for the player
            G: Current graph containing the player
            graphs: List of all graphs in the visualization
            graph_index: Index of current graph in the graphs list
            update_plot_callback: Function to call when plot needs updating
        """
        self.circle = circle
        self.text = text
        self.player_id = player_id
        self.G = G
        self.graphs = graphs
        self.graph_index = graph_index
        self.update_plot_callback = update_plot_callback
        self.press = None

        # Connect mouse event handlers
        self.cidpress = circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        """Handle mouse press events to start dragging."""
        if event.inaxes != self.circle.axes:
            return
            
        contains, _ = self.circle.contains(event)
        if contains:
            # Store offset between mouse and circle center
            self.press = (self.circle.center[0] - event.xdata, 
                         self.circle.center[1] - event.ydata)

    def on_motion(self, event):
        """Handle mouse motion events during dragging."""
        if self.press is None or event.inaxes != self.circle.axes:
            return
            
        # Calculate new position
        dx, dy = self.press
        new_x = event.xdata + dx
        new_y = event.ydata + dy

        # Update visual elements
        self.circle.center = (new_x, new_y)
        if hasattr(self.text, 'set_position'):
            self.text.set_position((new_x, new_y + 1))
        
        # Redraw the figure
        self.circle.figure.canvas.draw_idle()

    def on_release(self, event):
        """Handle mouse release events to finalize position change."""
        if self.press is None:
            return

        try:
            # Get the final position
            new_x, new_y = self.circle.center

            # Update the graph with new position
            updated_graph = update_graph_after_position_change(
                self.G, self.player_id, new_x, new_y
            )

            # Update the graph in the graphs list
            self.graphs[self.graph_index] = updated_graph
            self.G = updated_graph

            # Trigger full visualization update
            self.update_plot_callback()

        except Exception as e:
            print(f"Error updating graph for player {self.player_id}: {str(e)}")
        finally:
            self.press = None

    def disconnect(self):
        """Disconnect all event callbacks to prevent memory leaks."""
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)


# ============================================================================
# MAIN VISUALIZATION FUNCTION
# ============================================================================

def create_simple_visualization(model_reception_AT, graphs: List[nx.Graph], 
                               scaled_graphs: List[nx.Graph], xT_grid: np.ndarray):
    """
    Create an interactive visualization for football graph analysis.
    
    This function creates a comprehensive interactive visualization that allows
    users to explore different aspects of the football reception prediction model,
    including defender influences, attention weights, and player interactions.
    
    Args:
        model_reception_AT: Trained GAT model for reception prediction
        graphs: List of original NetworkX graphs
        scaled_graphs: List of scaled NetworkX graphs for model input
        xT_grid: Expected Threat grid for pitch value calculations
    """
    # State management for draggable elements
    visualization_state = {'draggable_players': []}

    # ========================================================================
    # UI CONTROLS SETUP
    # ========================================================================
    
    # Main graph navigation control
    graph_selector = widgets.IntSlider(
        value=0,
        min=0,
        max=len(graphs) - 1,
        description='Graph Index:',
        continuous_update=False,
        style={'description_width': 'initial'}
    )

    # Defender analysis controls
    defender_select = widgets.Dropdown(
        description='Defender:',
        options=['Select Defender'],
        disabled=False,
        style={'description_width': 'initial'}
    )

    show_defender_influence = widgets.Checkbox(
        value=False,
        description='Show Defender Influence (Absolute)',
        disabled=False
    )

    show_defender_performances = widgets.Checkbox(
        value=False,
        description='Show Defender Performances',
        disabled=False
    )

    # Player edge and attention analysis controls
    show_player_edges = widgets.Checkbox(
        value=False,
        description='Show Player Edges',
        disabled=False
    )
    
    show_attention_weights = widgets.Checkbox(
        value=False,
        description='Show Attention Weights',
        disabled=False
    )
    
    player_edge_select = widgets.Dropdown(
        description='Player:',
        options=['Select Player'],
        disabled=False,
        style={'description_width': 'initial'}
    )

    # Output widget for the visualization
    output = widgets.Output()

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def update_defender_list(change=None):
        """Update the defender dropdown based on current graph."""
        G = graphs[graph_selector.value]
        defenders = [
            node for node, data in G.nodes(data=True)
            if node != 'ball' and data['features'][7] == 0  # Defending team
        ]
        defender_select.options = ['Select Defender'] + defenders

    def update_player_edge_list(change=None):
        """Update the player dropdown for edge analysis."""
        G = graphs[graph_selector.value]
        players = [node for node in G.nodes if node != 'ball']
        player_edge_select.options = ['Select Player'] + players

    def calculate_defender_influence() -> Optional[Dict[str, float]]:
        """
        Calculate how removing a specific defender affects reception probabilities.
        
        Returns:
            Dictionary mapping attacker names to their influence values
        """
        if not (defender_select.value and defender_select.value != 'Select Defender'):
            return None
            
        G = graphs[graph_selector.value]
        SG = scaled_graphs[graph_selector.value]
        influences = {}

        # Get model data
        data = cd.convert_graph_to_pytorch_geometric_reception(SG)
        all_names = [s[0] for s in G.nodes(data=True)]
        attacker_names = [s[0] for s in G.nodes(data=True)
                         if (s[1]['features'][7] == 1) & (s[0] != 'ball')]

        model_reception_AT.eval()
        
        # Calculate influence for each attacker
        for attacker in attacker_names:
            with torch.no_grad():
                # Get original probabilities
                probs, _ = model_reception_AT(data.x, data.edge_index, data.edge_attr, None)
                
                # Get probabilities with defender removed
                probs_v2, _ = model_reception_AT(
                    data.x, data.edge_index, data.edge_attr, None,
                    test_mode=True,
                    mask_node_name=defender_select.value,
                    target_node_name=attacker,
                    graph=SG
                )

            # Calculate absolute influence
            attacker_index = all_names.index(attacker)
            prob_before = float(probs[attacker_index])
            prob_after = float(probs_v2[attacker_index])
            absolute_influence = prob_after - prob_before
            influences[attacker] = absolute_influence

        return influences

    def get_reception_probabilities() -> Dict[str, float]:
        """
        Get normalized reception probabilities for all attackers.
        
        Returns:
            Dictionary mapping attacker names to normalized probabilities
        """
        G = graphs[graph_selector.value]
        SG = scaled_graphs[graph_selector.value]
        
        # Get model predictions
        data = cd.convert_graph_to_pytorch_geometric_reception(SG)
        all_names = [s[0] for s in G.nodes(data=True)]
        attacker_names = [s[0] for s in G.nodes(data=True)
                         if (s[1]['features'][7] == 1) & (s[0] != 'ball')]
        
        model_reception_AT.eval()
        with torch.no_grad():
            probs, _ = model_reception_AT(data.x, data.edge_index, data.edge_attr, None)
        
        # Extract and normalize probabilities
        raw_probs = {}
        for attacker in attacker_names:
            attacker_index = all_names.index(attacker)
            raw_probs[attacker] = float(probs[attacker_index])
        
        # Normalize to sum to 1
        total_prob = sum(raw_probs.values())
        if total_prob > 0:
            return {attacker: prob / total_prob for attacker, prob in raw_probs.items()}
        else:
            # Equal distribution if all probabilities are 0
            equal_prob = 1.0 / len(attacker_names)
            return {attacker: equal_prob for attacker in attacker_names}

    def calculate_defender_performance(defender_name: str, G: nx.Graph, SG: nx.Graph) -> float:
        """
        Calculate overall performance value for a defender.
        
        This combines the defender's influence on each attacker with the threat
        level of that attacker's position to get an overall performance metric.
        
        Args:
            defender_name: Name of the defender to analyze
            G: Original graph
            SG: Scaled graph for model input
            
        Returns:
            Overall performance value for the defender
        """
        data = cd.convert_graph_to_pytorch_geometric_reception(SG)
        all_names = [s[0] for s in G.nodes(data=True)]
        attacker_names = [s[0] for s in G.nodes(data=True)
                         if (s[1]['features'][7] == 1) & (s[0] != 'ball')]

        total_performance = 0
        model_reception_AT.eval()

        for attacker in attacker_names:
            # Get attacker position and threat value
            attacker_data = G.nodes[attacker]
            x, y = attacker_data['features'][0], attacker_data['features'][1]
            xt_value = get_pitch_value(xT_grid, x, y)

            # Calculate defender's influence on this attacker
            with torch.no_grad():
                probs, _ = model_reception_AT(data.x, data.edge_index, data.edge_attr, None)
                probs_v2, _ = model_reception_AT(
                    data.x, data.edge_index, data.edge_attr, None,
                    test_mode=True,
                    mask_node_name=defender_name,
                    target_node_name=attacker,
                    graph=SG
                )

            # Calculate influence and weight by threat
            attacker_index = all_names.index(attacker)
            prob_before = float(probs[attacker_index])
            prob_after = float(probs_v2[attacker_index])
            influence = prob_after - prob_before
            
            # Add weighted influence to total performance
            total_performance += influence * xt_value * 100

        return total_performance

    def get_attention_weights_for_player(selected_player: str) -> Dict[str, float]:
        """
        Get attention weights for outgoing edges from a selected player.
        
        Args:
            selected_player: Name of the player to analyze
            
        Returns:
            Dictionary mapping target players to attention weights
        """
        SG = scaled_graphs[graph_selector.value]
        
        # Get attention analysis using the prediction function
        _, attention_analysis = predict_reception_probabilities(model_reception_AT, SG)
        
        # Convert to outgoing edges format (selected_player -> others)
        outgoing_weights = {}
        
        # Look through all nodes to find where selected_player is a source
        for target_player, attention_info in attention_analysis.items():
            for source_info in attention_info:
                if source_info['source_player'] == selected_player:
                    outgoing_weights[target_player] = source_info['attention_weight']
                    break
        
        return outgoing_weights

    # ========================================================================
    # MAIN PLOTTING FUNCTION
    # ========================================================================

    def update_plot(change=None):
        """
        Main function to update the visualization plot.
        
        This function handles all the drawing logic including players, edges,
        attention weights, and various metrics displays.
        """
        # Clear previous plots and disconnect old draggable elements
        plt.close('all')
        for draggable in visualization_state['draggable_players']:
            draggable.disconnect()
        
        with output:
            clear_output(wait=True)

            # Create new figure and pitch
            fig, ax = plt.subplots(figsize=(30, 15))
            pitch = Pitch(pitch_type="uefa", pitch_length=105, pitch_width=68, 
                         axis=False, label=False)
            pitch.draw(ax=ax)

            # Get current graphs
            G = graphs[graph_selector.value]
            SG = scaled_graphs[graph_selector.value]

            # Calculate metrics if needed
            defender_influences = None
            defender_selected = (show_defender_influence.value and 
                               defender_select.value != 'Select Defender')
            if defender_selected:
                defender_influences = calculate_defender_influence()
            
            reception_probabilities = None
            if not show_defender_performances.value:
                reception_probabilities = get_reception_probabilities()

            # Reset draggable players list
            visualization_state['draggable_players'] = []

            # ================================================================
            # DRAW PLAYERS AND BASIC ELEMENTS
            # ================================================================
            
            for player_id, data in G.nodes(data=True):
                x, y = data['features'][0], data['features'][1]
                is_attacker = data['features'][7] == 1

                # Determine player color
                if player_id == 'ball':
                    color = 'black'
                elif player_id == defender_select.value:
                    color = 'green'  # Highlight selected defender
                else:
                    color = 'red' if is_attacker else 'blue'

                # Draw player circle
                circle = plt.Circle((x, y), radius=1.7, color=color, alpha=0.8)
                ax.add_patch(circle)

                # Add player name and make draggable (except ball)
                if player_id != 'ball':
                    text = ax.text(x, y + 2.5, player_id, fontsize=14, ha='center',
                                  bbox=dict(facecolor='lightgray', alpha=0.5), zorder=1)
                    
                    # Create draggable player
                    draggable = DraggablePlayer(
                        circle, text, player_id, G, graphs,
                        graph_selector.value, lambda: update_plot()
                    )
                    visualization_state['draggable_players'].append(draggable)

                # Draw velocity vectors for players
                if player_id != 'ball':
                    velocities = data['features'][2:4]
                    ax.quiver(x, y, velocities[0], velocities[1],
                             color='k', scale=50, width=0.003, alpha=1)

            # ================================================================
            # DRAW EDGES AND ATTENTION WEIGHTS
            # ================================================================
            
            if ((show_player_edges.value or show_attention_weights.value) and 
                player_edge_select.value and player_edge_select.value != 'Select Player'):
                
                selected_player = player_edge_select.value
                selected_is_attacker = G.nodes[selected_player]['features'][7] == 1

                # Get outgoing edges for selected player
                outgoing_edges = [e for e in G.edges(selected_player)]

                # Get attention weights if needed
                attn_weights = {}
                if show_attention_weights.value:
                    attn_weights = get_attention_weights_for_player(selected_player)

                # Draw edges to opponent players only
                for _, target in outgoing_edges:
                    target_is_attacker = G.nodes[target]['features'][7] == 1
                    is_opponent = (selected_is_attacker != target_is_attacker)
                    
                    if not is_opponent:
                        continue  # Skip teammates

                    # Get positions
                    x1, y1 = G.nodes[selected_player]['features'][0], G.nodes[selected_player]['features'][1]
                    x2, y2 = G.nodes[target]['features'][0], G.nodes[target]['features'][1]

                    # Style edge based on attention weight
                    if show_attention_weights.value and target in attn_weights:
                        weight = attn_weights[target]
                        # Nonlinear scaling for dramatic visual effect
                        min_width, max_width = 0.5, 100
                        normalized_weight = min(max(weight, 0), 1)
                        linewidth = min_width + (max_width - min_width) * (normalized_weight ** 2)
                        color = plt.cm.viridis(normalized_weight)
                        alpha = 0.8 + 0.2 * normalized_weight
                    else:
                        color = 'orange'
                        linewidth = 2
                        alpha = 0.7

                    # Draw the edge
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                           alpha=alpha, zorder=20)

            # ================================================================
            # DRAW METRIC DISPLAYS
            # ================================================================
            
            for player_id, data in G.nodes(data=True):
                x, y = data['features'][0], data['features'][1]
                is_attacker = data['features'][7] == 1
                is_defender = not is_attacker and player_id != 'ball'
                label_offset = -1.7  # Starting position for labels below player

                # Show defender influence on attackers
                if (show_defender_influence.value and defender_influences and 
                    is_attacker and player_id != 'ball'):
                    
                    influence = defender_influences.get(player_id, 0)
                    influence_text = f"DI: {influence*100:.1f}%"
                    color = 'red' if influence > 0 else 'green'
                    ax.text(x, y + label_offset, influence_text,
                           color=color, ha='center', fontsize=14,
                           bbox=dict(facecolor='white', alpha=1), zorder=30)
                    label_offset -= 1.7

                    # Also show Expected Threat value
                    pitch_value = get_pitch_value(xT_grid, x, y)
                    xt_text = f"xT: {pitch_value:.3f}"
                    ax.text(x, y + label_offset, xt_text,
                           color='black', ha='center', fontsize=14,
                           bbox=dict(facecolor='white', alpha=1), zorder=30)

                # Show defender performance values
                if show_defender_performances.value and is_defender:
                    performance = calculate_defender_performance(player_id, G, SG)
                    perf_text = f"{performance:.3f}"
                    color = 'green' if performance > 0 else 'red'
                    ax.text(x, y - 2, perf_text, fontsize=14,
                           color=color, ha='center',
                           bbox=dict(facecolor='white', alpha=1), zorder=30)

            # Set plot title and display
            plt.title(f'Interactive Football Graph Visualization - Frame {graph_selector.value}', 
                     fontsize=16, pad=20)
            plt.show()

    # ========================================================================
    # CONNECT CONTROLS AND INITIALIZE
    # ========================================================================

    # Connect all controls to the update function
    graph_selector.observe(update_plot, names='value')
    defender_select.observe(update_plot, names='value')
    show_defender_influence.observe(update_plot, names='value')
    show_defender_performances.observe(update_plot, names='value')
    show_player_edges.observe(update_plot, names='value')
    show_attention_weights.observe(update_plot, names='value')
    player_edge_select.observe(update_plot, names='value')
    
    # Connect dropdown update functions
    graph_selector.observe(update_defender_list, names='value')
    graph_selector.observe(update_player_edge_list, names='value')

    # ========================================================================
    # CREATE UI LAYOUT
    # ========================================================================

    # Group related controls
    defender_controls = widgets.VBox([
        widgets.HTML(value='<b>Defender Analysis:</b>'),
        defender_select,
        show_defender_influence,
        show_defender_performances
    ])

    edge_controls = widgets.VBox([
        widgets.HTML(value='<b>Player Edge Analysis:</b>'),
        player_edge_select,
        show_player_edges,
        show_attention_weights
    ])

    # Main control panel
    controls = widgets.VBox([
        widgets.HTML(value='<h3>Interactive Football Graph Visualization</h3>'),
        graph_selector,
        defender_controls,
        edge_controls
    ])

    # Initialize dropdowns and display
    update_defender_list()
    update_player_edge_list()
    update_plot()

    # Display the complete interface
    display(controls, output)
  