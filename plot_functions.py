from mplsoccer import Pitch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import pandas as pd


# ============================================================================
# PITCH CREATION
# ============================================================================

def plot_pitch():
    """Create a standard UEFA football pitch."""
    pitch = Pitch(
        pitch_type="uefa", pitch_length=105, pitch_width=68, axis=False, label=False
    )
    fig, ax = pitch.draw()
    return fig, ax


# ============================================================================
# STATIC PLAYER VISUALIZATION
# ============================================================================

def plot_players_on_pitch(players_df, balls_df, annotate=True, show_velocities=True):
    """Plot player positions on pitch with optional annotations and velocity vectors."""
    fig, ax = plot_pitch()

    # Plot ball once (moved outside player loop)
    ball_data = balls_df[['x','y']].values[0]
    ax.scatter(ball_data[0], ball_data[1], s=50, label='Ball', color='black')

    # Add velocity reference scale once (moved outside player loop)
    if show_velocities:
        ref_velocity = 5  # reference velocity in m/s
        ax.quiver(5, 5, ref_velocity, 0, 
               color='gray', scale=50, width=0.003)
        ax.text(5, 3, f'{ref_velocity} m/s', 
              fontsize=8, ha='center', va='center', color='gray')

    # Loop through players
    for index, row in players_df.iterrows():
        x, y = row['x'], row['y']
        color = 'red' if row['is_home'] else 'blue'
        
        # Plot player position
        ax.scatter(x, y, color=color, s=100, label=row['playerName'])
        
        # Add player name annotation
        if annotate:
            player_surname = row['playerName'].split(' ')[-1]
            ax.text(x, y+1.5, player_surname, 
                 fontsize=8, ha='center', va='center', color='black')

        # Add velocity vectors
        if show_velocities and _has_velocity_data(players_df):
            scale_factor = 1.0
            ax.quiver(x, y, 
                   row['velocity_x'] * scale_factor, 
                   row['velocity_y'] * scale_factor,
                   color=color, scale=50, width=0.003, alpha=0.7)

            # Add velocity magnitude text
            if 'velocity_magnitude' in players_df.columns:
                velocity_mag = row['velocity_magnitude']
                ax.text(x, y-1.5, f'{velocity_mag:.1f}', 
                     fontsize=7, ha='center', va='center', color=color)

    return fig, ax


def _has_velocity_data(players_df):
    """Check if velocity data is available in the dataframe."""
    return 'velocity_x' in players_df.columns and 'velocity_y' in players_df.columns


# ============================================================================
# ANIMATION FUNCTIONALITY
# ============================================================================

def animate_player_positions(players_df, balls_df):
    """Create animated visualization of player movements over time."""
    fig, ax = plot_pitch()

    # Initialize visual elements
    home_scatter, away_scatter, ball_scatter = _initialize_scatter_plots(ax)
    home_quiver, away_quiver = _initialize_velocity_vectors(ax, players_df)
    home_texts, away_texts = _initialize_text_labels(ax, players_df)
    
    def init():
        """Initialize animation elements."""
        return _reset_animation_elements(
            home_scatter, away_scatter, ball_scatter,
            home_quiver, away_quiver, home_texts, away_texts
        )

    def update(frame_num):
        """Update animation for current frame."""
        return _update_animation_frame(
            frame_num, players_df, balls_df, ax,
            home_scatter, away_scatter, ball_scatter,
            home_quiver, away_quiver, home_texts, away_texts
        )

    # Create animation
    frame_numbers = sorted(players_df['frameNum'].unique())
    anim = animation.FuncAnimation(fig, update, frames=frame_numbers, 
                                 init_func=init, blit=True, repeat=False)
    
    plt.show()
    return anim


def _initialize_scatter_plots(ax):
    """Initialize scatter plot objects for animation."""
    home_scatter = ax.scatter([], [], color='red', s=100, label='Home Team')
    away_scatter = ax.scatter([], [], color='blue', s=100, label='Away Team')
    ball_scatter = ax.scatter([], [], color='black', s=60, label='Ball', 
                             marker='o', edgecolor='black')
    return home_scatter, away_scatter, ball_scatter


def _initialize_velocity_vectors(ax, players_df):
    """Initialize velocity vector objects for animation."""
    first_frame = players_df['frameNum'].iloc[0]
    initial_home = players_df[(players_df['frameNum'] == first_frame) & 
                             (players_df['is_home'] == True)]
    initial_away = players_df[(players_df['frameNum'] == first_frame) & 
                             (players_df['is_home'] == False)]

    home_quiver = ax.quiver(np.zeros(len(initial_home)), np.zeros(len(initial_home)), 
                          np.zeros(len(initial_home)), np.zeros(len(initial_home)), 
                          color='red', scale=50, width=0.003, alpha=0.7)
    away_quiver = ax.quiver(np.zeros(len(initial_away)), np.zeros(len(initial_away)), 
                          np.zeros(len(initial_away)), np.zeros(len(initial_away)), 
                          color='blue', scale=50, width=0.003, alpha=0.7)
    
    return home_quiver, away_quiver


def _initialize_text_labels(ax, players_df):
    """Initialize text label objects for animation."""
    first_frame = players_df['frameNum'].iloc[0]
    initial_home = players_df[(players_df['frameNum'] == first_frame) & 
                             (players_df['is_home'] == True)]
    initial_away = players_df[(players_df['frameNum'] == first_frame) & 
                             (players_df['is_home'] == False)]

    home_texts = [ax.text(0, 0, '', fontsize=8, ha='left', va='center', color='black') 
                 for _ in range(len(initial_home))]
    away_texts = [ax.text(0, 0, '', fontsize=8, ha='left', va='center', color='black') 
                 for _ in range(len(initial_away))]
    
    return home_texts, away_texts


def _reset_animation_elements(home_scatter, away_scatter, ball_scatter,
                            home_quiver, away_quiver, home_texts, away_texts):
    """Reset all animation elements to empty state."""
    home_scatter.set_offsets(np.empty((0, 2)))
    away_scatter.set_offsets(np.empty((0, 2)))
    ball_scatter.set_offsets(np.empty((0, 2)))
    
    # Reset text elements
    for text in home_texts + away_texts:
        text.set_position((0, 0))
        text.set_text('')
    
    return (home_scatter, away_scatter, ball_scatter, home_quiver, away_quiver, 
            *home_texts, *away_texts)


def _update_animation_frame(frame_num, players_df, balls_df, ax,
                          home_scatter, away_scatter, ball_scatter,
                          home_quiver, away_quiver, home_texts, away_texts):
    """Update all animation elements for current frame."""
    # Get current frame data
    frame_data = players_df[players_df['frameNum'] == frame_num]
    ball_data = balls_df[balls_df['frameNum'] == frame_num]
    
    # Split data by team
    home_data = frame_data[frame_data['is_home'] == True]
    away_data = frame_data[frame_data['is_home'] == False]
    
    # Update player positions
    home_scatter.set_offsets(home_data[['x', 'y']].values)
    away_scatter.set_offsets(away_data[['x', 'y']].values)
    
    # Update velocity vectors
    _update_velocity_vectors(home_quiver, home_data)
    _update_velocity_vectors(away_quiver, away_data)
    
    # Update player labels
    _update_player_labels(home_texts, home_data)
    _update_player_labels(away_texts, away_data)
    
    # Update ball position
    if not ball_data.empty:
        ball_scatter.set_offsets(ball_data[['x', 'y']].values)
    else:
        ball_scatter.set_offsets(np.empty((0, 2)))
    
    # Update title
    ax.set_title(f'Player Positions - Frame {frame_num}', fontsize=16)
    
    return (home_scatter, away_scatter, ball_scatter, home_quiver, away_quiver, 
            *home_texts, *away_texts)


def _update_velocity_vectors(quiver, data):
    """Update velocity vectors for a team."""
    if _has_velocity_data(data) and not data.empty:
        scale_factor = 1.0
        quiver.set_offsets(data[['x', 'y']].values)
        quiver.set_UVC(data['velocity_x'].values * scale_factor,
                      data['velocity_y'].values * scale_factor)


def _update_player_labels(texts, data):
    """Update player name labels for a team."""
    for i, (x, y, player_name) in enumerate(zip(data['x'], data['y'], data['playerName'])):
        if i < len(texts):
            texts[i].set_position((x + 1, y))
            texts[i].set_text(player_name.split(" ")[-1])


# ============================================================================
# GRAPH VISUALIZATION WITH ATTENTION
# ============================================================================

def visualize_graph_on_pitch_AT(G, sample_timestamp=0, ball=True, pitch_length=105, 
                               pitch_width=68, attention=False, attention_target=None,
                               attention_weights=None, edges=False):
    """Visualize NetworkX graph on football pitch with attention mechanisms."""
    fig, ax = plt.subplots(figsize=(40, 20))
    
    # Create pitch
    pitch = Pitch(
        pitch_type="uefa", pitch_length=pitch_length, pitch_width=pitch_width, 
        axis=False, label=False
    )
    pitch.draw(ax=ax)

    # Extract node positions and colors
    node_positions = _extract_node_positions(G)
    node_colors = _determine_node_colors(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, node_positions, node_size=300, 
                          node_color=node_colors, alpha=0.6, ax=ax)
    
    # Draw velocity vectors
    _draw_velocity_vectors_on_graph(G, ax)
        
    # Draw attention weights if requested
    if attention:
        _draw_attention_weights(G, node_positions, attention_target, 
                              attention_weights, ax)

    # Draw edges if requested
    if edges:
        _draw_graph_edges(G, node_positions, ax)

    # Draw player labels
    _draw_node_labels(G, node_positions, ax)

    # Finalize plot
    plt.title(f'Graph Visualization at Frame {sample_timestamp}')
    plt.axis('off')
    plt.show()
    
    return fig, ax


def _extract_node_positions(G):
    """Extract node positions from graph features."""
    return {player_id: (data['features'][0], data['features'][1]) 
            for player_id, data in G.nodes(data=True)}


def _determine_node_colors(G):
    """Determine node colors based on player type and team."""
    return ['black' if player_id == 'ball' 
            else 'red' if data['features'][6] == 1 
            else 'blue' 
            for player_id, data in G.nodes(data=True)]


def _draw_velocity_vectors_on_graph(G, ax):
    """Draw velocity vectors for all players in the graph."""
    for node, node_data in G.nodes(data=True):
        if node == 'ball':
            continue
        
        positions = node_data['features'][0:2]
        velocities = node_data['features'][2:4]
        ax.quiver(positions[0], positions[1], 
                 velocities[0], velocities[1],
                 color='k', scale=50, width=0.003, alpha=0.7)


def _draw_attention_weights(G, node_positions, attention_target, attention_weights, ax):
    """Draw attention weight connections between players."""
    if not attention_target or not attention_weights:
        return
        
    attention_weights = attention_weights[:10]  # Limit to top 10
    target_position = node_positions.get(attention_target, (0, 0))
    
    for weight in attention_weights:
        source_player = weight['source_player']
        source_position = node_positions.get(source_player, (0, 0))
        attention_weight = weight['attention_weight']

        # Draw connection line
        ax.plot([source_position[0], target_position[0]], 
               [source_position[1], target_position[1]], 
               color='gray', linewidth=1, alpha=0.5)

        # Add weight annotation
        mid_x = (source_position[0] + target_position[0]) / 2
        mid_y = (source_position[1] + target_position[1]) / 2
        ax.text(mid_x, mid_y, f'{attention_weight:.2f}', 
               fontsize=8, ha='center', va='center', color='black')


def _draw_graph_edges(G, node_positions, ax):
    """Draw graph edges with team-based coloring."""
    edges = G.edges(data=True)
    edge_colors = _determine_edge_colors(G, edges)

    # Draw edges
    nx.draw_networkx_edges(G, node_positions, edge_color=edge_colors, ax=ax)
    
    # Add edge labels
    edge_labels = {(u, v): "E" for u, v, data in edges}
    nx.draw_networkx_edge_labels(G, node_positions, edge_labels=edge_labels, 
                                font_color='black', font_size=8, ax=ax)


def _determine_edge_colors(G, edges):
    """Determine edge colors based on team relationships."""
    return ['green' if G.nodes[u]['features'][6] == G.nodes[v]['features'][6] 
            else 'red' for u, v, data in edges]


def _draw_node_labels(G, node_positions, ax):
    """Draw player ID labels on nodes."""
    labels = {player_id: player_id for player_id in G.nodes()}
    nx.draw_networkx_labels(G, node_positions, labels=labels, 
                           font_size=7, ax=ax)
