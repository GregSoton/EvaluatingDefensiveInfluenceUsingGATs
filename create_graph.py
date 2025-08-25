"""
Football Graph
=====================================

This module creates directed graphs from football tracking data with normalized coordinates
and calculates various features for players, ball, and their relationships.
"""

import networkx as nx
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np
import os
import math

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

# Set working directory
os.chdir('Data')
rosters = pd.read_csv('rosters_updated.csv')

# Create positions dictionary for one-hot encoding
all_positions = sorted(rosters['positionGroupType'].unique())
position_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}

# =============================================================================
# TEAM-LEVEL FEATURE CALCULATION
# =============================================================================

def calculate_team_features(df_time, team_is_home):
    """
    Calculate team-level spatial features for a specific team.
    
    Args:
        df_time (pd.DataFrame): Player tracking data for current frame
        team_is_home (bool): Whether to calculate features for home team
        
    Returns:
        dict: Dictionary containing team spatial features
    """
    # Filter for team players
    team_df = df_time[df_time['is_home'] == team_is_home]

    # Calculate team centroid
    centroid_x = team_df['x'].mean()
    centroid_y = team_df['y'].mean()

    # Calculate spatial distribution (standard deviation of positions)
    spread_x = team_df['x'].std()
    spread_y = team_df['y'].std()

    # Find furthest back player (smallest x value since attacking left to right)
    furthest_back_x = team_df['x'].min()

    # Calculate team compactness (average distance from centroid)
    distances_to_centroid = np.sqrt(
        (team_df['x'] - centroid_x)**2 + 
        (team_df['y'] - centroid_y)**2
    ).mean()

    return {
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'spread_x': spread_x,
        'spread_y': spread_y,
        'furthest_back_x': furthest_back_x,
        'team_compactness': distances_to_centroid
    }

# =============================================================================
# MAIN GRAPH CREATION FUNCTION
# =============================================================================

def create_normalized_graph_directed(players_df, balls_df, events_df, frameNum, 
                                   home_team_name, pitch_width=105, pitch_height=68, 
                                   proximity_threshold=10):
    """
    Create a directed graph with attacking team always moving left to right.
    
    Args:
        players_df (pd.DataFrame): Player tracking data
        balls_df (pd.DataFrame): Ball tracking data
        events_df (pd.DataFrame): Event data
        frameNum (int): Frame number to analyze
        home_team_name (str): Name of home team
        pitch_width (float): Width of pitch in meters
        pitch_height (float): Height of pitch in meters
        proximity_threshold (float): Distance threshold for proximity calculations
        
    Returns:
        nx.DiGraph or None: Directed graph with player and ball nodes, or None if invalid
    """
    # Filter data for the specific timestamp
    df_time = players_df[players_df['frameNum'] == frameNum].copy()
    balls_df_time = balls_df[balls_df['frameNum'] == frameNum].copy()
    
    # Validate ball data
    if (len(balls_df_time) == 0) or (balls_df_time['x'].isna().any()):
        return None

    # Extract ball possession and position information
    ball_info = _extract_ball_information(events_df, balls_df_time, frameNum, home_team_name)
    
    # Normalize coordinates if needed
    df_time, balls_df_time = _normalize_coordinates(
        df_time, balls_df_time, ball_info, pitch_width, pitch_height
    )

    # Check for upcoming events
    team_shot, player_shots, player_receptions = check_for_events(
        events_df, frameNum, ball_info['ball_team']
    )

    # Calculate team features
    home_team_features = calculate_team_features(df_time, team_is_home=True)
    away_team_features = calculate_team_features(df_time, team_is_home=False)

    # Initialize directed graph
    G = nx.DiGraph()
    
    # Add player nodes
    _add_player_nodes(
        G, df_time, ball_info, home_team_features, away_team_features, 
        player_shots, player_receptions
    )
    
    # Add ball node
    _add_ball_node(G, balls_df_time, ball_info, home_team_features, 
                   away_team_features, team_shot)
    
    # Add directed edges
    _add_directed_edges(G, df_time, ball_info['ball_pos'])

    return G

# =============================================================================
# HELPER FUNCTIONS FOR GRAPH CREATION
# =============================================================================

def _extract_ball_information(events_df, balls_df_time, frameNum, home_team_name):
    """Extract ball possession and position information."""
    # Get ball possession info
    ball_team = events_df.loc[
        events_df['frameNum'] == events_df[events_df['frameNum'] >= frameNum].frameNum.min(), 
        'team_name'
    ].values[0]
    
    ball_player = events_df.loc[
        events_df['frameNum'] == events_df[events_df['frameNum'] >= frameNum].frameNum.min(), 
        'player_name'
    ].values[0]
    
    ball_home = int(home_team_name == ball_team)
    ball_pos = np.array([balls_df_time['x'].values[0], balls_df_time['y'].values[0]])
    ball_veloc = np.array([balls_df_time['velocity_x'].values[0], balls_df_time['velocity_y'].values[0]])
    ball_accel = np.array([balls_df_time['acceleration_x'].values[0], balls_df_time['acceleration_y'].values[0]])
    home_team_left = balls_df_time['homeTeamLeft'].values[0]
    
    return {
        'ball_team': ball_team,
        'ball_player': ball_player,
        'ball_home': ball_home,
        'ball_pos': ball_pos,
        'ball_veloc': ball_veloc,
        'ball_accel': ball_accel,
        'home_team_left': home_team_left
    }

def _normalize_coordinates(df_time, balls_df_time, ball_info, pitch_width, pitch_height):
    """Normalize coordinates so attacking team always moves left to right."""
    # Determine if we need to flip coordinates
    need_to_flip = (
        (ball_info['ball_home'] and not ball_info['home_team_left']) or 
        (not ball_info['ball_home'] and ball_info['home_team_left'])
    )

    if need_to_flip:
        # Flip coordinates and velocities for players
        df_time['x'] = pitch_width - df_time['x']
        df_time['velocity_x'] = -df_time['velocity_x']
        df_time['acceleration_x'] = -df_time['acceleration_x']
        df_time['y'] = pitch_height - df_time['y']
        df_time['velocity_y'] = -df_time['velocity_y']
        df_time['acceleration_y'] = -df_time['acceleration_y']

        # Flip coordinates and velocities for ball
        balls_df_time['x'] = pitch_width - balls_df_time['x']
        balls_df_time['velocity_x'] = -balls_df_time['velocity_x']
        balls_df_time['acceleration_x'] = -balls_df_time['acceleration_x']
        balls_df_time['y'] = pitch_height - balls_df_time['y']
        balls_df_time['velocity_y'] = -balls_df_time['velocity_y']
        balls_df_time['acceleration_y'] = -balls_df_time['acceleration_y']
        
        # Update ball position in ball_info
        ball_info['ball_pos'] = np.array([balls_df_time['x'].values[0], balls_df_time['y'].values[0]])
        ball_info['ball_veloc'] = np.array([balls_df_time['velocity_x'].values[0], balls_df_time['velocity_y'].values[0]])
        ball_info['ball_accel'] = np.array([balls_df_time['acceleration_x'].values[0], balls_df_time['acceleration_y'].values[0]])

    return df_time, balls_df_time

def _add_player_nodes(G, df_time, ball_info, home_team_features, away_team_features, 
                     player_shots, player_receptions):
    """Add player nodes to the graph with feature vectors."""
    for idx, row in df_time.iterrows():
        player_pos = np.array([row['x'], row['y']])
        player_name = row['playerName']
        is_home = row['is_home']
        team_features = home_team_features if is_home else away_team_features
        
        # Calculate player-specific features
        is_attacking = (row['is_home'] and ball_info['ball_home']) or (not row['is_home'] and not ball_info['ball_home'])
        goal_features = calculate_normalized_goal_features(player_pos, is_attacking)
        
        # Calculate ball-related features
        ball_distance_features = _calculate_ball_distance_features(player_pos, ball_info['ball_pos'])
        
        # Create position one-hot encoding
        position_one_hot = _create_position_encoding(row['playerPos'])
        
        # Calculate team-relative features
        team_relative_features = _calculate_team_relative_features(player_pos, team_features)

        # Create feature vector
        feature_vector = _create_player_feature_vector(
            row, is_attacking, ball_info['ball_player'], goal_features, 
            ball_distance_features, df_time
        )

        # Add targets for attacking players
        player_shot = player_shots.get(player_name, 0) if is_attacking else 0
        player_reception = player_receptions.get(player_name, 0) if is_attacking else 0

        G.add_node(player_name, features=np.array(feature_vector),
                   shot_target=player_shot, reception_target=player_reception)

def _add_ball_node(G, balls_df_time, ball_info, home_team_features, away_team_features, team_shot):
    """Add ball node to the graph."""
    ball_pos = np.array([balls_df_time['x'].values[0], balls_df_time['y'].values[0]])
    ball_veloc = np.array([balls_df_time['velocity_x'].values[0], balls_df_time['velocity_y'].values[0]])
    ball_accel = np.array([balls_df_time['acceleration_x'].values[0], balls_df_time['acceleration_y'].values[0]])

    # Calculate ball's goal features (always attacking right goal)
    ball_goal_features = calculate_normalized_goal_features(ball_pos, True)
    
    ball_feature_vector = [
        ball_pos[0], ball_pos[1], ball_veloc[0], ball_veloc[1], 
        ball_accel[0], ball_accel[1], ball_info['ball_home'], 1, 1, 
        *ball_goal_features, 0, 0
    ]
    
    G.add_node('ball', features=np.array(ball_feature_vector), team_shot_target=team_shot)

def _add_directed_edges(G, df_time, ball_pos):
    """Add directed edges between all nodes."""
    for i, player1 in df_time.iterrows():
        player1_pos = np.array([player1['x'], player1['y']])
        player1_vel = np.array([player1['velocity_x'], player1['velocity_y']])

        # Directed edges to other players
        for j, player2 in df_time.iterrows():
            if player1['playerName'] != player2['playerName']:
                player2_pos = np.array([player2['x'], player2['y']])
                player2_vel = np.array([player2['velocity_x'], player2['velocity_y']])
                edge_features = calculate_normalized_edge_features(
                    player1_pos, player2_pos, player1_vel, player2_vel, 
                    player1['is_home'], player2['is_home'], ball_pos
                )
                G.add_edge(player1['playerName'], player2['playerName'], 
                          features=np.array(list(edge_features.values())))

        # Directed edge to ball
        ball_vel = np.array([0, 0])  # Ball velocity handled separately
        edge_features = calculate_normalized_edge_features(
            player1_pos, ball_pos, player1_vel, ball_vel, 
            player1['is_home'], True, ball_pos  # Ball treated as neutral
        )
        G.add_edge(player1['playerName'], 'ball', 
                  features=np.array(list(edge_features.values())))

        # Directed edge from ball to player
        edge_features = calculate_normalized_edge_features(
            ball_pos, player1_pos, ball_vel, player1_vel, 
            True, player1['is_home'], ball_pos
        )
        G.add_edge('ball', player1['playerName'], 
                  features=np.array(list(edge_features.values())))

# =============================================================================
# FEATURE CALCULATION HELPER FUNCTIONS
# =============================================================================

def _calculate_ball_distance_features(player_pos, ball_pos):
    """Calculate distance and angle features relative to ball."""
    ball_dx = ball_pos[0] - player_pos[0]  
    ball_dy = ball_pos[1] - player_pos[1]  
    distance_to_ball = np.sqrt(ball_dx**2 + ball_dy**2)  
    angle_to_ball = np.arctan2(ball_dy, ball_dx)  
    
    return {
        'distance_to_ball': distance_to_ball,
        'angle_to_ball': angle_to_ball
    }

def _create_position_encoding(player_position):
    """Create one-hot encoding for player position."""
    position_one_hot = np.zeros(len(all_positions))
    if pd.notna(player_position):
        position_one_hot[position_to_idx[player_position]] = 1
    return position_one_hot

def _calculate_team_relative_features(player_pos, team_features):
    """Calculate features relative to team formation."""
    centroid_pos = np.array([team_features['centroid_x'], team_features['centroid_y']])
    distance_to_centroid = np.linalg.norm(player_pos - centroid_pos)
    relative_to_back = player_pos[0] - team_features['furthest_back_x']
    
    return {
        'distance_to_centroid': distance_to_centroid,
        'relative_to_back': relative_to_back
    }

def _create_player_feature_vector(row, is_attacking, ball_player, goal_features, 
                                ball_distance_features, df_time):
    """Create comprehensive feature vector for a player."""
    feature_vector = [
        row['x'],
        row['y'],
        row['velocity_x'],
        row['velocity_y'],
        row['acceleration_x'],
        row['acceleration_y'],
        int(row['is_home']),
        int(is_attacking),
        int(row['playerName'] == ball_player),
        *goal_features,
        ball_distance_features['distance_to_ball'],
        ball_distance_features['angle_to_ball']
    ]

    # Add role features if they exist
    role_columns = [col for col in df_time.columns if 'role_' in col]
    feature_vector.extend(row[role_columns])

    return feature_vector

# =============================================================================
# GOAL AND SPATIAL FEATURE CALCULATIONS
# =============================================================================

def calculate_goal_features(position, is_home, home_team_left):
    """
    Calculate x and y distances to both goals for a given position.

    Args:
        position (np.array): [x, y] position
        is_home (bool): whether the position belongs to home team
        home_team_left (bool): whether home team is playing left to right

    Returns:
        tuple: (defending_goal_dx, defending_goal_dy, attacking_goal_dx, attacking_goal_dy)
    """
    # Define goal positions based on home_team_left
    if home_team_left:
        home_goal = np.array([0, 34])  # Left goal
        away_goal = np.array([105, 34])  # Right goal
    else:
        home_goal = np.array([105, 34])  # Right goal
        away_goal = np.array([0, 34])  # Left goal

    # Determine defending and attacking goals based on team
    defending_goal = home_goal if is_home else away_goal
    attacking_goal = away_goal if is_home else home_goal

    # Calculate x and y distances (position - goal)
    defending_goal_dx = position[0] - defending_goal[0]
    defending_goal_dy = position[1] - defending_goal[1]
    attacking_goal_dx = position[0] - attacking_goal[0]
    attacking_goal_dy = position[1] - attacking_goal[1]

    return (defending_goal_dx, defending_goal_dy, attacking_goal_dx, attacking_goal_dy)

def calculate_normalized_goal_features(pos, is_attacking):
    """
    Calculate goal-related features in normalized coordinates (attacking right).
    
    Args:
        pos (np.array): Player position [x, y]
        is_attacking (bool): Whether player is on attacking team
        
    Returns:
        list: [x_to_defending, y_to_defending, x_to_attacking, y_to_attacking]
    """
    # Goals are always at (0, 34) and (105, 34) in normalized coordinates
    attacking_goal = np.array([105, 34])
    defending_goal = np.array([0, 34])
    
    # Calculate distances to goals
    x_to_attacking = attacking_goal[0] - pos[0]
    y_to_attacking = attacking_goal[1] - pos[1]
    x_to_defending = defending_goal[0] - pos[0]
    y_to_defending = defending_goal[1] - pos[1]

    return [x_to_defending, y_to_defending, x_to_attacking, y_to_attacking]

# =============================================================================
# EDGE FEATURE CALCULATIONS
# =============================================================================

def calculate_edge_features(pos1, pos2, team1, team2):
    """
    Calculate basic features for an edge between two nodes.
    
    Args:
        pos1, pos2 (np.array): Positions of the two nodes
        team1, team2 (bool): Team affiliations of the two nodes
        
    Returns:
        dict: Dictionary of edge features
    """
    return {
        'dx': pos2[0] - pos1[0],
        'dy': pos2[1] - pos1[1],
        'same_team': int(team1 == team2)
    }

def calculate_normalized_edge_features(pos1, pos2, vel1, vel2, is_home1, is_home2, ball_pos):
    """
    Calculate comprehensive edge features in normalized coordinates.
    
    Args:
        pos1, pos2 (np.array): Positions of the two nodes
        vel1, vel2 (np.array): Velocities of the two nodes
        is_home1, is_home2 (bool): Team affiliations
        ball_pos (np.array): Ball position
        
    Returns:
        dict: Dictionary of edge features
    """
    # Calculate relative positions
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    rdx = pos1[0] - pos2[0]
    rdy = pos1[1] - pos2[1]
    
    # Calculate distance and angle
    euclidean_distance = math.sqrt(dx**2 + dy**2)
    angle = math.atan2(rdy, rdx)
    
    # Calculate relative velocity
    rel_vel_x = vel1[0] - vel2[0]
    rel_vel_y = vel1[1] - vel2[1]
    relative_speed = math.sqrt(rel_vel_x**2 + rel_vel_y**2)
    
    # Determine if nodes are on same team
    same_team = int(is_home1 == is_home2)
    
    # Calculate ball-related features for both nodes
    ball_features_1 = _calculate_node_ball_features(pos1, ball_pos)
    ball_features_2 = _calculate_node_ball_features(pos2, ball_pos)
    
    # Calculate differences in ball-related features
    diff_distance_to_ball = ball_features_1['distance'] - ball_features_2['distance']
    diff_angle_to_ball = ball_features_1['angle'] - ball_features_2['angle']

    return {
        'dx': dx,
        'dy': dy,
        'euclidean_distance': euclidean_distance,
        'angle': angle,
        'same_team': same_team,
        'diff_angle_to_ball': diff_angle_to_ball
    }

def _calculate_node_ball_features(pos, ball_pos):
    """Calculate ball-related features for a single node."""
    ball_dx = ball_pos[0] - pos[0]  
    ball_dy = ball_pos[1] - pos[1]  
    distance_to_ball = math.sqrt(ball_dx**2 + ball_dy**2)  
    angle_to_ball = math.atan2(ball_dy, ball_dx)  
    
    return {
        'distance': distance_to_ball,
        'angle': angle_to_ball
    }

# =============================================================================
# EVENT PREDICTION FUNCTIONS
# =============================================================================

def check_for_events(events_df, current_frame, ball_team, frames_ahead=150):
    """
    Check for shots and ball receptions in the next time window.
    Only considers events by the team currently on the ball.
    
    Args:
        events_df (pd.DataFrame): Event data
        current_frame (int): Current frame number
        ball_team (str): Team currently in possession
        frames_ahead (int): Number of frames to look ahead (default: 150 = 5 seconds at 30fps)
    
    Returns:
        tuple: (team_shot, dict of player_shots, dict of player_receptions)
    """
    end_frame = current_frame + frames_ahead
    
    # Get future events for the team in possession
    future_events = events_df[
        (events_df['frameNum'] > current_frame) & 
        (events_df['team_name'] == ball_team)
    ][:1]
    
    future_shot_events = events_df[
        (events_df['frameNum'] > current_frame) & 
        (events_df['team_name'] == ball_team)
    ][:5]

    # Initialize return values
    team_shot = 0
    player_shots = {}
    player_receptions = {}

    if not future_events.empty:
        # Check for shots
        shot_events = future_shot_events[future_shot_events['possessionEventType'] == 'SH']
        if not shot_events.empty:
            first_shot = shot_events.iloc[0]
            team_shot = 1
            player_shots[first_shot['player_name']] = 1

        # Group by player and get first event for each (receptions)
        first_events = future_events.groupby('player_name').first()
        for player in first_events.index:
            player_receptions[player] = 1
  
    return team_shot, player_shots, player_receptions
