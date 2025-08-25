from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import pandas as pd

class GraphFeatureScaler:
    def __init__(self):
        # Separate scalers for different feature types
        self.position_scaler = StandardScaler()
        self.velocity_scaler = StandardScaler()
        self.acceleration_scaler = StandardScaler()
        self.goal_features_scaler = StandardScaler()
        self.ball_features_scaler = StandardScaler()
        self.team_features_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, graphs):
        """Fit scalers on a list of graphs"""
        node_positions = []
        node_velocities = []
        node_accelerations = []
        node_goal_features = []
        node_ball_features = []
        node_team_features = []
        edge_features = []

        for G in graphs:
            # Collect node features
            for node in G.nodes():
                features = G.nodes[node]['features']
                node_positions.append(features[0:2])  # x, y
                node_velocities.append(features[2:4])  # velocity_x, velocity_y
                node_accelerations.append(features[4:6])  # acceleration_x, acceleration_y
                node_goal_features.append(features[9:13])  # goal features
                node_ball_features.append(features[13:15]) #ball_dist,ball_angle
                #node_team_features.append(features[13:]) #Team Features

            # Collect edge features
            for _, _, edge_data in G.edges(data=True):
                edge_features.append(edge_data['features'])

        # Fit scalers
        self.position_scaler.fit(node_positions)
        self.velocity_scaler.fit(node_velocities)
        self.acceleration_scaler.fit(node_accelerations)
        self.goal_features_scaler.fit(node_goal_features)
        self.ball_features_scaler.fit(node_ball_features)
        #self.team_features_scaler.fit(node_team_features)
        self.edge_scaler.fit(edge_features)

        self.is_fitted = True

    def transform_graph(self, G):
        """Transform a single graph's features"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transforming")

        G_scaled = G.copy()

        # Transform node features
        for node in G_scaled.nodes():
            features = G_scaled.nodes[node]['features']

            # Scale different components separately
            pos_scaled = self.position_scaler.transform([features[0:2]])[0]
            vel_scaled = self.velocity_scaler.transform([features[2:4]])[0]
            acc_scaled = self.acceleration_scaler.transform([features[4:6]])[0]
            goal_features_scaled = self.goal_features_scaler.transform([features[9:13]])[0]
            ball_features_scaled = self.ball_features_scaler.transform([features[13:]])[0]
            #team_features_scaled = self.team_features_scaler.transform([features[13:]])[0]

            # Binary features don't need scaling (is_home, is_attacking, has_ball, position_one_hot)
            binary_features = features[6:9]
            #position_one_hot = features[15:]

            # Combine all features
            scaled_features = np.concatenate([
                pos_scaled,
                vel_scaled,
                acc_scaled,
                binary_features,
                goal_features_scaled,
                ball_features_scaled,
                #team_features_scaled
            ])

            G_scaled.nodes[node]['features'] = scaled_features

        # Transform edge features
        for edge in G_scaled.edges():
            edge_features = G_scaled.edges[edge]['features']
            scaled_edge_features = self.edge_scaler.transform([edge_features])[0]
            G_scaled.edges[edge]['features'] = scaled_edge_features

        return G_scaled