from command import Command
import numpy as np
from buttons import Buttons
import csv
import os
import torch
import torch.nn as nn
import joblib
import pandas as pd

# Define the MLP model class (same as training)
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class Bot:
    _frame_id = 0

    def __init__(self):
        # Load the trained model and scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_dim=15, output_dim=10).to(self.device)  # 15 features, 10 labels
        self.model.load_state_dict(torch.load('mlp_street_fighter_best.pth'))
        self.model.eval()
        self.scaler = joblib.load('scaler.pkl')

        # Define feature and label columns (must match training)
        self.feature_cols = ['p1_x', 'p1_y', 'p1_health', 'p1_jumping', 'p1_crouching', 'p1_in_move', 'p1_move_id',
                            'p2_x', 'p2_y', 'p2_health', 'p2_jumping', 'p2_crouching', 'p2_in_move', 'p2_move_id', 'timer']
        self.label_cols = ['up', 'down', 'left', 'right', 'Y', 'B', 'X', 'A', 'L', 'R']
        self.bool_cols = ['p1_jumping', 'p1_crouching', 'p1_in_move', 'p2_jumping', 'p2_crouching', 'p2_in_move']

        # Initialize command and buttons
        self.my_command = Command()
        self.buttn = Buttons()
        self.prev_opponent_health = None

    def log_data(self, game_state, command, player_id):
        Bot._frame_id += 1

        # Get Player 1's button inputs (bot or opponent, depending on player_id)
        buttons_p1 = command.player_buttons if player_id == "1" else command.player2_buttons

        row = [
            game_state.player1.player_id,
            game_state.player1.x_coord,
            game_state.player1.y_coord,
            game_state.player1.health,
            game_state.player1.is_jumping,
            game_state.player1.is_crouching,
            game_state.player1.is_player_in_move,
            game_state.player1.move_id,
            game_state.player2.x_coord,
            game_state.player2.y_coord,
            game_state.player2.health,
            game_state.player2.is_jumping,
            game_state.player2.is_crouching,
            game_state.player2.is_player_in_move,
            game_state.player2.move_id,
            game_state.timer,
            buttons_p1.up,
            buttons_p1.down,
            buttons_p1.left,
            buttons_p1.right,
            buttons_p1.Y,
            buttons_p1.B,
            buttons_p1.X,
            buttons_p1.A,
            buttons_p1.L,
            buttons_p1.R
        ]

        csv_file = "game_log.csv"
        write_header = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0
        try:
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    header = [
                        "player_id",
                        "p1_x", "p1_y", "p1_health", "p1_jumping", "p1_crouching", "p1_in_move", "p1_move_id",
                        "p2_x", "p2_y", "p2_health", "p2_jumping", "p2_crouching", "p2_in_move", "p2_move_id",
                        "timer",
                        "up", "down", "left", "right", "Y", "B", "X", "A", "L", "R"
                    ]
                    writer.writerow(header)
                writer.writerow(row)
        except IOError as e:
            print(f"Error writing to CSV: {e}")

    def preprocess_state(self, game_state, player):
        # Extract features based on player perspective
        if player == "1":
            state_data = {
                'p1_x': game_state.player1.x_coord,
                'p1_y': game_state.player1.y_coord,
                'p1_health': game_state.player1.health,
                'p1_jumping': int(game_state.player1.is_jumping),
                'p1_crouching': int(game_state.player1.is_crouching),
                'p1_in_move': int(game_state.player1.is_player_in_move),
                'p1_move_id': game_state.player1.move_id,
                'p2_x': game_state.player2.x_coord,
                'p2_y': game_state.player2.y_coord,
                'p2_health': game_state.player2.health,
                'p2_jumping': int(game_state.player2.is_jumping),
                'p2_crouching': int(game_state.player2.is_crouching),
                'p2_in_move': int(game_state.player2.is_player_in_move),
                'p2_move_id': game_state.player2.move_id,
                'timer': game_state.timer
            }
        else:  # player == "2"
            state_data = {
                'p1_x': game_state.player2.x_coord,
                'p1_y': game_state.player2.y_coord,
                'p1_health': game_state.player2.health,
                'p1_jumping': int(game_state.player2.is_jumping),
                'p1_crouching': int(game_state.player2.is_crouching),
                'p1_in_move': int(game_state.player2.is_player_in_move),
                'p1_move_id': game_state.player2.move_id,
                'p2_x': game_state.player1.x_coord,
                'p2_y': game_state.player1.y_coord,
                'p2_health': game_state.player1.health,
                'p2_jumping': int(game_state.player1.is_jumping),
                'p2_crouching': int(game_state.player1.is_crouching),
                'p2_in_move': int(game_state.player1.is_player_in_move),
                'p2_move_id': game_state.player1.move_id,
                'timer': game_state.timer
            }

        # Convert to DataFrame
        df_state = pd.DataFrame([state_data])

        # Handle missing values
        if df_state.isnull().sum().any():
            print("Warning: Missing values detected in game state. Filling with defaults.")
            for col in self.bool_cols:
                df_state[col] = df_state[col].fillna(0)  # Assume FALSE (0) for booleans
            numeric_cols = [col for col in self.feature_cols if col not in self.bool_cols]
            df_state[numeric_cols] = df_state[numeric_cols].fillna(df_state[numeric_cols].median())

        # Validate features
        for col in self.feature_cols:
            if col not in df_state.columns:
                raise ValueError(f"Missing feature: {col}")
            if not np.issubdtype(df_state[col].dtype, np.number):
                raise ValueError(f"Non-numeric value in {col}: {df_state[col].iloc[0]}")

        # Scale the features
        X = df_state[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Convert to tensor
        return torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

    def predict_actions(self, state_tensor):
        with torch.no_grad():
            outputs = self.model(state_tensor)
            # Apply threshold to get binary predictions
            predictions = (torch.sigmoid(outputs) >= 0.5).float().cpu().numpy()[0]
        return dict(zip(self.label_cols, predictions))

    def fight(self, current_game_state, player):
        if self.prev_opponent_health is None:
            self.prev_opponent_health = current_game_state.player2.health if player == "1" else current_game_state.player1.health

        # Reset buttons to avoid stale inputs
        self.buttn = Buttons()

        try:
            # Preprocess the current game state
            state_tensor = self.preprocess_state(current_game_state, player)

            # Predict actions using the trained model
            actions = self.predict_actions(state_tensor)

            # Set buttons based on model predictions
            self.buttn.up = bool(actions['up'])
            self.buttn.down = bool(actions['down'])
            self.buttn.left = bool(actions['left'])
            self.buttn.right = bool(actions['right'])
            self.buttn.Y = bool(actions['Y'])
            self.buttn.B = bool(actions['B'])
            self.buttn.X = bool(actions['X'])
            self.buttn.A = bool(actions['A'])
            self.buttn.L = bool(actions['L'])
            self.buttn.R = bool(actions['R'])

        except Exception as e:
            print(f"Error in prediction: {e}")
            # Fallback to no action if preprocessing fails
            pass

        # Assign buttons to command
        if player == "1":
            self.my_command.player_buttons = self.buttn
        else:
            self.my_command.player2_buttons = self.buttn

        # Log the current state
        self.log_data(current_game_state, self.my_command, player)

        return self.my_command

    def run_command(self, com, player):
        # This method is no longer needed for model-based play, but kept for compatibility
        pass