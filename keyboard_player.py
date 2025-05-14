from command import Command
from buttons import Buttons
import csv
from pynput import keyboard
import os
import threading

class KeyboardPlayer:
    # Class-level flag to track if CSV header has been written
    # Class-level frame counter
    _frame_id = 0
    # Number of frames to skip between logs (log every 10th frame)

    def __init__(self):
        self.my_command = Command()
        self.buttons = Buttons()
        self.prev_opponent_health = None
        # Dictionary to track current key states
        self.key_states = {
            'up': False, 'down': False, 'left': False, 'right': False,
            'a': False, 's': False, 'd': False, 'w': False,
            'e': False, 'q': False, 'enter': False, 'space': False
        }
        # Start keyboard listener in a separate thread
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        """Handle key press events."""
        try:
            # Map keyboard keys to game buttons
            if key == keyboard.Key.up:
                self.key_states['up'] = True
            elif key == keyboard.Key.down:
                self.key_states['down'] = True
            elif key == keyboard.Key.left:
                self.key_states['left'] = True
            elif key == keyboard.Key.right:
                self.key_states['right'] = True
            elif key == keyboard.KeyCode.from_char('a'):
                self.key_states['a'] = True
            elif key == keyboard.KeyCode.from_char('s'):
                self.key_states['s'] = True
            elif key == keyboard.KeyCode.from_char('d'):
                self.key_states['d'] = True
            elif key == keyboard.KeyCode.from_char('w'):
                self.key_states['w'] = True
            elif key == keyboard.KeyCode.from_char('e'):
                self.key_states['e'] = True
            elif key == keyboard.KeyCode.from_char('q'):
                self.key_states['q'] = True
            elif key == keyboard.Key.enter:
                self.key_states['enter'] = True
            elif key == keyboard.Key.space:
                self.key_states['space'] = True
        except AttributeError:
            pass

    def on_release(self, key):
        """Handle key release events."""
        try:
            if key == keyboard.Key.up:
                self.key_states['up'] = False
            elif key == keyboard.Key.down:
                self.key_states['down'] = False
            elif key == keyboard.Key.left:
                self.key_states['left'] = False
            elif key == keyboard.Key.right:
                self.key_states['right'] = False
            elif key == keyboard.KeyCode.from_char('a'):
                self.key_states['a'] = False
            elif key == keyboard.KeyCode.from_char('s'):
                self.key_states['s'] = False
            elif key == keyboard.KeyCode.from_char('d'):
                self.key_states['d'] = False
            elif key == keyboard.KeyCode.from_char('w'):
                self.key_states['w'] = False
            elif key == keyboard.KeyCode.from_char('e'):
                self.key_states['e'] = False
            elif key == keyboard.KeyCode.from_char('q'):
                self.key_states['q'] = False
            elif key == keyboard.Key.enter:
                self.key_states['enter'] = False
            elif key == keyboard.Key.space:
                self.key_states['space'] = False
        except AttributeError:
            pass

    def log_data(self, game_state, command, player_id):
        """Log game state, actions, and outcomes to a CSV file."""
        # Calculate damage dealt
        current_opponent_health = game_state.player2.health if player_id == "1" else game_state.player1.health
        damage_dealt = 0
        if self.prev_opponent_health is not None:
            damage_dealt = max(0, self.prev_opponent_health - current_opponent_health)
        self.prev_opponent_health = current_opponent_health

        # Prepare data row
        buttons_p1 = command.player_buttons if player_id == "1" else game_state.player1.player_buttons
        buttons_p2 = command.player2_buttons if player_id == "2" else game_state.player2.player_buttons

        row = [
            KeyboardPlayer._frame_id,
            game_state.player1.player_id,
            game_state.player1.health,
            game_state.player1.x_coord,
            game_state.player1.y_coord,
            game_state.player1.is_jumping,
            game_state.player1.is_crouching,
            game_state.player1.is_player_in_move,
            game_state.player1.move_id,
            game_state.player2.player_id,
            game_state.player2.health,
            game_state.player2.x_coord,
            game_state.player2.y_coord,
            game_state.player2.is_jumping,
            game_state.player2.is_crouching,
            game_state.player2.is_player_in_move,
            game_state.player2.move_id,
            game_state.player2.x_coord - game_state.player1.x_coord,  # Distance
            game_state.timer,
            game_state.has_round_started,
            game_state.is_round_over,
            game_state.fight_result or "ongoing",  # Use "ongoing" if no result yet
            buttons_p1.up,
            buttons_p1.down,
            buttons_p1.left,
            buttons_p1.right,
            buttons_p1.A,
            buttons_p1.B,
            buttons_p1.X,
            buttons_p1.Y,
            buttons_p1.L,
            buttons_p1.R,
            buttons_p1.select,
            buttons_p1.start,
            buttons_p2.up,
            buttons_p2.down,
            buttons_p2.left,
            buttons_p2.right,
            buttons_p2.A,
            buttons_p2.B,
            buttons_p2.X,
            buttons_p2.Y,
            buttons_p2.L,
            buttons_p2.R,
            buttons_p2.select,
            buttons_p2.start,
            damage_dealt
        ]

        # Define CSV file path
        csv_file = "game_log.csv"
        is_empty = os.stat(csv_file).st_size == 0
        # Write header if not already written
        if is_empty:
            header = [
                "frame_id",
                "p1_character", "p1_health", "p1_x", "p1_y", "p1_jumping", "p1_crouching", "p1_in_move", "p1_move_id",
                "p2_character", "p2_health", "p2_x", "p2_y", "p2_jumping", "p2_crouching", "p2_in_move", "p2_move_id",
                "distance", "timer", "round_started", "round_over", "fight_result",
                "p1_up", "p1_down", "p1_left", "p1_right", "p1_A", "p1_B", "p1_X", "p1_Y", "p1_L", "p1_R", "p1_select", "p1_start",
                "p2_up", "p2_down", "p2_left", "p2_right", "p2_A", "p2_B", "p2_X", "p2_Y", "p2_L", "p2_R", "p2_select", "p2_start",
                "damage_dealt"
            ]
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
        else:
            # Append to existing file
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

    def play(self, current_game_state, player):
        """Generate command based on keyboard inputs."""
        # Increment frame ID
        KeyboardPlayer._frame_id += 1

        # Initialize previous health if not set
        if self.prev_opponent_health is None:
            self.prev_opponent_health = current_game_state.player2.health if player == "1" else current_game_state.player1.health

        # Map key states to button states
        self.buttons.up = self.key_states['up']
        self.buttons.down = self.key_states['down']
        self.buttons.left = self.key_states['left']
        self.buttons.right = self.key_states['right']
        self.buttons.A = self.key_states['a']
        self.buttons.B = self.key_states['s']
        self.buttons.X = self.key_states['d'] 
        self.buttons.Y = self.key_states['w']
        self.buttons.L = self.key_states['e']
        self.buttons.R = self.key_states['q']
        self.buttons.select = self.key_states['enter']
        self.buttons.start = self.key_states['space']

        # Update command based on player
        if player == "1":
            self.my_command.player_buttons = self.buttons
        elif player == "2":
            self.my_command.player2_buttons = self.buttons

        # Log every _log_interval frames or on the first frame
        self.log_data(current_game_state, self.my_command, player)

        return self.my_command

    def stop(self):
        """Stop the keyboard listener."""
        self.listener.stop()