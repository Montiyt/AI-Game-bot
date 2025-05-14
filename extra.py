

import socket
import json
from game_state import GameState
#from bot import fight
import sys
from bot import Bot
def connect(port):
    #For making a connection with the game
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", port))
    server_socket.listen(5)
    (client_socket, _) = server_socket.accept()
    print ("Connected to game!")
    return client_socket

def send(client_socket, command):
    #This function will send your updated command to Bizhawk so that game reacts according to your command.
    command_dict = command.object_to_dict()
    pay_load = json.dumps(command_dict).encode()
    client_socket.sendall(pay_load)

def receive(client_socket):
    #receive the game state and return game state
    pay_load = client_socket.recv(4096)
    input_dict = json.loads(pay_load.decode())
    game_state = GameState(input_dict)

    return game_state

def main():
    if (sys.argv[1]=='1'):
        client_socket = connect(9999)
    elif (sys.argv[1]=='2'):
        client_socket = connect(10000)
    current_game_state = None
    #print( current_game_state.is_round_over )
    bot=Bot()
    while (current_game_state is None) or (not current_game_state.is_round_over):

        current_game_state = receive(client_socket)
        bot_command = bot.fight(current_game_state,sys.argv[1])
        send(client_socket, bot_command)
if __name__ == '__main__':
   main()





######################################### The above code is a Python script that connects to a game server, receives game state updates, and sends commands back to the server.
#the below code is from keyboard input player
import socket
import json
from game_state import GameState
import sys
from keyboard_player import KeyboardPlayer
import logging

# Configure logging to file for debugging
logging.basicConfig(
    filename="controller.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def connect(port, max_attempts=8, timeout=10):
    """
    Attempt to connect to BizHawk with a timeout and retry mechanism.
    Returns client_socket if successful, None otherwise.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.settimeout(timeout)
            server_socket.bind(("127.0.0.1", port))
            server_socket.listen(5)
            logging.info(f"Listening on port {port}, attempt {attempt + 1}/{max_attempts}")
            print(f"Waiting for BizHawk connection on port {port}...")
            (client_socket, addr) = server_socket.accept()
            client_socket.settimeout(timeout)
            logging.info(f"Connected to BizHawk at {addr}")
            print("Connected to game!")
            return client_socket
        except socket.timeout:
            attempt += 1
            logging.warning(f"Connection timeout on port {port}, attempt {attempt}/{max_attempts}")
            print(f"Timeout waiting for BizHawk, retrying ({attempt}/{max_attempts})...")
        except socket.error as e:
            logging.error(f"Socket error on port {port}: {e}")
            print(f"Socket error: {e}")
            break
        finally:
            server_socket.close()
    logging.error(f"Failed to connect after {max_attempts} attempts on port {port}")
    print(f"Failed to connect to BizHawk after {max_attempts} attempts.")
    return None

def send(client_socket, command):
    """
    Send command to BizHawk.
    """
    try:
        command_dict = command.object_to_dict()
        pay_load = json.dumps(command_dict).encode()
        client_socket.sendall(pay_load)
        logging.info("Sent command to BizHawk")
    except socket.error as e:
        logging.error(f"Failed to send command: {e}")
        raise

def receive(client_socket):
    """
    Receive game state from BizHawk with timeout.
    """
    try:
        pay_load = client_socket.recv(4096)
        if not pay_load:
            logging.warning("Received empty payload from BizHawk")
            return None
        input_dict = json.loads(pay_load.decode())
        game_state = GameState(input_dict)
        logging.info("Received game state from BizHawk")
        return game_state
    except socket.timeout:
        logging.warning("Timeout receiving game state")
        return None
    except (socket.error, json.JSONDecodeError) as e:
        logging.error(f"Error receiving game state: {e}")
        raise

def main():
    # Select port based on command-line argument
    if sys.argv[1] == '1':
        port = 9999
    elif sys.argv[1] == '2':
        port = 9090
    else:
        logging.error(f"Invalid player argument: {sys.argv[1]}")
        print("Please specify player '1' or '2' as argument")
        sys.exit(1)

    # Attempt to connect
    client_socket = connect(port)
    if client_socket is None:
        logging.critical("Exiting due to connection failure")
        print("Cannot proceed without connection to BizHawk. Exiting.")
        sys.exit(1)

    try:
        current_game_state = None
        player = KeyboardPlayer()

        while (current_game_state is None) or (not current_game_state.is_round_over):
            # Receive game state
            current_game_state = receive(client_socket)
            if current_game_state is None:
                logging.warning("No game state received, skipping iteration")
                continue

            

            # Get keyboard command and send it
            player_command = player.play(current_game_state, sys.argv[1])
            send(client_socket, player_command)
            # Check if round is over
            if current_game_state.is_round_over:
                logging.info("Round is over, exiting loop")
                break
    except Exception as e:
        logging.error(f"Main loop error: {e}")
        print(f"Error in main loop: {e}")
    finally:
        player.stop()
        client_socket.close()
        logging.info("Closed client socket")

if __name__ == '__main__':
    main()
    
    
    
    
    ######################################
    #rule based bot from command import Command
import numpy as np
from buttons import Buttons
import csv
import os

class Bot:
    _frame_id = 0

    def __init__(self):
        # Comprehensive move set (assuming Ryu-like character)
        self.moves = {
            # Long-range moves
            "hadoken": [">", "!>", "v", "v+>", "!v+!>", "Y", "!Y"],  # Fireball
            "long_kick": [">", "!>", "B", "!B"],  # Standing heavy kick
            # Mid-range moves
            "shoryuken": [">", "!>", "v", "v+>", "!v+!>", ">+Y", "!>+!Y"],  # Uppercut
            "medium_punch": ["Y", "!Y"],  # Standing medium punch
            # Close-range moves
            "crouch_kick": ["v", "B", "!v", "!B"],  # Crouching light kick
            "throw": [">", "Y", "!>", "!Y"],  # Basic throw
            # Defensive moves
            "block_high": ["<", "!<"],  # Standing block
            "block_low": ["v+<", "!v+!<"],  # Crouching block
            # Movement
            "jump_forward": ["^", ">+^", "!>+!^"],  # Jump forward
            "walk_forward": [">", "!>"],  # Walk forward
            "walk_backward": ["<", "!<"]  # Walk backward
        }
        self.exe_code = 0
        self.start_fire = True
        self.remaining_code = []
        self.my_command = Command()
        self.buttn = Buttons()
        self.prev_opponent_health = None

    def log_data(self, game_state, command, player_id):
        Bot._frame_id += 1

        # Get Player 1's button inputs (bot or opponent, depending on player_id)
        buttons_p1 = command.player_buttons if player_id == "1" else command.player2_buttons

        row = [
            game_state.player1.player_id,  # player_id
            game_state.player1.x_coord,    # p1_x
            game_state.player1.y_coord,    # p1_y
            game_state.player1.health,     # p1_health
            game_state.player1.is_jumping, # p1_jumping
            game_state.player1.is_crouching, # p1_crouching
            game_state.player1.is_player_in_move, # p1_in_move
            game_state.player1.move_id,    # p1_move_id
            game_state.player2.x_coord,    # p2_x
            game_state.player2.y_coord,    # p2_y
            game_state.player2.health,     # p2_health
            game_state.player2.is_jumping, # p2_jumping
            game_state.player2.is_crouching, # p2_crouching
            game_state.player2.is_player_in_move, # p2_in_move
            game_state.player2.move_id,    # p2_move_id
            game_state.timer,              # timer
            buttons_p1.up,                 # up
            buttons_p1.down,               # down
            buttons_p1.left,               # left
            buttons_p1.right,              # right
            buttons_p1.Y,                  # Y
            buttons_p1.B,                  # B
            buttons_p1.X,                  # X
            buttons_p1.A,                  # A
            buttons_p1.L,                  # L
            buttons_p1.R                   # R
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

    def fight(self, current_game_state, player):
        if self.prev_opponent_health is None:
            self.prev_opponent_health = current_game_state.player2.health if player == "1" else current_game_state.player1.health

        opponent = current_game_state.player2 if player == "1" else current_game_state.player1
        diff = opponent.x_coord - current_game_state.player1.x_coord if player == "1" else current_game_state.player1.x_coord - current_game_state.player2.x_coord
        my_health = current_game_state.player1.health if player == "1" else current_game_state.player2.health

        # Reset buttons to avoid stale inputs
        self.buttn = Buttons()

        # Decision tree based on game state
        move = None
        if self.exe_code != 0:
            # Continue executing current move sequence
            self.run_command([], current_game_state.player1 if player == "1" else current_game_state.player2)
        elif opponent.is_player_in_move or my_health < 15:  # Lowered defensive threshold
            # Defensive play: block or dodge
            if opponent.is_jumping:
                move = "shoryuken"  # Anti-air
            elif opponent.is_crouching:
                move = "medium_punch"  # Attack instead of block high
            else:
                move = "block_low"  # Block low attacks
        elif abs(diff) > 100:
            # Long range: use projectiles or approach
            if np.random.random() < 0.85:  # Increased projectile frequency
                move = "hadoken"  # Fireball
            else:
                move = "jump_forward"  # Approach faster
        elif abs(diff) > 50:
            # Mid range: mix attacks and movement
            if opponent.is_jumping:
                move = "shoryuken"  # Anti-air
            elif np.random.random() < 0.7:  # Increased attack frequency
                move = "medium_punch"  # Quick attack
            else:
                move = "long_kick"  # Poke
        else:
            # Close range: high-damage moves or throws
            if opponent.health < 20:
                move = "throw"  # Finish with throw
            elif np.random.random() < 0.7:  # Increased special frequency
                move = "shoryuken"  # High-damage uppercut
            else:
                move = "crouch_kick"  # Low attack

        if move:
            self.run_command(self.moves[move], current_game_state.player1 if player == "1" else current_game_state.player2)

        # Assign buttons to command
        if player == "1":
            self.my_command.player_buttons = self.buttn
        else:
            self.my_command.player2_buttons = self.buttn

        # Log the current state
        self.log_data(current_game_state, self.my_command, player)
        return self.my_command

    def run_command(self, com, player):
        if not com and self.exe_code >= len(self.fire_code):
            self.exe_code = 0
            self.start_fire = False
            self.remaining_code = []
            return

        if com:
            self.fire_code = com
            self.exe_code = 0
            self.remaining_code = self.fire_code[:]
            self.start_fire = True

        if not self.remaining_code:
            self.exe_code = 0
            return

        self.exe_code += 1
        cmd = self.remaining_code[0]
        print(f"Executing: {cmd}")

        # Reset buttons to avoid overlap
        self.buttn = Buttons()

        # Handle movement and attacks
        if cmd == "v":
            self.buttn.down = True
        elif cmd == "!v":
            self.buttn.down = False
        elif cmd == "<":
            self.buttn.left = True
        elif cmd == "!<":
            self.buttn.left = False
        elif cmd == ">":
            self.buttn.right = True
        elif cmd == "!>":
            self.buttn.right = False
        elif cmd == "^":
            self.buttn.up = True
        elif cmd == "!^":
            self.buttn.up = False
        elif cmd == "v+<":
            self.buttn.down = True
            self.buttn.left = True
        elif cmd == "!v+!<":
            self.buttn.down = False
            self.buttn.left = False
        elif cmd == "v+>":
            self.buttn.down = True
            self.buttn.right = True
        elif cmd == "!v+!>":
            self.buttn.down = False
            self.buttn.right = False
        elif cmd == ">+Y":
            self.buttn.right = True
            self.buttn.Y = True
        elif cmd == "!>+!Y":
            self.buttn.right = False
            self.buttn.Y = False
        elif cmd == "<+Y":
            self.buttn.left = True
            self.buttn.Y = True
        elif cmd == "!<+!Y":
            self.buttn.left = False
            self.buttn.Y = False
        elif cmd == ">+^":
            self.buttn.right = True
            self.buttn.up = True
        elif cmd == "!>+!^":
            self.buttn.right = False
            self.buttn.up = False
        elif cmd == "Y":
            self.buttn.Y = not player.player_buttons.Y
        elif cmd == "!Y":
            self.buttn.Y = False
        elif cmd == "B":
            self.buttn.B = not player.player_buttons.B
        elif cmd == "!B":
            self.buttn.B = False
        elif cmd == "X":
            self.buttn.X = not player.player_buttons.X
        elif cmd == "!X":
            self.buttn.X = False
        elif cmd == "A":
            self.buttn.A = not player.player_buttons.A
        elif cmd == "!A":
            self.buttn.A = False

        self.remaining_code = self.remaining_code[1:]
        return