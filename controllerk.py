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

def connect(port, max_attempts=20, timeout=10):
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
        port = 10000
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