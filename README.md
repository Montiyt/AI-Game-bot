# AI-Game-bot
A trained AI game bot playing Street fighting game.

# AI Street Fighter

This project is an AI-powered agent trained to play a **street fighting game**. It includes both human and AI-controlled players, game state logging, and model training utilities. The AI learns from gameplay data and is capable of making strategic decisions to compete in a simulated environment.

## Features

- 🕹️ **Human and AI Controller Support**
- 🤖 **Bot AI using Trained Model**
- 📊 **Game Logging and Analysis**
- 🧠 **Model Training with Game Data**
- ⌨️ **Keyboard-Based Human Player**

## File Descriptions

- `bot.py` - Main AI agent that executes actions in the game.
- `buttons.py` - Defines button mappings used for actions in the game.
- `command.py` - Handles command interface or action execution logic.
- `controller.py` - Controls the AI/human inputs to interact with the game.
- `controllerk.py` - Possibly a variation or keyboard-specific controller.
- `controller.log` - Log output generated during game execution.
- `extra.py` - Additional helper functions or utilities.
- `game_log.csv` - Stores historical game performance data.
- `game_state.py` - Handles game state tracking and transitions.
- `keyboard_player.py` - Handles keyboard input for a human player.
- `player.py` - Generic player class (AI or human).
- `train_model.py` - Training script for the AI using game data.
- `README.md` - This file.

## Getting Started

### Prerequisites

- Python 3.7+
- Common ML libraries: `numpy`, `pandas`, `tensorflow` or `pytorch` (depending on your framework)

### How to Run

1. **Play with Keyboard:**
   ```bash
   python keyboard_player.py

