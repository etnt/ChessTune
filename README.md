# ChessTune

ChessTune is a project that uses machine learning to train a model on chess games and then allows you to play against the trained AI. It leverages the power of the GPT-2 language model to learn patterns from a large dataset of chess games and generate moves based on the current board state.

## Features

- **Data Processing**: Converts PGN (Portable Game Notation) files into a format suitable for training.
- **Model Training**: Uses a GPT-2 model to learn chess move patterns from processed game data.
- **Interactive Gameplay**: Allows users to play chess against the trained AI model.
- **Move Generation**: The AI generates multiple move suggestions and chooses the best legal move.

## How It Works

1. **Data Preparation**: The script processes a large PGN file containing chess games, converting them into a format suitable for training a language model.

2. **Model Training**: A GPT-2 model is fine-tuned on the processed chess data, learning to predict the next move given a sequence of previous moves.

3. **Gameplay**: The `play-chess.py` script sets up a chess board and alternates between human and AI moves. The AI generates move suggestions using the trained model and chooses a legal move from these suggestions.

## Getting Started

1. Ensure you have Python 3.x installed.
2. Clone this repository.
3. Run `make all` to set up the virtual environment and prepare the data.
4. Use `make run` to start the training process.
5. After training, run `python play-chess.py` to play against the AI.

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- Python-chess

A complete list of requirements can be found in `requirements.txt`.

## File Structure

- `chess-tune.py`: Main script for processing data and training the model.
- `play-chess.py`: Script for playing chess against the trained AI.

## Future Improvements

- Implement a GUI for a more user-friendly chess playing experience.
- Optimize the model for better move prediction and gameplay.
- Add support for different chess variants.

