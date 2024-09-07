# ChessTune

ChessTune is a project that uses machine learning to train a model on chess games and then allows you to play against the trained AI. It leverages the power of the GPT-2 language model to learn patterns from a large dataset of chess games and generate moves based on the current board state.

## Features

- **Data Processing**: Converts PGN (Portable Game Notation) files into a format suitable for training.
- **Model Training**: Uses a GPT-2 model to learn chess move patterns from processed game data.
- **Interactive Gameplay**: Allows users to play chess against the trained AI model.
- **Move Generation**: The AI generates multiple move suggestions and chooses the best legal move.
- **HTTP API**: Provides a simple HTTP API to interact with the trained model.

## How It Works

1. **Data Preparation**: The script processes a large PGN file containing chess games, converting them into a format suitable for training a language model.

2. **Model Training**: A GPT-2 model is fine-tuned on the processed chess data, learning to predict the next move given a sequence of previous moves. When we fine-tune a model, we're adjusting the weights of the neural network to better perform on our specific task (chess moves, in this case). 

3. **Gameplay**: The `play-chess.py` script sets up a chess board and alternates between human and AI moves. The AI generates move suggestions using the trained model and chooses a legal move from these suggestions.

## Getting Started

1. Ensure you have Python 3.x installed.
2. Clone this repository.
3. Run `make all` to set up the virtual environment and prepare the data.
4. Run: `./venv/bin/python chess-tune.py <pgn-directory> [--max-games <max-games>]` to start the training process.
5. After training, run: `./venv/bin/python play-chess.py` to play against the AI.
or `./venv/bin/python chess-server.py` to run the chess server.

## Requirements

A complete list of requirements can be found in `requirements.txt`.

## File Structure

- `chess-tune.py`: Main script for processing data and training the model.
- `play-chess.py`: Script for playing chess against the trained AI.
- `chess-server.py`: Script for running the chess server.
- `test_chess_server.py`: Script for testing the chess server.
- `PGN`: directory containing PGN files for training.

## Time to train

On a Macbook Pro M1 Max with 64 GB of RAM, training capacity is roughly 11 games per minute.
For example: 1067 games took 83 minutes to train, and 2924 games took 264 minutes to train.

## HTTP API

Initialize a game:

```bash
$ curl -is -X POST http://127.0.0.1:5000/init
HTTP/1.1 200 OK
                                      
{               
  "message": "New game initialized",
  "status": "ok"                                                                                     
}                                    
```

Make a move:

```bash
$ curl -is -X POST -H "Content-Type: application/json" -d '{"move": "e2e4"}' http://127.0.0.1:5000/move
HTTP/1.1 200 OK
{
  "message": "Move e2e4 applied",
  "status": "ok"
}
```

Get the AI's move:

```bash
$ curl -is -X GET http://127.0.0.1:5000/get_move
HTTP/1.1 200 OK
{
  "move": "Nf6",
  "status": "ok"
}
```

Get the board state:

```bash
$ curl -is -X GET http://127.0.0.1:5000/board
HTTP/1.1 200 OK
{                                                 
  "fen": "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
  "status": "ok"                                  
}   
```


## Some Notes

### Load a pre-trained model and its associated tokenizer

When we say "Load a pre-trained model and its associated tokenizer,"
we're referring to two key components of natural language processing (NLP)
using transformer-based models. Let's break this down:
    
Pre-trained model:
    
* This is a neural network that has already been trained on a large corpus of text data.
    
* In this case, we're using GPT-2 (specifically, the "medium" size version), which was
trained by OpenAI on a diverse range of internet text.
    
* The model has learned to predict the next word in a sequence, capturing complex patterns
and relationships in language.
    
* By loading a pre-trained model, we're leveraging this existing knowledge as a starting point,
rather than training a model from scratch.
    
Associated tokenizer:
    
* A tokenizer is a tool that breaks down text into smaller units called tokens.
    
* These tokens could be words, parts of words, or even individual characters,
depending on the tokenization strategy.
    
* The tokenizer associated with a pre-trained model knows exactly how the text
was split during the original training process.
    
* It's crucial to use the same tokenizer as the pre-trained model to ensure
consistency in how text is processed.

### Ensure the tokenizer has a padding token
 
1. Padding tokens:
    
    * In NLP, we often need to process sequences of different lengths in batches.
    
    * To do this efficiently, we typically pad shorter sequences to match the length
    of the longest sequence in the batch.
    
    * A padding token is a special token used for this purpose. It doesn't carry
    any meaning and is ignored during processing.
    
2. GPT-2's lack of a padding token:
    
    * Unlike some other models, GPT-2 doesn't have a dedicated padding token by default.
    
    * This can cause issues when trying to process batches of sequences with varying lengths.
    
3. Using the EOS token as a padding token:
    
    * EOS stands for "End of Sequence".
    
    * By setting tokenizer.pad_token = tokenizer.eos_token, we're telling the tokenizer
    to use the End of Sequence token as a padding token.
    
    * This is a common workaround for GPT-2 models.
    
4. Why this works:
    
    * The EOS token is already treated specially by the model (it signifies the end of input).

    * Using it as a padding token ensures that the model will effectively ignore the padding,
    as it would stop processing at the first EOS token it encounters.
    
5. Implications:
    
    * This allows us to process batches of sequences with different lengths efficiently.
    
    * It ensures that the padding doesn't interfere with the model's understanding of the actual content.
    
GPT-2 doesn't have a padding token by default, so we use the EOS token instead
