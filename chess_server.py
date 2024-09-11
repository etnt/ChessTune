"""
Chess Server API Endpoints:

1. Initialize a new game:
   POST /init
   curl -X POST http://localhost:5000/init

2. Make a move:
   POST /move
   curl -X POST -H "Content-Type: application/json" -d '{"move": "e4"}' http://localhost:5000/move

3. Get AI's move suggestion:
   GET /get_move
   curl http://localhost:5000/get_move

4. Make AI's move on current board:
   POST /make_ai_move
   curl -X POST -H "Content-Type: application/json" -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}' http://localhost:5000/make_ai_move

5. Get current board state:
   GET /board
   curl http://localhost:5000/board
"""

from flask import Flask, request, jsonify
import chess
import chess.pgn
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import argparse
import logging
import random
from flask_cors import CORS

app = Flask(__name__)
#CORS(app)  # This will enable CORS for all routes

# Global variables to store the model, tokenizer, and current game state
model = None
tokenizer = None
board = None

logging.basicConfig(level=logging.INFO)

def load_model(model_path):
    global model, tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    # By using the original GPT-2 tokenizer, we're ensuring that our fine-tuned model
    # receives input in the format it expects, based on its original architecture and training.
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

def generate_move(num_moves=5):
    global model, tokenizer, board
    
    # Convert the entire game history to a string of moves in algebraic notation
    game = chess.pgn.Game.from_board(board)
    moves_str = ""
    try:
        moves_str = " ".join([board.san(move) for move in game.mainline_moves()])
    except AssertionError as e:
        logging.error(f"Error generating move history: {str(e)}")
        # If we can't generate the move history, we'll just use the current board state
        moves_str = board.fen()

    # Add a prompt to indicate it's the model's turn
    prompt = f"{moves_str} Next move:"
    
    # Tokenize the moves string for input to the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate the next move using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 10,
            num_return_sequences=num_moves,
            do_sample=True,
            temperature=0.7,
            attention_mask=inputs.attention_mask
        )

    # Decode the generated moves, taking only the first word of each generated sequence
    generated_moves = [tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().split()[0] for output in outputs]

    return generated_moves

@app.route('/init', methods=['POST'])
def init_game():
    """
    Initialize a new chess game.

    This function is called when a POST request is made to the '/init' endpoint.
    It creates a new chess board and sets it as the global board state.

    Returns:
    JSON: A dictionary containing the status of the operation and a message.
        - status: 'ok' if the game was successfully initialized.
        - message: A string confirming that a new game was initialized.
    """
    global board
    board = chess.Board()
    return jsonify({"status": "ok", "message": "New game initialized"})

@app.route('/move', methods=['POST'])
def make_move():
    """
    Apply a move to the current chess board.

    This function is called when a POST request is made to the '/move' endpoint.
    It expects a JSON payload with a 'move' key containing the move in Standard Algebraic Notation (SAN).

    Args:
        None (uses global 'board' variable)

    Returns:
    JSON: A dictionary containing the status of the operation and a message.
        - If the move is valid:
            status: 'ok'
            message: A string confirming that the move was applied
        - If the move is invalid:
            status: 'error'
            message: 'Invalid move'
            HTTP status code: 400 (Bad Request)

    Raises:
        ValueError: If the provided move is not valid in the current board position
    """
    global board
    move = request.json.get('move')
    try:
        board.push_san(move)
        return jsonify({"status": "ok", "message": f"Move {move} applied"})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid move"}), 400

@app.route('/get_move', methods=['GET'])
def get_ai_move():
    """
    Generate and apply an AI move to the current chess board.

    This function is called when a GET request is made to the '/get_move' endpoint.
    It generates a list of possible moves using the AI model, attempts to apply them,
    and falls back to a random legal move if necessary.

    Returns:
    JSON: A dictionary containing the status of the operation and move information.
        - If the game is over:
            status: 'game_over'
            result: The result of the game
        - If a valid move is made:
            status: 'ok'
            move: The move in Standard Algebraic Notation (SAN)
            new_fen: The new board state in Forsyth-Edwards Notation (FEN)
        - If no valid move is found:
            status: 'error'
            message: 'No valid move found'
            HTTP status code: 500 (Internal Server Error)

    Global Variables:
        board (chess.Board): The current state of the chess game

    Functions called:
        generate_move(): Generate a list of possible moves using the AI model
        board.parse_san(move_san): Parse a move in SAN format
        board.push(move): Apply a move to the board
        board.fen(en_passant='fen'): Get the current board state in FEN notation
        random.choice(legal_moves): Select a random move from the list of legal moves
    """
    global board
    if board.is_game_over():
        return jsonify({"status": "game_over", "result": board.result()})
    
    generated_moves = generate_move()
    
    for move_san in generated_moves:
        try:
            print(f"Attempting to parse move: {move_san}")
            move = board.parse_san(move_san)
            print(f"Parsed move: {move}")
            if move in board.legal_moves:
                print(f"Move {move_san} is legal")
                board.push(move)
                return jsonify({
                    "status": "ok",
                    "move": move_san,
                    "new_fen": board.fen(en_passant='fen')
                })
            else:
                print(f"Move {move_san} is not legal in the current position")
        except ValueError as e:
            print(f"Error parsing move {move_san}: {str(e)}")
            continue
    # If no generated move is legal, choose a random move
    legal_moves = list(board.legal_moves)
    if legal_moves:
        random_move = random.choice(legal_moves)
        board.push(random_move)
        return jsonify({"status": "ok",
                         "move": board.san(random_move),
                         "new_fen": board.fen(en_passant='fen')
                        })
    else:
        return jsonify({"status": "error", "message": "No valid move found"}), 500

@app.route('/board', methods=['GET'])
def get_board():
    """
    Endpoint to get the current state of the chess board.

    This function handles GET requests to the '/board' route.
    It returns the current state of the chess board in FEN (Forsyth–Edwards Notation) format.

    Returns:
        JSON: A dictionary containing:
            - 'status': Always 'ok' for this endpoint
            - 'fen': The current board state in FEN notation

    Global Variables:
        board (chess.Board): The current state of the chess game

    Functions called:
        board.fen(en_passant='fen'): Get the current board state in FEN notation
    """
    global board
    return jsonify({"status": "ok", "fen": board.fen(en_passant='fen')})


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess server using GPT-2 model")
    parser.add_argument("--model", default="./chess_model", help="Path to the trained model (default: ./chess_model)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on (default: 5000)")
    
    args = parser.parse_args()

    load_model(args.model)
    app.run(debug=True, port=args.port)
