from flask import Flask, request, jsonify
import chess
import chess.pgn
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import argparse
import logging
import sys
import traceback

# Initialize a new game:
#   curl -X POST http://localhost:5000/init
#
# Make a move:
#   curl -X POST -H "Content-Type: application/json" -d '{"move": "e4"}' http://localhost:5000/move
#
# Get AI's move:
#   curl http://localhost:5000/get_move
#
# Get current board state:
#   curl http://localhost:5000/board

app = Flask(__name__)

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
    global board
    board = chess.Board()
    return jsonify({"status": "ok", "message": "New game initialized"})

@app.route('/move', methods=['POST'])
def make_move():
    global board
    move = request.json.get('move')
    try:
        board.push_san(move)
        return jsonify({"status": "ok", "message": f"Move {move} applied"})
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid move"}), 400

@app.route('/get_move', methods=['GET'])
def get_ai_move():
    global board
    if board.is_game_over():
        return jsonify({"status": "game_over", "result": board.result()})
    
    generated_moves = generate_move()
    
    for move_san in generated_moves:
        try:
            move = board.parse_san(move_san)
            if move in board.legal_moves:
                board.push(move)
                return jsonify({"status": "ok", "move": move_san})
        except ValueError:
            continue
    
    return jsonify({"status": "error", "message": "No valid move found"}), 500

@app.route('/board', methods=['GET'])
def get_board():
    global board
    return jsonify({"status": "ok", "fen": board.fen(en_passant='fen')})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess server using GPT-2 model")
    parser.add_argument("--model", default="./chess_model", help="Path to the trained model (default: ./chess_model)")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on (default: 5000)")
    
    args = parser.parse_args()

    load_model(args.model)
    app.run(debug=True, port=args.port)
