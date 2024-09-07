import chess
import chess.pgn
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch
import random
import argparse

def load_model(model_path):
    """
    Load the trained model and tokenizer.
    
    Args:
    model_path (str): Path to the saved model

    Returns:
    tuple: (model, tokenizer)
    """
    # Load the pre-trained model from the specified path
    model = GPT2LMHeadModel.from_pretrained(model_path)
    # Load the tokenizer associated with the GPT-2 medium model
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    # Set the pad token to be the same as the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_move(model, tokenizer, board, num_moves=5):
    """
    Generate chess moves using the trained model.
    
    Args:
    model: The trained GPT-2 model
    tokenizer: The associated tokenizer
    board (chess.Board): The current chess board state
    num_moves (int): Number of move suggestions to generate

    Returns:
    list: Generated move suggestions
    """
    # Convert the entire game history to a string of moves in algebraic notation
    game = chess.pgn.Game.from_board(board)
    moves_str = ""
    current_board = chess.Board()
    for move in game.mainline_moves():
        moves_str += current_board.san(move) + " "
        current_board.push(move)

    # Add a prompt to indicate it's the model's turn
    prompt = f"{moves_str}Next move:"
    
    # Tokenize the moves string for input to the model
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate the next move using the model
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 10,
            num_return_sequences=num_moves,
            do_sample=True,
            temperature=0.7,
            attention_mask=inputs.attention_mask
        )

    # Decode the generated moves, taking only the first word of each generated sequence
    generated_moves = [tokenizer.decode(output[inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True).strip().split()[0] for output in outputs]

    return generated_moves

def algebraic_to_uci(board, move_san):
    """
    Convert a move from algebraic notation to UCI notation.
    
    Args:
    board (chess.Board): The current chess board state
    move_san (str): Move in algebraic notation

    Returns:
    str: Move in UCI notation, or None if invalid
    """
    try:
        move = board.parse_san(move_san)
        return move.uci()
    except ValueError:
        return None

def play_game(model_path):
    """
    Main function to play a game of chess against the AI.
    """
    # Load the trained model and tokenizer
    model, tokenizer = load_model(model_path)
    # Initialize a new chess board
    board = chess.Board()

    while not board.is_game_over():
        # Display the current board state
        print(board)
        # Show legal moves in algebraic notation
        print("\nLegal moves:", [board.san(move) for move in board.legal_moves])

        if board.turn == chess.WHITE:
            # Human player's turn (White)
            while True:
                move = input("Enter your move (in algebraic notation, e.g., 'e4' or 'Nf3'): ")
                if move.lower() == "quit":
                    return
                try:
                    # Attempt to make the move
                    board.push_san(move)
                    break
                except ValueError:
                    print("Invalid move. Try again.")
        else:
            # AI's turn (Black)
            print("AI is thinking...")
            # Generate move suggestions using the model
            generated_moves = generate_move(model, tokenizer, board)
            print("AI's top 5 move suggestions:", generated_moves)

            # Try moves until a legal one is found
            for move_san in generated_moves:
                try:
                    # Parse the move in algebraic notation
                    move = board.parse_san(move_san)
                    if move in board.legal_moves:
                        # Make the move if it's legal
                        board.push(move)
                        print(f"AI plays: {move_san}")
                        break
                except ValueError:
                    # If the move is invalid, continue to the next suggestion
                    continue
            else:
                # If no generated move is legal, choose a random move
                print("AI couldn't generate a legal move. Choosing randomly.")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    random_move = random.choice(legal_moves)
                    board.push(random_move)
                    try:
                        # Try to display the move in algebraic notation
                        print(f"AI plays: {board.san(random_move)}")
                    except AssertionError:
                        # If that fails, display in UCI notation
                        print(f"AI plays: {random_move.uci()}")
                else:
                    print("No legal moves available. Game over.")
                    break

    # Game has ended
    print("Game Over")
    print("Result:", board.result())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play chess against an AI trained with GPT-2.")
    parser.add_argument("--model", default="./chess_model", help="Path to the trained model (default: ./chess_model)")
    
    args = parser.parse_args()

    play_game(args.model)
