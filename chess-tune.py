import chess
import chess.pgn
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import torch

def preprocess_pgn(pgn_file_path, max_games=None):
    """
    Preprocess a PGN (Portable Game Notation) file and extract chess games as a list of move sequences.

    This function reads a PGN file, extracts the moves from each game, and converts them to a string
    representation. It can process either all games in the file or a specified maximum number of games.

    Args:
        pgn_file_path (str): The file path to the PGN file to be processed.
        max_games (int, optional): The maximum number of games to process. If None, all games are processed.

    Returns:
        list: A list of dictionaries, where each dictionary contains a 'text' key with the value being
              a string of space-separated moves in Standard Algebraic Notation (SAN) for one game.

    Note:
        - The function uses the chess library to parse PGN and handle chess logic.
        - Each move is converted to SAN before being added to the move list.
        - The function stops processing if it reaches the end of the file or the max_games limit.
    """
    games = []
    with open(pgn_file_path) as pgn:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = []
            for move in game.mainline_moves():
                moves.append(board.san(move))
                board.push(move)
            moves_str = " ".join(moves)
            games.append({"text": moves_str})
            game_count += 1
            if max_games is not None and game_count >= max_games:
                break
    return games


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)



def main(pgn_dir, max_games=None, resume_from=None):
    """
    Main function to prepare and train a chess model.

    This function orchestrates the entire process of preparing a dataset from PGN files,
    tokenizing the data, and training a language model using the Hugging Face Transformers library.

    Args:
        pgn_dir (str): The directory containing PGN files to be processed.
        max_games (int, optional): The maximum number of games to process. If None, all games are processed.
        resume_from (str, optional): Path to the model to resume training from.

    Returns:
        None

    The function performs the following steps:
    1. Processes all PGN files in the specified directory, extracting and preprocessing chess games.
    2. Tokenizes the move sequences using a GPT-2 tokenizer.
    3. Creates a Dataset object from the tokenized data.
    4. Initializes a tokenizer for the chosen model.
    5. Tokenizes the dataset.
    6. Checks for MPS (Metal Performance Shaders) availability and sets up the device accordingly.
    7. Initializes the model, either by resuming from a checkpoint or starting from scratch.
    8. Sets up data collator and training arguments.
    9. Initializes the Trainer and starts training.

    """
    all_chess_data = []
    for filename in os.listdir(pgn_dir):
        if filename.endswith('.pgn'):
            pgn_file_path = os.path.join(pgn_dir, filename)
            chess_data = preprocess_pgn(pgn_file_path, max_games)
            all_chess_data.extend(chess_data)
            if max_games and len(all_chess_data) >= max_games:
                all_chess_data = all_chess_data[:max_games]
                break
    
    print(f"Processed {len(all_chess_data)} games in total")

    # Create a Dataset object
    dataset = Dataset.from_list(all_chess_data)
    print(f"Dataset size: {len(dataset)}")

    # Initialize tokenizer
    global tokenizer
    # GPT2-medium is chosen here, but you might want to experiment with other models
    model_name = "gpt2-medium"

    # Download and initialize the tokenizer that was used with our choosen model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset tokenized")

    # Check for MPS availability
    if torch.backends.mps.is_available():
        # Mac with an Mx chip, the GPU being labeled as MPS stands for Metal Performance Shaders.
        # MPS is part of Apple's Metal framework, which is a low-level, high-performance API for
        # graphics and compute tasks on macOS, iOS, and other Apple devices.
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    elif torch.cuda.is_available():
        # CUDA (Compute Unified Device Architecture) is NVIDIA's proprietary technology for
        # general-purpose computing on GPUs. It allows developers to leverage the parallel
        # processing capabilities of NVIDIA GPUs for accelerating various tasks, particularly
        # in the fields of deep learning and machine learning.
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Initialize model
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        model = GPT2LMHeadModel.from_pretrained(resume_from)
    else:
        print("Starting training from scratch")
        model = GPT2LMHeadModel.from_pretrained(model_name)
    
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)  # Move model to the selected device

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",

        # An epoch is one complete pass through the entire training dataset.
        # More epochs can lead to better learning, but too many can cause overfitting.
        num_train_epochs=3,  # Adjust this as needed

        # This sets the batch size for training on each device (e.g., GPU or CPU).
        # Larger batch sizes can lead to faster training but require more memory.
        per_device_train_batch_size=4,

        # This parameter determines how often the model is saved during training.
        # A "step" typically refers to processing one batch of data.
        # In this case, the model will be saved every 1000 steps.
        save_steps=1000,

        # This parameter limits the total number of checkpoint files saved.
        # When this limit is reached, older checkpoints are deleted.
        # This is useful for saving disk space, as model checkpoints can be large
        save_total_limit=2,

        
        logging_steps=100,
        eval_strategy="steps",  # Instead of evaluation_strategy
        eval_steps=500,
        load_best_model_at_end=True
        #use_mps_device=True,  # Use this instead of 'device'
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # Using the same dataset for evaluation (you might want to create a separate validation set)
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed")

    # Save the model
    trainer.save_model("./chess_model")
    print("Model saved")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine-tune a language model on chess games.")
    parser.add_argument("pgn_dir", help="Directory containing PGN files")
    parser.add_argument("--max-games", type=int, help="Maximum number of games to process (default: all games)")
    parser.add_argument("--resume-from", help="Path to the model to resume training from")
    
    # Parse arguments
    args = parser.parse_args()

    # pgn_file_path = "Mikhail-Tal-Best-Games.pgn"  # Replace with your actual file path

    main(args.pgn_dir, args.max_games, args.resume_from)
