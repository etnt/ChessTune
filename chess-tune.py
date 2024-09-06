import chess
import chess.pgn
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def preprocess_pgn(pgn_file_path, max_games=None):
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

def main(pgn_file_path, max_games=None):
    # Load and preprocess data
    chess_data = preprocess_pgn(pgn_file_path, max_games)
    print(f"Processed {len(chess_data)} games")

    # Create a Dataset object
    dataset = Dataset.from_list(chess_data)
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

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

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
        load_best_model_at_end=True,
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
    parser.add_argument("pgn_file_path", help="Path to the PGN file containing chess games")
    parser.add_argument("--max-games", type=int, help="Maximum number of games to process (default: all games)")
    
    # Parse arguments
    args = parser.parse_args()

    # pgn_file_path = "Mikhail-Tal-Best-Games.pgn"  # Replace with your actual file path

    main(args.pgn_file_path, args.max_games)
