import chess
import chess.pgn
from datasets import Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def preprocess_pgn(pgn_file_path, max_games=10):
    games = []
    with open(pgn_file_path) as pgn:
        for _ in range(max_games):
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
    return games

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

def main(pgn_file_path):
    # Load and preprocess data
    chess_data = preprocess_pgn(pgn_file_path)
    print(f"Processed {len(chess_data)} games")

    # Create a Dataset object
    dataset = Dataset.from_list(chess_data)
    print(f"Dataset size: {len(dataset)}")

    # Initialize tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print("Dataset tokenized")

    # Initialize model
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.resize_token_embeddings(len(tokenizer))

    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,  # Adjust this as needed
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        evaluation_strategy="steps",
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
    pgn_file_path = "Mikhail-Tal-Best-Games.pgn"  # Replace with your actual file path
    main(pgn_file_path)
