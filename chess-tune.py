import chess.pgn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def preprocess_pgn(pgn_file):
    """
    Preprocess a PGN file, extracting moves in algebraic notation.
    
    :param pgn_file: An open file object containing PGN data
    :return: A list of dictionaries, each containing a game's moves as text
    """
    games = []
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        # Extract moves in Standard Algebraic Notation (SAN)
        moves = " ".join(move.san() for move in game.mainline_moves())
        games.append({"text": moves})
    return games

def tokenize_function(examples):
    """
    Tokenize the input text data.
    
    :param examples: A dictionary containing the text data
    :return: A dictionary containing the tokenized data
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def main():
    # Load and preprocess data
    with open("chess_games.pgn") as pgn_file:
        chess_data = preprocess_pgn(pgn_file)

    # Create a Dataset object from the preprocessed data
    # This creates a Hugging Face Dataset, which is optimized for ML tasks
    dataset = Dataset.from_list(chess_data)

    # Make tokenizer global so tokenize_function can access it. 
    # After this declaration, when we assign AutoTokenizer.from_pretrained(model_name)
    # to tokenizer, we're modifying the global tokenizer variable, which can then be
    # accessed by tokenize_function().
    global tokenizer  

    # Load a pre-trained model and its associated tokenizer
    #
    # When we say "Load a pre-trained model and its associated tokenizer,"
    # we're referring to two key components of natural language processing (NLP)
    # using transformer-based models. Let's break this down:
    #
    # Pre-trained model:
    #
    # * This is a neural network that has already been trained on a large corpus of text data.
    #
    # * In this case, we're using GPT-2 (specifically, the "medium" size version), which was
    #  trained by OpenAI on a diverse range of internet text.
    #
    # * The model has learned to predict the next word in a sequence, capturing complex patterns
    #  and relationships in language.
    #
    # * By loading a pre-trained model, we're leveraging this existing knowledge as a starting point,
    #  rather than training a model from scratch.
    #
    # Associated tokenizer:
    #
    # * A tokenizer is a tool that breaks down text into smaller units called tokens.
    #
    # * These tokens could be words, parts of words, or even individual characters,
    #  depending on the tokenization strategy.
    #
    # * The tokenizer associated with a pre-trained model knows exactly how the text
    #  was split during the original training process.
    #
    # * It's crucial to use the same tokenizer as the pre-trained model to ensure
    #  consistency in how text is processed.
    #
    # GPT2-medium is chosen here, but you might want to experiment with other models
    model_name = "gpt2-medium"
    # Download and initialize the tokenizer that was used with our choosen model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Download and initialize the pre-trained model itself.
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure the tokenizer has a padding token
    # GPT-2 doesn't have a padding token by default, so we use the EOS token instead
    tokenizer.pad_token = tokenizer.eos_token

    # Apply the tokenization function to our entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./chess_model",  # Directory to save the model checkpoints
        overwrite_output_dir=True,   # Overwrite the content of the output directory
        num_train_epochs=5,          # Number of training epochs
        per_device_train_batch_size=8,  # Batch size for training
        save_steps=1000,             # Number of updates steps before two checkpoint saves
        save_total_limit=2,          # Limit the total amount of checkpoints. Deletes the older checkpoints.
        learning_rate=2e-5,          # Learning rate
        warmup_steps=500,            # Number of warmup steps for learning rate scheduler
        weight_decay=0.01,           # Strength of weight decay
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,                 # The instantiated model to be trained
        args=training_args,          # Training arguments, defined above
        train_dataset=tokenized_dataset,  # Training dataset
    )

    # Start the training process
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained("./chess_model_final")
    tokenizer.save_pretrained("./chess_model_final")

if __name__ == "__main__":
    main()