import chess
import chess.pgn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import argparse
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        print(f"Logits shape: {logits.shape}")
        print(f"Labels shape: {labels.shape if labels is not None else 'No labels'}")

        if labels is not None:
            labels = labels.to(dtype=torch.long)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            print(f"Initial loss shape: {loss.shape}")
            loss = loss.mean()
            print(f"Final loss (should be scalar): {loss.item()}")
            return loss
        else:
            return logits

def chess_dataset(dataset):
    # Add debugging print statements
    print("Sample item from dataset:", dataset[0])
    
    encoded_data = tokenizer(dataset["text"], truncation=True, padding=True, return_tensors="pt")
    
    # Check the structure of encoded_data
    print("Sample encoded item:", encoded_data)
    
    return {
        "input_ids": encoded_data["input_ids"].squeeze(0),
        "attention_mask": encoded_data["attention_mask"].squeeze(0),
        "labels": encoded_data["input_ids"].clone().squeeze(0)
    }

def preprocess_pgn(pgn_file_path):
    games = []
    with open(pgn_file_path) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            moves = []
            for move in game.mainline_moves():
                try:
                    moves.append(board.san(move))
                    board.push(move)
                except AssertionError as e:
                    print(f"Illegal move encountered: {move} in position {board.fen()}")
                    print(f"Full game moves up to error: {' '.join(moves)}")
                    break
            moves_str = " ".join(moves)
            games.append(moves_str)
    return games

def tokenize_function(examples):
    """
    Tokenize the input text data.
    
    :param examples: A dictionary containing the text data
    :return: A dictionary containing the tokenized data
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

def main(pgn_file_path):
    # Load and preprocess data
    chess_data = preprocess_pgn(pgn_file_path)

    # Ensure chess_data is a list of dictionaries
    chess_data = [{"text": game} for game in chess_data]

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
    model = ChessModel(AutoModelForCausalLM.from_pretrained(model_name))

    # Ensure the tokenizer has a padding token
    #
    # 1. Padding tokens:
    #
    # * In NLP, we often need to process sequences of different lengths in batches.
    #
    # * To do this efficiently, we typically pad shorter sequences to match the length
    #  of the longest sequence in the batch.
    #
    # * A padding token is a special token used for this purpose. It doesn't carry
    #  any meaning and is ignored during processing.
    #
    # 2. GPT-2's lack of a padding token:
    #
    # * Unlike some other models, GPT-2 doesn't have a dedicated padding token by default.
    #
    # * This can cause issues when trying to process batches of sequences with varying lengths.
    #
    # 3. Using the EOS token as a padding token:
    #
    # * EOS stands for "End of Sequence".
    #
    # * By setting tokenizer.pad_token = tokenizer.eos_token, we're telling the tokenizer
    #  to use the End of Sequence token as a padding token.
    #
    # * This is a common workaround for GPT-2 models.
    #
    # 4. Why this works:
    #
    # * The EOS token is already treated specially by the model (it signifies the end of input).
    # * Using it as a padding token ensures that the model will effectively ignore the padding,
    #  as it would stop processing at the first EOS token it encounters.
    #
    # 5. Implications:
    #
    # * This allows us to process batches of sequences with different lengths efficiently.
    # * It ensures that the padding doesn't interfere with the model's understanding of the actual content.
    #
    # GPT-2 doesn't have a padding token by default, so we use the EOS token instead
    tokenizer.pad_token = tokenizer.eos_token

    # Apply the tokenization function to our entire dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define num_eval_samples before using it
    num_eval_samples = min(max(100, int(0.1 * len(dataset))), len(dataset))  # 10% of dataset or at least 100 samples

    # Apply chess_dataset to the entire tokenized dataset
    train_dataset = chess_dataset(tokenized_dataset)
    eval_dataset = chess_dataset(dataset.select(range(len(dataset) - num_eval_samples, len(dataset))))

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./chess_model",    # Directory to save the model checkpoints
        overwrite_output_dir=True,     # Overwrite the content of the output directory

        # An epoch is one complete pass through the entire training dataset.
        # More epochs can lead to better learning, but too many can cause overfitting.
        num_train_epochs=5,            # Number of training epochs

        # This sets the batch size for training on each device (e.g., GPU or CPU).
        # Larger batch sizes can lead to faster training but require more memory.
        per_device_train_batch_size=8, # Batch size for training

        # This parameter determines how often the model is saved during training.
        # A "step" typically refers to processing one batch of data.
        # In this case, the model will be saved every 1000 steps.
        save_steps=1000,               # Number of updates steps before two checkpoint saves

        # This parameter limits the total number of checkpoint files saved.
        # When this limit is reached, older checkpoints are deleted.
        # This is useful for saving disk space, as model checkpoints can be large
        save_total_limit=2,            # Limit the total amount of checkpoints. Deletes the older checkpoints.

        # The learning rate is a crucial hyperparameter in machine learning.
        # It determines the step size at each iteration while moving toward a minimum of the loss function.
        # In this case, the learning rate is set to 2e-5 (0.00002).
        # A smaller learning rate:
        #  - Allows the model to explore the loss landscape more carefully
        #  - Can lead to more precise convergence, which may take longer time
        # A larger learning rate:
        #  - Can speed up training
        #  - But might overshoot the optimal solution or cause unstable training
        # 2e-5 is a relatively small learning rate, often used in fine-tuning pre-trained models.
        learning_rate=2e-5,            # Learning rate

        # This parameter is part of the learning rate scheduler.
        # During the warmup phase, the learning rate gradually increases from 0 to the set learning rate.
        # In this case, the learning rate will increase gradually over the first 500 steps.
        # Warmup steps help to:
        #   - Stabilize training in the early stages
        #   - Prevent large, potentially harmful updates when the model is still "cold"
        warmup_steps=500,              # Number of warmup steps for learning rate scheduler

        # 1. Purpose of weight decay:
        #   - It adds a penalty term to the loss function based on the magnitude of the model's weights.
        #   - This encourages the model to use smaller weights, which can lead to simpler and more generalizable models. 
        # 2. How it works:
        #   - During each update step, the weights are not only adjusted based on the gradient of the loss function
        #     but also slightly reduced in proportion to their current value.
        #   - This helps prevent the weights from growing too large, which can cause instability or overfitting.
        # 3. Effect on training:
        #   - It slows down the rate of change in the weights, allowing for more gradual adjustments.
        #   - This can lead to better convergence and more stable training.
        # 4. Choosing a value:
        #   - The optimal value depends on the specific model and dataset.
        #   - 0.01 is a common starting point for weight decay.
        weight_decay=0.01,             # Strength of weight decay
    )

    # Define eval_dataset before using it
    eval_dataset = dataset.select(range(len(dataset) - num_eval_samples, len(dataset)))
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,                      # The instantiated model to be trained
        args=training_args,               # Training arguments, defined above
        train_dataset=train_dataset,      # Wrap with chess_dataset
        eval_dataset=eval_dataset        # Wrap with chess_dataset
    )

    # Start the training process
    trainer.train()

    # Save the final model and tokenizer
    model.save_pretrained("./chess_model_final")
    tokenizer.save_pretrained("./chess_model_final")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine-tune a language model on chess games.")
    parser.add_argument("pgn_file", help="Path to the PGN file containing chess games")
    
    # Parse arguments
    args = parser.parse_args()

    # Call main function with the provided PGN file path
    main(args.pgn_file)