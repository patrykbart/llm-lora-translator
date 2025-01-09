import json
import logging
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sklearn.metrics import top_k_accuracy_score
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from utils import load_config, load_model_and_tokenizer

# Constants
INVALID_LABEL = -100
TOP_K_VALUES = [5, 25, 100]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    """Compute top-k accuracy metrics for evaluation predictions."""
    predictions, labels = eval_pred

    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)

    # Create a mask for valid labels
    valid_mask = labels != INVALID_LABEL

    # Apply the mask to filter predictions and labels
    filtered_predictions = predictions[valid_mask]
    filtered_labels = labels[valid_mask]

    # Precompute the range of labels
    label_range = np.arange(predictions.shape[-1])

    # Compute top-k accuracies
    accuracies = {
        f"top_{k}_accuracy": top_k_accuracy_score(filtered_labels, filtered_predictions, k=k, labels=label_range)
        for k in TOP_K_VALUES
    }

    return accuracies

def prepare_dataset(tokenizer, model, split: str = "train[-20:]"):
    """Load and tokenize the evaluation dataset."""
    eval_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split, cache_dir="./cache")
    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples.")

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=model.config.max_position_embeddings,
            padding='max_length',
        )

    tokenized_eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=eval_dataset.column_names,
        desc='Tokenizing',
    )
    logger.info(f'Tokenized dataset:\n{tokenized_eval_dataset}')
    return tokenized_eval_dataset

def main():
    model_name = "lora_r32"

    # Setup paths
    current_dir = Path(__file__).parent
    config = load_config(current_dir / "config.json")
    model_dir = current_dir.parent / "outputs" / model_name / "final-model"
    output_dir = current_dir.parent / "eval_outputs" / model_name

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # Prepare dataset
    tokenized_eval_dataset = prepare_dataset(tokenizer, model)

    # Initialize Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=config["batch_size"],
        fp16=config["fp16"],
        eval_accumulation_steps=5,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save evaluation results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)

    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()