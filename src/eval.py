import json
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import numpy as np
import torch
from sklearn.metrics import top_k_accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Flatten predictions and labels
    predictions = predictions.reshape(-1, predictions.shape[-1])
    labels = labels.reshape(-1)

    # Create a mask for valid labels
    valid_mask = labels != -100

    # Apply the mask to filter predictions and labels
    filtered_predictions = predictions[valid_mask]
    filtered_labels = labels[valid_mask]

    # Precompute the range of labels
    label_range = np.arange(predictions.shape[-1])

    # Compute top-k accuracies
    top_5_accuracy = top_k_accuracy_score(filtered_labels, filtered_predictions, k=5, labels=label_range)
    top_25_accuracy = top_k_accuracy_score(filtered_labels, filtered_predictions, k=25, labels=label_range)
    top_100_accuracy = top_k_accuracy_score(filtered_labels, filtered_predictions, k=100, labels=label_range)

    return {
        "top_5_accuracy": top_5_accuracy,
        "top_25_accuracy": top_25_accuracy,
        "top_100_accuracy": top_100_accuracy,
    }

def main():
    model = "lora_r32"

    # Setup paths
    current_dir = Path(__file__).parent
    config = load_config(current_dir / "config.json")
    model_dir = current_dir.parent / "outputs" / model / "final-model"
    output_dir = current_dir.parent / "eval_outputs" / model

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)
        
    # Load pre-split dataset
    eval_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[-20:]", cache_dir="./cache")
    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples.")

    # Tokenize the evaluation dataset
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

    # Initialize Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal language modeling
    )

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