import wandb

import json
import torch
import random
import logging
import multiprocessing
from pathlib import Path
import numpy as np
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
import torch.multiprocessing as mp
from utils import load_config

# Constants
MAX_PROCESSES = 16
NUM_PROC = 1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_wandb(config: dict):
    """Initialize Weights & Biases for experiment tracking."""
    wandb.init(project='llm-lora-translator', config=config, name=f'lora_r{config["lora_r"]}_final')

def load_model_and_tokenizer(model_name: str):
    """Load the model and tokenizer from the pretrained model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir="./cache",
    )
    logger.info(f'Loaded model: {model.__class__.__name__}')
    logger.info(f"Using device: {model.device}")
    return model, tokenizer

def prepare_dataset(config: dict, tokenizer, num_proc: int, max_position_embeddings: int):
    """Load and preprocess the dataset."""
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split='train', cache_dir="./cache", num_proc=num_proc)
    logger.info(f'Loaded dataset:\n{dataset}')

    if config['num_samples'] is not None:
        dataset = dataset.select(range(config['num_samples']))
    logger.info(f'Selected {len(dataset)} samples')

    dataset = dataset.train_test_split(test_size=config['valid_size'])
    logger.info(f'Split dataset into train and test:\n{dataset}')

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_position_embeddings,
            padding='max_length',
        )

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=dataset['train'].column_names,
        desc='Tokenizing',
    )
    logger.info(f'Tokenized dataset:\n{tokenized_datasets}')
    return tokenized_datasets

def configure_training(config: dict, output_dir: Path):
    """Set up training arguments."""
    return TrainingArguments(
        output_dir=str(output_dir),
        fp16=config['fp16'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        num_train_epochs=config['epochs'],
        warmup_steps=config['warmup_steps'],
        max_grad_norm=config['max_grad_norm'],
        logging_steps=config['logging_steps'],
        eval_steps=config['eval_every'],
        save_steps=config['eval_every'],
        eval_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        report_to='wandb',
        seed=config['seed'],
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )

def main():
    current_dir = Path(__file__).parent
    config = load_config(current_dir / 'config.json')
    output_dir = Path(f'./outputs/lora_r{config["lora_r"]}_final')

    set_seed(config['seed'])
    initialize_wandb(config)

    num_proc = min(multiprocessing.cpu_count() - 1, MAX_PROCESSES)
    logger.info(f'Number of processes: {num_proc}')

    model_name = "Azurro/APT3-1B-Base"
    model, tokenizer = load_model_and_tokenizer(model_name)

    tokenized_datasets = prepare_dataset(config, tokenizer, num_proc, model.config.max_position_embeddings)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias=config['bias'],
        task_type=config['task_type'],
        target_modules=config['target_modules'],
    )

    model = get_peft_model(model, lora_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.info(f"Trainable parameters (%): {trainable_params / total_params * 100:.2f}%")

    training_args = configure_training(config, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()

    # Save final model
    logger.info('Saving final model...')
    trainer.save_model(output_dir / 'final-model')
    model.save_pretrained(output_dir / 'final-model')
    tokenizer.save_pretrained(output_dir / 'final-model')

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f)

    logger.info('Training completed!')

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()