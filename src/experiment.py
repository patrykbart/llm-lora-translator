import torch
import logging
import multiprocessing
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    model_name = "Azurro/APT3-1B-Base"
    logger.info(f'Model ID: {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    logger.info(f'Loaded model: {model.__class__.__name__}')

    dataset = load_dataset("wikitext", "wikitext-2-v1")
    logger.info(f'Loaded dataset:\n{dataset}')

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model.config.max_position_embeddings,
            padding="max_length",
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=min(multiprocessing.cpu_count() - 1, 32),
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    logger.info(f'Tokenized dataset:\n{tokenized_datasets}')

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    model = get_peft_model(model, lora_config)
    logger.info(f"Trainable parameters: {model.print_trainable_parameters()}")

    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1000,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=10_000,
        save_steps=10_000,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    trainer.train()