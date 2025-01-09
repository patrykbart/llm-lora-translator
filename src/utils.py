import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_config(config_path: Path) -> dict:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_model_and_tokenizer(model_dir: str):
    """Load the model and tokenizer from a directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    return model, tokenizer 