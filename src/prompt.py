import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_dir: str):
    """
    Load the model and tokenizer from a directory containing the final trained model with LoRA.
    """
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

    # Load the LoRA weights
    model = PeftModel.from_pretrained(model, model_dir)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    return model, tokenizer

def generate_response(prompt: str, model, tokenizer, max_length=128, temperature=0.0):
    """
    Generate a response from the model for a given prompt.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(inputs["input_ids"], max_length=max_length, temperature=temperature, do_sample=True)

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model = "lora_r64"

    # Path to the directory where the final model with LoRA weights is saved
    model_dir = f"./outputs/{model}/final-model"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    logger.info(f"Model and tokenizer loaded successfully from {model_dir}")

    # Prompt for evaluation
    prompt = input(">>>")

    # Generate response
    response = generate_response(prompt, model, tokenizer)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    main()