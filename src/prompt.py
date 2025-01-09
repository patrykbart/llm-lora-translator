import logging
from utils import load_model_and_tokenizer

# Constants
DEFAULT_MAX_LENGTH = 128
DEFAULT_TEMPERATURE = 0.0

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_response(prompt: str, model, tokenizer, max_length=DEFAULT_MAX_LENGTH, temperature=DEFAULT_TEMPERATURE):
    """
    Generate a response from the model for a given prompt.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        temperature=temperature, 
        do_sample=True
    )

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_name = "lora_r64"

    # Path to the directory where the final model with LoRA weights is saved
    model_dir = f"./outputs/{model_name}/final-model"

    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_dir)
    logger.info(f"Model and tokenizer loaded successfully from {model_dir}")

    # Prompt for evaluation
    prompt = input(">>> ")

    # Generate response
    response = generate_response(prompt, model, tokenizer)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    main()