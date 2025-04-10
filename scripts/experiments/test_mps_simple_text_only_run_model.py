import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

print(f"--- MPS Generation Test ---")
print(f"PyTorch version: {torch.__version__}")

# --- Configuration ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PROMPT = "Write a short story about a robot learning to paint:"

# --- Check MPS Availability ---
if not torch.backends.mps.is_available():
    print("ERROR: MPS device not found.")
    exit()
if not torch.backends.mps.is_built():
    print("ERROR: PyTorch was not built with MPS support.")
    exit()

device = torch.device("mps")
print(f"Using device: {device}")

# --- Check MPS Fallback Setting (Informational) ---
print(f"PYTORCH_ENABLE_MPS_FALLBACK set to: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")

# --- Load Tokenizer ---
try:
    print(f"\nLoading tokenizer for {LLM_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    # Ensure pad token is set for generation
    if tokenizer.pad_token is None:
        print("Warning: pad_token not set, setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded.")
except Exception as e:
    print(f"ERROR: Failed to load tokenizer: {e}")
    exit()

# --- Load Model ---
try:
    print(f"\nLoading model {LLM_MODEL_NAME}...")
    # Load model directly onto MPS device
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME).to(device)
    model.eval() # Set to evaluation mode
    print("Model loaded successfully to MPS device.")
    # Optional: Print model memory footprint if needed (requires psutil)
    # try:
    #     import psutil
    #     process = psutil.Process(os.getpid())
    #     mem_info = process.memory_info()
    #     print(f"Approx. memory usage after model load: {mem_info.rss / (1024**3):.2f} GB")
    # except ImportError:
    #     pass
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    exit()

# --- Prepare Inputs ---
try:
    print(f"\nTokenizing prompt: '{TEST_PROMPT}'")
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt", truncation=True, max_length=512).to(device)
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention Mask shape: {inputs['attention_mask'].shape}")
except Exception as e:
    print(f"ERROR: Failed to tokenize input: {e}")
    exit()

# --- Run Generation ---
print("\nAttempting model.generate() on MPS...")
output_ids = None # Initialize to None
start_time = time.time()
try:
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True, # Use sampling for more varied output
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()
    print("model.generate() completed without Python exception.")
    print(f"Generation took {end_time - start_time:.2f} seconds.")

except Exception as e:
    # Catch Python-level exceptions during generate
    end_time = time.time()
    print(f"!!! ERROR: model.generate() raised a Python exception after {end_time - start_time:.2f} seconds: {type(e).__name__} - {e}")
    import traceback
    traceback.print_exc()
    exit() # Exit if Python exception occurs

# --- Decode and Print Output ---
if output_ids is not None:
    try:
        print(f"\nOutput tensor shape: {output_ids.shape}")
        print("Decoding output...")
        # Decode the generated sequence (including the prompt)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\n--- Generated Text ---")
        print(output_text)
        print("--- End Generated Text ---")
        print("\nTest finished successfully.")
    except Exception as e:
        print(f"ERROR: Failed to decode output: {e}")
else:
     print("\nSkipping decoding as generation did not complete or resulted in an error.")

# If the script reaches here without Abort trap 6, MPS generation worked at this basic level.
# If it still crashes with Abort trap 6, the issue lies within the base LLM's generation on MPS.