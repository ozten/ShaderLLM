import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

print(f"--- MPS Generation Test with Text Embeddings ---")
print(f"PyTorch version: {torch.__version__}")

# --- Configuration ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEST_PROMPT = "Explain the concept of photosynthesis in simple terms:"

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
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    exit()

# --- Prepare Inputs (Tokenize and Get Embeddings) ---
text_embeddings = None
attention_mask = None
try:
    print(f"\nTokenizing prompt: '{TEST_PROMPT}'")
    inputs = tokenizer(TEST_PROMPT, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"] # Keep the original attention mask

    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")

    print("\nGetting embeddings from input_ids...")
    embed_layer = model.get_input_embeddings()
    with torch.no_grad(): # No need for gradients here
        text_embeddings = embed_layer(input_ids)
    print(f"Text embeddings shape: {text_embeddings.shape}")

except Exception as e:
    print(f"ERROR: Failed to prepare inputs: {e}")
    exit()

# --- Run Generation using inputs_embeds ---
print("\nAttempting model.generate() using inputs_embeds on MPS...")
output_ids = None
start_time = time.time()
try:
    if text_embeddings is not None and attention_mask is not None:
        with torch.no_grad():
            output_ids = model.generate(
                inputs_embeds=text_embeddings,      # Use embeddings instead of input_ids
                attention_mask=attention_mask,      # Use the corresponding attention mask
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        end_time = time.time()
        print("model.generate() using inputs_embeds completed without Python exception.")
        print(f"Generation took {end_time - start_time:.2f} seconds.")
    else:
        print("Skipping generation because input preparation failed.")

except Exception as e:
    # Catch Python-level exceptions during generate
    end_time = time.time()
    print(f"!!! ERROR: model.generate() raised a Python exception after {end_time - start_time:.2f} seconds: {type(e).__name__} - {e}")
    import traceback
    traceback.print_exc()
    # It might still crash with Abort trap 6 before reaching here
    exit()

# --- Decode and Print Output ---
if output_ids is not None:
    try:
        print(f"\nOutput tensor shape: {output_ids.shape}")
        print("Decoding output...")
        # Decode the generated sequence
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\n--- Generated Text ---")
        print(output_text)
        print("--- End Generated Text ---")
        print("\nTest finished successfully.")
    except Exception as e:
        print(f"ERROR: Failed to decode output: {e}")
else:
     print("\nSkipping decoding as generation did not complete or resulted in an error.")

# Outcome Interpretation:
# Success: Using inputs_embeds derived solely from text works on MPS.
#          The problem likely lies in the concatenation or the specific visual embedding.
# Crash (Abort trap 6): Using inputs_embeds for generate is fundamentally problematic on MPS for this model/setup.