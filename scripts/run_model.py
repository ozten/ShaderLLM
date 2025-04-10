# run_model.py  <- Renamed from train_with_vision.py in the provided text
import os
import json
from glob import glob
import torch
from PIL import Image
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    # TrainingArguments, Trainer # Not needed for inference
)
# from datasets import Dataset # Not needed for inference
from torch import nn

# Configuration (MUST match the training configuration)
#  or  "Qwen/Qwen2.5-1.5B-Instruct"  too big w/o some tweaks -> "Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Coder model
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP vision encoder
OUTPUT_DIR = "./shader_model"
# TRAIN_DATA_DIR = "./data/train" # Not needed for inference
# VAL_DATA_DIR = "./data/val" # Not needed for inference
# MAX_LENGTH = 2048 # Define if needed for generation/tokenizer
IMAGE_SIZE = (224, 224)  # Standard size for vision models

# --- Custom model definition needs to be here or imported ---
class VisionAugmentedLLM(nn.Module):
    def __init__(self, llm_model_name, vision_model_name): # Removed num_visual_tokens as it's implicitly 1 now
        super().__init__()
        # Load vision encoder
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)

        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
             print("Tokenizer missing pad token, setting to eos_token.")
             self.tokenizer.pad_token = self.tokenizer.eos_token
             # Note: Resizing embeddings might be needed if truly adding a token
             # self.llm.resize_token_embeddings(len(self.tokenizer))

        # Vision features projection
        vision_dim = self.vision_model.config.projection_dim
        llm_dim = self.llm.config.hidden_size

        # Project vision features to token space
        self.vision_projection = nn.Linear(vision_dim, llm_dim) # Project CLIP output (512) to LLM dim

        # No need to delegate config or gradient checkpointing methods for inference

    # --- FORWARD METHOD FOR INFERENCE (Simplified/Modified) ---
    # Note: The forward used during training is complex. For generation,
    # we usually just need the core components or use the .generate method.
    # We'll define a specific generation function instead of relying on forward.

    # We need access to components for the generation function
    def get_llm(self):
        return self.llm

    def get_tokenizer(self):
        return self.tokenizer

    def get_image_processor(self):
        return self.image_processor

    def get_vision_model(self):
        return self.vision_model

    def get_vision_projection(self):
        return self.vision_projection


# --- ADJUSTED Generation function for inference ---
@torch.no_grad()
def generate_shader_from_image(model, image_path, instruction="Generate an HLSL shader that reproduces this image", max_new_tokens=1024, device="cpu"):
    model.eval() # Set model to evaluation mode
    # Move model to device AND set dtype
    model.to(device, dtype=torch.float32) # Force float32

    # --- Access components (they are now on device and float32) ---
    image_processor = model.get_image_processor()
    vision_model = model.get_vision_model() # Already moved by model.to()
    vision_projection = model.get_vision_projection() # Already moved
    llm = model.get_llm() # Already moved
    tokenizer = model.get_tokenizer()

    # --- 1. Process Image ---
    # ---> ADDED BACK THE MISSING IMAGE LOADING <---
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    # ---> END ADDED BACK <---

    # Process image - ensure inputs match the model dtype
    image_inputs = image_processor(images=image, return_tensors="pt").to(device)
    pixel_values = image_inputs['pixel_values'].to(dtype=torch.float32) # Match dtype

    # Get image features and project them
    image_features = vision_model.get_image_features(pixel_values=pixel_values)
    projected_features = vision_projection(image_features)
    projected_features = projected_features.unsqueeze(1) # [1, 1, llm_dim]

    # --- 2. Process Text Prompt ---
    prompt = f"""### Instruction:
{instruction}

### Input:
Reference image provided

### Response:
"""
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attention_mask = prompt_inputs["attention_mask"]

    embed_layer = llm.get_input_embeddings()
    prompt_embeds = embed_layer(prompt_input_ids) # Embeddings will be float32

    # --- 3. Combine Embeddings ---
    combined_embeds = torch.cat([projected_features, prompt_embeds], dim=1)

    # --- 4. Create Combined Attention Mask ---
    visual_attention = torch.ones(projected_features.shape[:2], device=device)
    combined_attention_mask = torch.cat([visual_attention, prompt_attention_mask], dim=1)

    # --- 5. Generate ---
    print("Generating text...")
    output_ids = llm.generate(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    print("Generation complete.")

    # --- 6. Decode ---
    print("Decoding output...")
    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    response_marker = "### Response:\n"
    response_start_index = full_output_text.find(response_marker)
    if response_start_index != -1:
        shader_code = full_output_text[response_start_index + len(response_marker):].strip()
    else:
        # Fallback if marker not found
        prompt_only_decoded = tokenizer.decode(prompt_input_ids[0], skip_special_tokens=True)
        # This fallback is approximate
        prompt_end_index = len(prompt_only_decoded)
        shader_code = full_output_text[prompt_end_index:].strip() # May include part of the prompt if generation repeated it

    print("Decoding complete.")
    return shader_code

# Make sure the rest of your run_model.py script uses this corrected function.
# The main loading part should look like this:

if __name__ == "__main__":
    # --- Manual Loading ---
    print(f"Loading model components for inference...")
    model = VisionAugmentedLLM(LLM_MODEL_NAME, VISION_MODEL_NAME)
    weights_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}.")
    print(f"Loading state dict from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Model weights loaded successfully.")

    # --- Set device for inference ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # Uncomment to force CPU for testing
    print(f"Using device: {device}")

    # --- Run Inference ---
    image_file = "data/val/080.png" # Adjust path as needed
    if not os.path.exists(image_file):
         print(f"Warning: Test image {image_file} not found. Skipping generation.")
    else:
        print(f"Generating shader for image: {image_file}...")
        # Use the corrected generate_shader_from_image function
        shader_code = generate_shader_from_image(model, image_file, device=device)
        print("\n--- Generated Shader Code ---")
        print(shader_code)
        print("--- End Generated Shader Code ---")