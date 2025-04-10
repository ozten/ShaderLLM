# run_model.py
import os
import json
# from glob import glob # Not needed for inference unless loading data examples
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
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (MUST match the training configuration for model names) ---
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
# Directory where the trained model weights (.bin file) are saved
# This should match the TrainingArguments output_dir used previously
TRAINED_MODEL_DIR = "./shader_model_with_embeddings" # Corrected dir name based on training script
IMAGE_SIZE = (224, 224) # Standard size for CLIP vision model

# --- Custom Model Definition ---
# This class structure is needed for INFERENCE because it includes
# the vision model and image processor required to handle a raw image input.
# The trained weights loaded later will update llm and vision_projection.
class VisionAugmentedLLM(nn.Module):
    def __init__(self, llm_model_name, vision_model_name):
        super().__init__()
        logger.info(f"Initializing VisionAugmentedLLM for inference:")
        logger.info(f"  LLM: {llm_model_name}")
        logger.info(f"  Vision Encoder: {vision_model_name}")

        # Load vision encoder components
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        logger.info("  Vision model and processor loaded.")

        # Load LLM components
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
             logger.warning("Tokenizer missing pad token, setting to eos_token.")
             self.tokenizer.pad_token = self.tokenizer.eos_token
             # Note: Resizing embeddings shouldn't be necessary if loading trained weights
             # that were trained with this tokenizer state.
             # self.llm.resize_token_embeddings(len(self.tokenizer))
        logger.info("  LLM and tokenizer loaded.")

        # Vision features projection layer
        # Dimensions must match the models used
        vision_dim = self.vision_model.config.projection_dim # e.g., 512 for CLIP base
        llm_dim = self.llm.config.hidden_size # e.g., 896 for Qwen2.5-0.5B
        self.vision_projection = nn.Linear(vision_dim, llm_dim)
        logger.info(f"  Created vision projection layer: {vision_dim} -> {llm_dim}")

        # No need for config delegation or gradient checkpointing methods in inference

    # --- Helper methods to access components ---
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

    # The forward method here is less relevant for typical HF generation,
    # as we'll use the .generate() method of the underlying LLM.


# --- Generation function for inference ---
@torch.no_grad() # Ensure no gradients are computed during inference
def generate_shader_from_image(
    model: VisionAugmentedLLM,
    image_path: str,
    instruction: str = "Generate an HLSL shader that reproduces this image",
    max_new_tokens: int = 1024,
    device: torch.device = torch.device("cpu"),
    generation_dtype: torch.dtype = torch.float32 # Use float32 for MPS compatibility
):
    """
    Generates shader code from an image using the trained VisionAugmentedLLM.

    Args:
        model: The loaded VisionAugmentedLLM instance.
        image_path: Path to the input image file.
        instruction: The instruction text for the model.
        max_new_tokens: Maximum number of tokens to generate for the shader code.
        device: The torch device ('mps', 'cuda', 'cpu').
        generation_dtype: The dtype to use for generation (float32 recommended for MPS).

    Returns:
        The generated shader code as a string.
    """
    model.eval() # Set model to evaluation mode
    # Move the *entire* model to the target device and set dtype
    model.to(device, dtype=generation_dtype)
    logger.info(f"Model moved to {device} with dtype {generation_dtype}")

    # --- Access model components (now on the correct device/dtype) ---
    image_processor = model.get_image_processor()
    vision_model = model.get_vision_model()
    vision_projection = model.get_vision_projection()
    llm = model.get_llm()
    tokenizer = model.get_tokenizer()

    # --- 1. Load and Process Image ---
    try:
        logger.info(f"Loading and processing image from: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE) # Resize to expected input size
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return "[Error: Image file not found]"
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return f"[Error processing image: {e}]"

    # Process image using the image_processor - output will be tensors
    # Ensure tensor is moved to the correct device and dtype
    image_inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = image_inputs['pixel_values'].to(device=device, dtype=generation_dtype)

    # Get image features from the vision model
    # Vision model is already on the correct device/dtype from model.to()
    image_features = vision_model.get_image_features(pixel_values=pixel_values)
    # image_features should have shape [batch_size, vision_dim], e.g., [1, 512]

    # Project image features to the LLM's embedding dimension
    # Projection layer is already on the correct device/dtype
    projected_features = vision_projection(image_features)
    # Add sequence dimension: [1, 1, llm_dim]
    projected_features = projected_features.unsqueeze(1)
    logger.info(f"Image processed and features projected. Shape: {projected_features.shape}")
    #logger.info(f"Instructions: {instruction}")
    #logger.info(f"{os.path.basename(image_path)}")


    # --- 2. Process Text Prompt ---
    # Format the prompt exactly as used during training (but without the response)
    prompt = f"""### Instruction:
{instruction}

### Input Reference:
Image corresponding to embedding for: {os.path.basename(image_path)}

### Response:
""" # Note: Changed Input format slightly to match training dataset example
    # Tokenize the prompt
    logger.info(f"Instructions: {prompt}")
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_attention_mask = prompt_inputs["attention_mask"]
    logger.info(f"Prompt tokenized. Input ID shape: {prompt_input_ids.shape}")


    # --- 3. Get Prompt Embeddings ---
    # Get the LLM's embedding layer (already on correct device/dtype)
    embed_layer = llm.get_input_embeddings()
    # Embed the prompt tokens
    prompt_embeds = embed_layer(prompt_input_ids) # Shape: [1, prompt_seq_len, llm_dim]


    # --- 4. Combine Visual and Text Embeddings ---
    # Concatenate along the sequence dimension (dim=1)
    # Visual token first: [1, 1, dim] + [1, prompt_seq_len, dim] -> [1, 1 + prompt_seq_len, dim]
    combined_embeds = torch.cat([projected_features, prompt_embeds], dim=1)
    logger.info(f"Combined embedding shape: {combined_embeds.shape}")


    # --- 5. Create Combined Attention Mask ---
    # Attention mask for the single visual token (always attend to it)
    visual_attention = torch.ones(projected_features.shape[:2], dtype=torch.long, device=device) # Shape: [1, 1]
    # Concatenate attention masks: [1, 1] + [1, prompt_seq_len] -> [1, 1 + prompt_seq_len]
    combined_attention_mask = torch.cat([visual_attention, prompt_attention_mask], dim=1)
    logger.info(f"Combined attention mask shape: {combined_attention_mask.shape}")


    # --- 6. Generate Text using LLM ---
    logger.info(f"Starting generation with max_new_tokens={max_new_tokens}...")
    # Use the llm's generate method with the combined embeddings and mask
    output_ids = llm.generate(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,      # Enable sampling
        temperature=0.6,     # Control randomness (lower = less random)
        top_p=0.9,           # Nucleus sampling
        num_return_sequences=1, # Generate one sequence
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id # Stop generation at EOS
    )
    logger.info("Generation complete.")


    # --- 7. Decode the Output ---
    logger.info("Decoding output...")
    # output_ids contains the full sequence (prompt + generated)
    # Decode the generated part by slicing off the prompt tokens
    # Note: output_ids shape is [batch_size, total_sequence_length]
    prompt_length = prompt_input_ids.shape[1] # Length of the input prompt tokens
    # Add 1 for the visual token that was prepended
    effective_prompt_length = 1 + prompt_length
    # Slice the output_ids to get only the generated token IDs
    generated_ids = output_ids[0, effective_prompt_length:]

    # Decode the generated IDs into text
    shader_code = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Alternative decoding (decode full, then strip prompt - sometimes simpler)
    # full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # response_marker = "### Response:\n"
    # response_start_index = full_output_text.find(response_marker)
    # if response_start_index != -1:
    #    shader_code = full_output_text[response_start_index + len(response_marker):].strip()
    # else:
        # Fallback: If marker somehow missing, try basic stripping (less reliable)
    #    logger.warning("Response marker '### Response:\n' not found in generated text. Using fallback decoding.")
    #    shader_code = full_output_text[len(prompt):].strip() # Approximate

    logger.info("Decoding complete.")
    return shader_code.strip()

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Determine Device ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        generation_dtype = torch.float32 # MPS often prefers float32
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        generation_dtype = torch.float16 # CUDA can usually handle float16/bfloat16
    else:
        device = torch.device("cpu")
        generation_dtype = torch.float32

    device = torch.device("cpu")
    logger.info(f"Using device: {device} with dtype: {generation_dtype}")

    # --- Instantiate the Model Structure ---
    # We instantiate the full structure including vision parts for inference.
    logger.info(f"Instantiating model structure: VisionAugmentedLLM")
    model_structure = VisionAugmentedLLM(LLM_MODEL_NAME, VISION_MODEL_NAME)

    # --- Load the Trained Weights ---
    # Expecting pytorch_model.bin due to previous saving issues with safetensors
    weights_filename = "pytorch_model.bin"
    weights_path = os.path.join(TRAINED_MODEL_DIR, weights_filename)

    if not os.path.exists(weights_path):
        logger.error(f"Model weights not found at {weights_path}.")
        logger.error("Ensure the TRAINED_MODEL_DIR is correct and training was successful.")
        exit(1) # Exit if weights are missing

    try:
        logger.info(f"Loading state dictionary from: {weights_path}")
        # Load weights onto CPU first to avoid potential device mismatches
        state_dict = torch.load(weights_path, map_location="cpu")

        # Load the state dict into the model structure
        # Set strict=False because the state_dict from training (EmbeddingAugmentedLLM)
        # will NOT contain weights for the vision_model.* parts. This is expected.
        # The vision_model parts will keep their pre-trained weights.
        missing_keys, unexpected_keys = model_structure.load_state_dict(state_dict, strict=False)

        logger.info("State dictionary loaded.")
        if any("vision_model." in k for k in missing_keys):
             logger.info("Missing keys related to 'vision_model.*' are expected (using pre-trained weights).")
             # Optionally filter vision model keys from missing_keys for cleaner logging
             missing_keys = [k for k in missing_keys if "vision_model." not in k]
        if missing_keys:
            logger.warning(f"Unexpected missing keys after loading state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys found in state_dict: {unexpected_keys}")

    except Exception as e:
        logger.exception(f"Failed to load model weights from {weights_path}: {e}")
        exit(1)

    # --- Prepare for Inference ---
    # Select a test image
    # Make sure this path exists relative to where you run the script
    # Example: if run from root, path should be relative to root
    # Example: if run from scripts/, path might need "../data/..."
    image_file_to_test = "data/val/080.png" # Example path, adjust as necessary

    if not os.path.exists(image_file_to_test):
         logger.error(f"Test image file not found: {image_file_to_test}")
         logger.error("Please provide a valid path to an image file.")
    else:
        logger.info(f"--- Running Inference for: {image_file_to_test} ---")
        # Call the generation function
        generated_code = generate_shader_from_image(
            model=model_structure,
            image_path=image_file_to_test,
            device=device,
            generation_dtype=generation_dtype,
            max_new_tokens=512 # Adjust token limit as needed
        )

        print("\n" + "="*30)
        print("   Generated Shader Code")
        print("="*30)
        print(generated_code)
        print("="*30)

    logger.info("Inference script finished.")