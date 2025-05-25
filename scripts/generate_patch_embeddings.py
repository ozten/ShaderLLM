# generate_patch_embeddings.py
import os
import json
from glob import glob
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TRAIN_DATA_DIR = "./data/train"
VAL_DATA_DIR = "./data/val"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"
OUTPUT_INSTRUCTION = "Generate an HLSL shader that reproduces this image"
DEVICE = "cpu"  # Force CPU for macOS compatibility

logger.info(f"Using device: {DEVICE}")

# Load CLIP model and processor
logger.info(f"Loading CLIP model: {VISION_MODEL_NAME}...")
try:
    processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
    model = AutoModel.from_pretrained(VISION_MODEL_NAME).to(DEVICE)
    model.eval()
    logger.info("CLIP model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading CLIP model: {e}")
    exit(1)

def generate_clip_patch_embeddings(image_path, model, processor, device):
    """
    Generates CLIP patch embeddings for a given image.
    Returns patch features instead of global image features.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Get patch features from vision model
        with torch.no_grad():
            # Get the vision model outputs (includes patch embeddings)
            vision_outputs = model.vision_model(**inputs)
            
            # Extract patch embeddings (exclude CLS token)
            # vision_outputs.last_hidden_state has shape [1, num_patches + 1, hidden_dim]
            # We exclude the first token (CLS token) to get only patch tokens
            patch_embeddings = vision_outputs.last_hidden_state[:, 1:, :]  # [1, 49, 768] for ViT-B/32
            
            # Move to CPU and convert to numpy
            patch_embeddings = patch_embeddings.cpu().numpy()
            
            # Remove batch dimension and convert to list
            patch_embeddings = patch_embeddings[0].tolist()  # [49, 768]
            
            logger.debug(f"Generated patch embeddings with shape: {len(patch_embeddings)} patches x {len(patch_embeddings[0])} dim")
            
        return patch_embeddings
        
    except FileNotFoundError:
        logger.error(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None

def process_directory(data_dir):
    """Processes a directory containing .frag and .png pairs, generating patch embeddings."""
    logger.info(f"\nProcessing directory: {data_dir}")
    frag_files = glob(os.path.join(data_dir, "*.frag"))

    if not frag_files:
        logger.warning(f"Warning: No .frag files found in {data_dir}")
        return

    processed_count = 0
    skipped_count = 0

    for frag_path in frag_files:
        base_name = os.path.splitext(os.path.basename(frag_path))[0]
        image_filename = f"{base_name}.png"
        image_path = os.path.join(data_dir, image_filename)
        json_output_path = os.path.join(data_dir, f"example_patch_{base_name}.json")

        logger.info(f"  Processing: {base_name}")

        # Check if corresponding PNG exists
        if not os.path.exists(image_path):
            logger.warning(f"    Warning: Matching image file not found: {image_path}. Skipping.")
            skipped_count += 1
            continue

        # Check if patch embeddings JSON already exists and is newer than image
        if os.path.exists(json_output_path):
            json_mtime = os.path.getmtime(json_output_path)
            image_mtime = os.path.getmtime(image_path)
            frag_mtime = os.path.getmtime(frag_path)
            
            if json_mtime > max(image_mtime, frag_mtime):
                logger.info(f"    Patch embeddings already exist and are up-to-date: {json_output_path}")
                processed_count += 1
                continue

        # Read shader code
        try:
            with open(frag_path, 'r', encoding='utf-8') as f:
                shader_code = f.read()
        except Exception as e:
            logger.error(f"    Error reading shader file {frag_path}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Generate CLIP patch embeddings
        logger.info(f"    Generating CLIP patch embeddings for {image_filename}...")
        clip_patch_embeddings = generate_clip_patch_embeddings(image_path, model, processor, DEVICE)

        if clip_patch_embeddings is None:
            logger.error(f"    Error generating CLIP patch embeddings for {image_path}. Skipping.")
            skipped_count += 1
            continue
            
        num_patches = len(clip_patch_embeddings)
        patch_dim = len(clip_patch_embeddings[0]) if clip_patch_embeddings else 0
        logger.info(f"    CLIP patch embeddings generated: {num_patches} patches x {patch_dim} dimensions")

        # Create JSON data structure
        example_data = {
            "instruction": OUTPUT_INSTRUCTION,
            "input": image_filename,
            "output": shader_code,
            "clip_patch_embeddings": clip_patch_embeddings,
            "num_patches": num_patches,
            "patch_dim": patch_dim
        }

        # Write JSON file
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, indent=2)  # Less indentation to save space
            logger.info(f"    Successfully created patch embeddings JSON: {json_output_path}")
            processed_count += 1
        except Exception as e:
            logger.error(f"    Error writing JSON file {json_output_path}: {e}. Skipping.")
            skipped_count += 1
            continue

    logger.info(f"\nFinished processing {data_dir}.")
    logger.info(f"  Successfully processed: {processed_count} examples.")
    logger.info(f"  Skipped: {skipped_count} examples.")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DATA_DIR, exist_ok=True)

    # Process both training and validation directories
    process_directory(TRAIN_DATA_DIR)
    process_directory(VAL_DATA_DIR)

    logger.info("\nPatch embedding generation complete.")
    logger.info("Use the generated example_patch_*.json files with the updated training script.")