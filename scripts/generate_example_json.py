# generate_example_json.py
import os
import json
from glob import glob
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel # Using AutoModel is more general for CLIP
import numpy as np

# Configuration
TRAIN_DATA_DIR = "./data/train"
VAL_DATA_DIR = "./data/val"
VISION_MODEL_NAME = "openai/clip-vit-base-patch32" # Match the training script
OUTPUT_INSTRUCTION = "Generate an HLSL shader that reproduces this image"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load CLIP model and processor
print(f"Loading CLIP model: {VISION_MODEL_NAME}...")
try:
    processor = AutoProcessor.from_pretrained(VISION_MODEL_NAME)
    model = AutoModel.from_pretrained(VISION_MODEL_NAME).to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    exit(1)

def generate_clip_embedding(image_path, model, processor, device):
    """Generates CLIP embedding for a given image."""
    try:
        image = Image.open(image_path).convert("RGB")
        # Process image
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Get image features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)

        # Move features to CPU, convert to numpy array, then to list
        embedding = image_features.cpu().numpy().flatten().tolist()
        return embedding
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def process_directory(data_dir):
    """Processes a directory containing .frag and .png pairs."""
    print(f"\nProcessing directory: {data_dir}")
    frag_files = glob(os.path.join(data_dir, "*.frag"))

    if not frag_files:
        print(f"Warning: No .frag files found in {data_dir}")
        return

    processed_count = 0
    skipped_count = 0

    for frag_path in frag_files:
        base_name = os.path.splitext(os.path.basename(frag_path))[0]
        image_filename = f"{base_name}.png"
        image_path = os.path.join(data_dir, image_filename)
        json_output_path = os.path.join(data_dir, f"example_{base_name}.json")

        print(f"  Processing: {base_name}")

        # Check if corresponding PNG exists
        if not os.path.exists(image_path):
            print(f"    Warning: Matching image file not found: {image_path}. Skipping.")
            skipped_count += 1
            continue

        # Read shader code
        try:
            with open(frag_path, 'r', encoding='utf-8') as f:
                shader_code = f.read()
        except Exception as e:
            print(f"    Error reading shader file {frag_path}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Generate CLIP embedding
        print(f"    Generating CLIP embedding for {image_filename}...")
        clip_embedding = generate_clip_embedding(image_path, model, processor, DEVICE)

        if clip_embedding is None:
            print(f"    Error generating CLIP embedding for {image_path}. Skipping.")
            skipped_count += 1
            continue
        print(f"    CLIP embedding generated (dimension: {len(clip_embedding)}).")

        # Create JSON data structure
        example_data = {
            "instruction": OUTPUT_INSTRUCTION,
            "input": image_filename, # Store only the filename as input reference
            "output": shader_code,
            "clip_embeddings": clip_embedding # Add the new embedding property
        }

        # Write JSON file
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(example_data, f, indent=4)
            print(f"    Successfully created JSON: {json_output_path}")
            processed_count += 1
        except Exception as e:
            print(f"    Error writing JSON file {json_output_path}: {e}. Skipping.")
            skipped_count += 1
            continue

    print(f"\nFinished processing {data_dir}.")
    print(f"  Successfully processed: {processed_count} examples.")
    print(f"  Skipped: {skipped_count} examples.")

if __name__ == "__main__":
    # Create directories if they don't exist (optional, good practice)
    os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DATA_DIR, exist_ok=True)

    # Process both training and validation directories
    process_directory(TRAIN_DATA_DIR)
    process_directory(VAL_DATA_DIR)

    print("\nJSON generation complete.")