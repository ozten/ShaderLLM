# train_with_vision.py
import os
import json
from glob import glob
import torch
# from PIL import Image # No longer needed here if using pre-computed embeddings
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # AutoImageProcessor, # No longer needed for training if embeddings are pre-computed
    # AutoModel, # No longer needed for training if embeddings are pre-computed
    TrainingArguments,
    Trainer
)
# from datasets import Dataset # Not used directly
from torch import nn
import logging # Added for better logging

import evaluate # <-- Import evaluate
import numpy as np
import nltk # <-- Often needed for rouge/bleu tokenization

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Coder model
# VISION_MODEL_NAME = "openai/clip-vit-base-patch32" # Still needed to know the embedding dimension
CLIP_EMBEDDING_DIM = 512 # Standard dimension for openai/clip-vit-base-patch32 image features
OUTPUT_DIR = "./shader_model_with_embeddings" # Changed output dir name
TRAIN_DATA_DIR = "./data/train"
VAL_DATA_DIR = "./data/val"
MAX_LENGTH = 2048
# IMAGE_SIZE = (224, 224) # No longer needed for training


# --- Define Compute Metrics ---
# Ensure nltk punkt tokenizer is downloaded (needed for ROUGE)
try:
    # Check if 'punkt' is already available
    nltk.data.find('tokenizers/punkt')
    logger.info("NLTK 'punkt' resource found.")
except LookupError:
    # If not found, download it
    logger.warning("NLTK 'punkt' resource not found. Attempting download...")
    try:
        nltk.download('punkt', quiet=True)
        logger.info("NLTK 'punkt' downloaded successfully.")
        # Verify download (optional but good practice)
        nltk.data.find('tokenizers/punkt')
    except Exception as download_error: # Catch potential download errors (network, permissions, etc.)
        logger.error(f"Failed to download NLTK 'punkt' resource: {download_error}")
        logger.error("Please try running 'python -m nltk.downloader punkt' manually.")
        # Depending on your needs, you might want to exit or raise an error here
        # For now, we'll just log the error and proceed, ROUGE might fail later.
        # raise RuntimeError("Failed to download NLTK 'punkt', cannot compute ROUGE.") from download_error

# Load the ROUGE metric calculator
rouge_metric = evaluate.load("rouge")


# --- Model Definition ---
# Enhanced PatchVisionAugmentedLLM with patch-based features
class PatchEmbeddingAugmentedLLM(nn.Module):
    def __init__(self, llm_model_name, clip_patch_dim=768, num_patches=49): # ViT-B/32 has 49 patches (7x7)
        super().__init__()
        logger.info(f"Initializing PatchEmbeddingAugmentedLLM with LLM: {llm_model_name}")
        logger.info(f"Using patch-based vision with {num_patches} patches of dimension {clip_patch_dim}")
        
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
             logger.warning("Tokenizer missing pad token, setting to eos_token.")
             self.tokenizer.pad_token = self.tokenizer.eos_token

        # Vision features projection for patch-based features
        llm_dim = self.llm.config.hidden_size
        self.clip_patch_dim = clip_patch_dim
        self.num_patches = num_patches

        # Project CLIP patch features (768 for ViT-B/32 patches) to LLM dim
        # This handles a sequence of patch embeddings
        self.patch_projection = nn.Linear(self.clip_patch_dim, llm_dim)
        logger.info(f"Created patch projection layer: {self.clip_patch_dim} -> {llm_dim}")
        
        # Optional: Learnable patch position embeddings
        self.patch_position_embeddings = nn.Embedding(self.num_patches, llm_dim)
        logger.info(f"Created position embeddings for {self.num_patches} patches")

        # --- Delegate config for Trainer compatibility ---
        self.config = self.llm.config

    # --- Delegate gradient checkpointing methods ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing for LLM...")
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            logger.warning("Underlying LLM does not support gradient_checkpointing_enable.")

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            logger.info("Disabling gradient checkpointing for LLM...")
            self.llm.gradient_checkpointing_disable()
        else:
            logger.warning("Underlying LLM does not support gradient_checkpointing_disable.")

    # Modified forward to accept 'clip_patch_embeddings' for patch-based features
    def forward(self, input_ids, attention_mask, clip_patch_embeddings=None, labels=None):
        batch_size = input_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()
        llm_device = embed_layer.weight.device # Get device from LLM embedding layer

        if clip_patch_embeddings is not None:
            # Ensure embeddings are float and on the correct device
            # Expected shape: [batch_size, num_patches, clip_patch_dim]
            if len(clip_patch_embeddings.shape) != 3:
                raise ValueError(f"Input clip_patch_embeddings must be 3D [batch, patches, dim], got shape {clip_patch_embeddings.shape}")
            if clip_patch_embeddings.shape[1] != self.num_patches:
                raise ValueError(f"Input patch count ({clip_patch_embeddings.shape[1]}) does not match expected ({self.num_patches})")
            if clip_patch_embeddings.shape[2] != self.clip_patch_dim:
                raise ValueError(f"Input patch dim ({clip_patch_embeddings.shape[2]}) does not match expected dim ({self.clip_patch_dim})")

            clip_patch_embeddings = clip_patch_embeddings.to(dtype=self.patch_projection.weight.dtype, device=llm_device)

            # Project patch features to LLM embedding dimension
            # Shape: [batch_size, num_patches, llm_dim]
            projected_patches = self.patch_projection(clip_patch_embeddings)
            
            # Add position embeddings to each patch
            position_ids = torch.arange(self.num_patches, device=llm_device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.patch_position_embeddings(position_ids)
            projected_features = projected_patches + position_embeds

            # Get text embeddings
            input_ids = input_ids.to(llm_device)
            inputs_embeds = embed_layer(input_ids) # Shape: [batch_size, seq_len, llm_dim]

            # Combine patch features and text embeddings
            # Shape: [batch_size, num_patches + seq_len, llm_dim]
            combined_embeds = torch.cat([projected_features, inputs_embeds], dim=1)

            # Extend attention mask for all patches
            attention_mask = attention_mask.to(llm_device)
            patch_attention = torch.ones(batch_size, self.num_patches, dtype=attention_mask.dtype, device=llm_device)
            # Shape: [batch_size, num_patches + seq_len]
            new_attention_mask = torch.cat([patch_attention, attention_mask], dim=1)

            # Adjust labels if provided (ignore all patch tokens in loss)
            if labels is not None:
                labels = labels.to(llm_device)
                # Shape: [batch_size, num_patches] filled with -100
                patch_labels = torch.full((batch_size, self.num_patches), -100, dtype=torch.long, device=llm_device)
                # Shape: [batch_size, num_patches + seq_len]
                new_labels = torch.cat([patch_labels, labels], dim=1)
            else:
                new_labels = None

            # --- Label Padding/Truncation (Important!) ---
            # Ensure labels match the final sequence length after prepending patch tokens
            expected_seq_len = combined_embeds.shape[1]
            if new_labels is not None and new_labels.shape[1] != expected_seq_len:
                 current_len = new_labels.shape[1]
                 # Pad with -100 if new_labels is shorter
                 if current_len < expected_seq_len:
                     pad_len = expected_seq_len - current_len
                     padding = torch.full((batch_size, pad_len), -100, dtype=torch.long, device=llm_device)
                     new_labels = torch.cat([new_labels, padding], dim=1)
                 # Truncate if new_labels is longer
                 elif current_len > expected_seq_len:
                     new_labels = new_labels[:, :expected_seq_len]

            if new_attention_mask.shape[1] != expected_seq_len:
                 raise ValueError(f"Attention mask length mismatch: Expected {expected_seq_len}, got {new_attention_mask.shape[1]}")
            if new_labels is not None and new_labels.shape[1] != expected_seq_len:
                 raise ValueError(f"Label length mismatch after adjustment: Expected {expected_seq_len}, got {new_labels.shape[1]}")

            # Forward pass using inputs_embeds
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                use_cache=not self.training or not (hasattr(self.llm, 'is_gradient_checkpointing') and self.llm.is_gradient_checkpointing) # Check attribute exists
            )
            return outputs
        else:
            # Text-only forward pass (ensure gradient checkpointing status is respected)
             logger.warning("Performing text-only forward pass (no clip_patch_embeddings provided).")
             input_ids = input_ids.to(llm_device)
             attention_mask = attention_mask.to(llm_device)
             labels = labels.to(llm_device) if labels is not None else None
             return self.llm(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 labels=labels,
                 use_cache=not self.training or not (hasattr(self.llm, 'is_gradient_checkpointing') and self.llm.is_gradient_checkpointing) # Check attribute exists
             )


# --- Data Preparation ---
# Modified data loading function for patch embeddings
def load_examples_with_patch_embeddings(data_dir):
    examples = []
    required_keys = ["instruction", "input", "output", "clip_patch_embeddings"]
    # Look for patch embedding files first, fall back to regular embeddings
    json_files = glob(f"{data_dir}/example_patch_*.json")
    if not json_files:
        logger.warning(f"No patch embedding files found, falling back to regular embeddings")
        json_files = glob(f"{data_dir}/example_*.json")
    
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

    skipped_count = 0
    loaded_count = 0
    
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                example = json.load(f)

            # Check if this is a patch embedding file or regular embedding file
            is_patch_file = "clip_patch_embeddings" in example
            is_regular_file = "clip_embeddings" in example
            
            if is_patch_file:
                # Validate patch embedding format
                if not all(key in example for key in required_keys):
                    logger.warning(f"Skipping {json_path}: Missing required keys for patch embeddings.")
                    skipped_count += 1
                    continue

                patch_embeddings = example["clip_patch_embeddings"]
                if not isinstance(patch_embeddings, list) or not patch_embeddings:
                    logger.warning(f"Skipping {json_path}: 'clip_patch_embeddings' is not a non-empty list.")
                    skipped_count += 1
                    continue

                # Validate patch structure
                if not all(isinstance(patch, list) for patch in patch_embeddings):
                    logger.warning(f"Skipping {json_path}: Invalid patch embedding structure.")
                    skipped_count += 1
                    continue

                # Check dimensions
                expected_patches = 49  # For ViT-B/32
                expected_dim = 768
                
                if len(patch_embeddings) != expected_patches:
                    logger.warning(f"Skipping {json_path}: Expected {expected_patches} patches, got {len(patch_embeddings)}")
                    skipped_count += 1
                    continue
                    
                if patch_embeddings and len(patch_embeddings[0]) != expected_dim:
                    logger.warning(f"Skipping {json_path}: Expected patch dim {expected_dim}, got {len(patch_embeddings[0])}")
                    skipped_count += 1
                    continue

                examples.append(example)
                loaded_count += 1
                
            elif is_regular_file:
                # Convert regular embeddings to patch format (duplicate single embedding)
                logger.info(f"Converting regular embedding to patch format for {json_path}")
                
                regular_embedding = example["clip_embeddings"]
                if len(regular_embedding) != CLIP_EMBEDDING_DIM:
                    logger.warning(f"Skipping {json_path}: Wrong embedding dimension")
                    skipped_count += 1
                    continue
                
                # Create fake patch embeddings by duplicating the global embedding
                # This is not ideal but allows backward compatibility
                fake_patch_embeddings = [regular_embedding] * 49  # Duplicate for 49 patches
                
                # Create new example with patch format
                patch_example = example.copy()
                patch_example["clip_patch_embeddings"] = fake_patch_embeddings
                patch_example["num_patches"] = 49
                patch_example["patch_dim"] = CLIP_EMBEDDING_DIM
                
                examples.append(patch_example)
                loaded_count += 1
                
            else:
                logger.warning(f"Skipping {json_path}: No valid embedding format found")
                skipped_count += 1

        except json.JSONDecodeError:
            logger.error(f"Skipping {json_path}: Invalid JSON.")
            skipped_count += 1
        except Exception as e:
            logger.error(f"Error loading example from {json_path}: {e}")
            skipped_count += 1

    logger.info(f"Loaded {loaded_count} valid examples with patch embeddings from {data_dir}. Skipped {skipped_count}.")
    return examples

# Modified Dataset to use patch embeddings
class ShaderPatchEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format prompt (using the 'input' field which contains the image filename)
        image_ref = example['input'] # Should be like "002.png"
        prompt = f"""### Instruction:
{example['instruction']}

### Input Reference:
Image corresponding to patch embeddings for: {image_ref}

### Response:
"""
        response = example['output']

        # Tokenize text (prompt + response)
        # Account for patch tokens taking up sequence space
        tokenized_input = self.tokenizer(
            prompt + response + self.tokenizer.eos_token,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Create labels: -100 for prompt tokens to ignore them in loss
        labels = tokenized_input["input_ids"].clone().squeeze()

        # Tokenize prompt separately to find length accurately
        tokenized_prompt_only = self.tokenizer(
            prompt,
            add_special_tokens=True  # Match the full tokenization
        )["input_ids"]
        prompt_token_length = len(tokenized_prompt_only)

        # Mask prompt tokens
        labels[:prompt_token_length] = -100

        # Handle padding
        input_ids_squeezed = tokenized_input["input_ids"].squeeze()
        pad_token_indices = (input_ids_squeezed == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        first_pad_index = pad_token_indices[0].item() if len(pad_token_indices) > 0 else len(labels)
        if first_pad_index < len(labels):
             labels[first_pad_index:] = -100

        # Load pre-computed patch embeddings and convert to tensor
        clip_patch_embeddings = example["clip_patch_embeddings"]
        # Convert list of lists to tensor: [num_patches, patch_dim]
        clip_patch_tensor = torch.tensor(clip_patch_embeddings, dtype=torch.float)

        return {
            "input_ids": input_ids_squeezed,
            "attention_mask": tokenized_input["attention_mask"].squeeze(),
            "labels": labels,
            "clip_patch_embeddings": clip_patch_tensor  # [49, 768] for ViT-B/32
        }

# Modified data collator for patch embeddings
def patch_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    # Stack patch embeddings: [batch_size, num_patches, patch_dim]
    clip_patch_embeddings = torch.stack([item["clip_patch_embeddings"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "clip_patch_embeddings": clip_patch_embeddings  # Pass patch embeddings in the batch
    }

# Modified Trainer for patch embeddings
class PatchEmbeddingLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract clip_patch_embeddings instead of clip_embeddings
        clip_patch_embeddings = inputs.pop("clip_patch_embeddings", None)
        labels = inputs.get("labels") # Keep labels in inputs for the model

        if clip_patch_embeddings is not None:
            # Ensure patch embeddings are on the correct device (matching the model)
            if list(model.parameters()):
                model_device = next(model.parameters()).device
                clip_patch_embeddings = clip_patch_embeddings.to(model_device)
            else:
                logger.warning("Model has no parameters, cannot determine device for patch embeddings.")
                clip_patch_embeddings = clip_patch_embeddings.to('cpu')

            # Forward pass with patch embeddings
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                clip_patch_embeddings=clip_patch_embeddings  # Pass patch embeddings to the model
            )
        else:
            # Text-only forward pass (should ideally not happen if data is prepared correctly)
            logger.warning("Trainer compute_loss: clip_patch_embeddings not found in inputs. Performing text-only forward pass.")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

        loss = outputs.loss if hasattr(outputs, "loss") else None
        if loss is None:
             logger.error("Model output did not contain 'loss'.")
             raise ValueError("Model output missing 'loss' attribute.")

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_preds):
    """Computes ROUGE scores and shader-specific metrics for evaluation."""
    preds, labels = eval_preds
    # preds are logits, labels are token ids

    # Decode predictions
    # Replace -100 in labels as tokenizer cannot decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Get token ids from logits using argmax
    pred_ids = np.argmax(preds[0] if isinstance(preds, tuple) else preds, axis=-1) # preds might be tuple (logits, ...)

    # Decode tokens to text
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects newline after each sentence
    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge_metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)

    # Extract specific ROUGE scores
    result = {key: value * 100 for key, value in result.items()} # Scale to 0-100

    # Add prediction lengths (optional but useful)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_ids]
    result["gen_len"] = np.mean(prediction_lens)

    # Add shader-specific metrics
    try:
        from evaluation_utils import validate_shader_compilation
        
        # Compute compilation rate
        compilation_successes = 0
        total_predictions = len(decoded_preds)
        
        for pred in decoded_preds:
            if pred.strip():  # Only evaluate non-empty predictions
                validation_result = validate_shader_compilation(pred)
                if validation_result["valid"]:
                    compilation_successes += 1
        
        result["compilation_rate"] = (compilation_successes / total_predictions * 100) if total_predictions > 0 else 0.0
        
    except ImportError:
        logger.warning("evaluation_utils not available, skipping shader compilation metrics")
        result["compilation_rate"] = 0.0
    except Exception as e:
        logger.warning(f"Error computing compilation metrics: {e}")
        result["compilation_rate"] = 0.0

    return {k: round(v, 4) for k, v in result.items()} # Round for cleaner logging





# --- Main Training Script ---
if __name__ == "__main__":
    # Load data with patch embeddings
    logger.info("Loading training data...")
    train_examples = load_examples_with_patch_embeddings(TRAIN_DATA_DIR)
    logger.info("Loading validation data...")
    val_examples = load_examples_with_patch_embeddings(VAL_DATA_DIR)

    if not train_examples:
        raise ValueError(f"No valid training examples found in {TRAIN_DATA_DIR}. Check JSON files and patch embedding generation.")
    if not val_examples:
        logger.warning(f"No valid validation examples found in {VAL_DATA_DIR}. Proceeding without validation.")

    # Initialize patch-based model
    logger.info("Initializing patch-based model...")
    model = PatchEmbeddingAugmentedLLM(
        llm_model_name=LLM_MODEL_NAME,
        clip_patch_dim=768,  # ViT-B/32 patch dimension
        num_patches=49       # ViT-B/32 number of patches (7x7)
    )

    # Ensure model's tokenizer is accessible for dataset creation
    tokenizer = model.tokenizer

    # Create datasets with patch embeddings
    logger.info("Creating patch embedding datasets...")
    train_dataset = ShaderPatchEmbeddingDataset(
        train_examples,
        tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = None
    if val_examples:
         val_dataset = ShaderPatchEmbeddingDataset(
             val_examples,
             tokenizer,
             max_length=MAX_LENGTH
         )
    else:
         logger.warning("Validation dataset is empty, evaluation will be skipped.")


    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,  # Increased for better training
        per_device_train_batch_size=2,  # Keep small for memory efficiency on macOS
        gradient_accumulation_steps=8,  # Increased effective batch size = 2 * 8 = 16
        per_device_eval_batch_size=2,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else 0,  # More frequent evaluation
        save_steps=100,
        warmup_ratio=0.1,  # Use ratio instead of fixed steps
        logging_steps=10,
        learning_rate=1e-5,  # Lower learning rate for stability with patch features
        # --- Keep disabled for macOS compatibility ---
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        save_total_limit=3,  # Keep more checkpoints
        load_best_model_at_end=True if val_dataset else False,
        # --- Use compilation rate as primary metric ---
        metric_for_best_model="compilation_rate" if val_dataset else None,
        greater_is_better=True if val_dataset else None,
        # ---        
        report_to="wandb",
        save_safetensors=False,
        # --- Enhanced logging ---
        logging_dir=f"{OUTPUT_DIR}/logs",
        run_name="patch_vision_shader_model",
        # --- Data efficiency ---
        dataloader_num_workers=0,  # Disable multiprocessing for macOS stability
        remove_unused_columns=False,  # Keep patch embeddings
    )

    # Ensure gradient_checkpointing_kwargs is set correctly if needed
    if training_args.gradient_checkpointing:
         training_args.gradient_checkpointing_kwargs={'use_reentrant': False}


    # Initialize trainer - Use the patch embedding trainer class
    logger.info("Initializing patch embedding trainer...")
    trainer = PatchEmbeddingLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Pass None if not available
        data_collator=patch_collate_fn,  # Use patch collator
        tokenizer=tokenizer, # Pass tokenizer for saving purposes
        compute_metrics=compute_metrics if val_dataset else None
    )

    # Train model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished.")
        logger.info(f"Saving model to {OUTPUT_DIR}")
        trainer.save_model(OUTPUT_DIR)
        # Save tokenizer explicitly alongside the model
        tokenizer.save_pretrained(OUTPUT_DIR)
        logger.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}") # Log the full traceback


    logger.info("Script finished.")