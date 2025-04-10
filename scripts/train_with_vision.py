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

# --- Model Definition ---
# Modified VisionAugmentedLLM to accept pre-computed embeddings
class EmbeddingAugmentedLLM(nn.Module):
    # Removed vision_model_name argument
    def __init__(self, llm_model_name, clip_embedding_dim, num_visual_tokens=1): # num_visual_tokens often 1 for this approach
        super().__init__()
        logger.info(f"Initializing EmbeddingAugmentedLLM with LLM: {llm_model_name}")
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
             logger.warning("Tokenizer missing pad token, setting to eos_token.")
             self.tokenizer.pad_token = self.tokenizer.eos_token
             # Optional: Resize embeddings if you are *sure* the token didn't exist before
             # self.llm.resize_token_embeddings(len(self.tokenizer))
             # logger.info(f"Resized LLM token embeddings to: {len(self.tokenizer)}")

        # Vision features projection (using the known CLIP dimension)
        llm_dim = self.llm.config.hidden_size
        self.clip_embedding_dim = clip_embedding_dim # Store dimension

        # Project CLIP embedding features (e.g., 512) to LLM dim
        # This layer *will* be trained
        self.vision_projection = nn.Linear(self.clip_embedding_dim, llm_dim)
        logger.info(f"Created projection layer: {self.clip_embedding_dim} -> {llm_dim}")

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

    # Modified forward to accept 'clip_embeddings' instead of 'images'
    def forward(self, input_ids, attention_mask, clip_embeddings=None, labels=None):
        batch_size = input_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()
        llm_device = embed_layer.weight.device # Get device from LLM embedding layer

        if clip_embeddings is not None:
            # Ensure embeddings are float and on the correct device
            # Expected shape: [batch_size, clip_embedding_dim]
            if clip_embeddings.shape[-1] != self.clip_embedding_dim:
                 raise ValueError(f"Input clip_embeddings dim ({clip_embeddings.shape[-1]}) does not match expected dim ({self.clip_embedding_dim})")

            clip_embeddings = clip_embeddings.to(dtype=self.vision_projection.weight.dtype, device=llm_device) # Match projection layer type

            # Project vision features to LLM embedding dimension
            # Shape: [batch_size, llm_dim]
            projected_features = self.vision_projection(clip_embeddings)

            # Unsqueeze to add sequence dimension: [batch_size, 1, llm_dim] (Prepending ONE token)
            projected_features = projected_features.unsqueeze(1)

            # Get text embeddings
            input_ids = input_ids.to(llm_device)
            inputs_embeds = embed_layer(input_ids) # Shape: [batch_size, seq_len, llm_dim]

            # Combine visual feature and text embeddings
            # Shape: [batch_size, 1 + seq_len, llm_dim]
            combined_embeds = torch.cat([projected_features, inputs_embeds], dim=1)

            # Extend attention mask
            attention_mask = attention_mask.to(llm_device)
            visual_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=llm_device) # Match dtype
            # Shape: [batch_size, 1 + seq_len]
            new_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

            # Adjust labels if provided (ignore visual token in loss)
            if labels is not None:
                labels = labels.to(llm_device)
                # Shape: [batch_size, 1] filled with -100
                visual_labels = torch.full((batch_size, 1), -100, dtype=torch.long, device=llm_device)
                # Shape: [batch_size, 1 + seq_len]
                new_labels = torch.cat([visual_labels, labels], dim=1)
            else:
                new_labels = None

            # --- Label Padding/Truncation (Important!) ---
            # Ensure labels match the final sequence length after prepending visual token
            expected_seq_len = combined_embeds.shape[1]
            if new_labels is not None and new_labels.shape[1] != expected_seq_len:
                 current_len = new_labels.shape[1]
                 # Pad with -100 if new_labels is shorter
                 if current_len < expected_seq_len:
                     pad_len = expected_seq_len - current_len
                     padding = torch.full((batch_size, pad_len), -100, dtype=torch.long, device=llm_device)
                     new_labels = torch.cat([new_labels, padding], dim=1)
                 # Truncate if new_labels is longer (shouldn't typically happen if input_ids were padded correctly)
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
             logger.warning("Performing text-only forward pass (no clip_embeddings provided).")
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
# Modified data loading function
def load_examples_with_embeddings(data_dir):
    examples = []
    required_keys = ["instruction", "input", "output", "clip_embeddings"]
    json_files = glob(f"{data_dir}/example_*.json")
    logger.info(f"Found {len(json_files)} JSON files in {data_dir}")

    skipped_count = 0
    loaded_count = 0
    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                example = json.load(f)

            # Validate required keys and embedding format
            if not all(key in example for key in required_keys):
                logger.warning(f"Skipping {json_path}: Missing one or more required keys ({required_keys}).")
                skipped_count += 1
                continue

            if not isinstance(example["clip_embeddings"], list) or not example["clip_embeddings"]:
                 logger.warning(f"Skipping {json_path}: 'clip_embeddings' is not a non-empty list.")
                 skipped_count += 1
                 continue

            # Optionally check embedding dimension here if known
            if len(example["clip_embeddings"]) != CLIP_EMBEDDING_DIM:
                 logger.warning(f"Skipping {json_path}: Embedding dimension is {len(example['clip_embeddings'])}, expected {CLIP_EMBEDDING_DIM}.")
                 skipped_count += 1
                 continue

            # We don't need the image path anymore for training
            # but keep the original 'input' field as it might be used in the prompt
            examples.append(example)
            loaded_count += 1

        except json.JSONDecodeError:
            logger.error(f"Skipping {json_path}: Invalid JSON.")
            skipped_count += 1
        except Exception as e:
            logger.error(f"Error loading example from {json_path}: {e}")
            skipped_count += 1

    logger.info(f"Loaded {loaded_count} valid examples with embeddings from {data_dir}. Skipped {skipped_count}.")
    return examples

# Modified Dataset to use embeddings
class ShaderEmbeddingDataset(torch.utils.data.Dataset):
    # Removed image_processor
    def __init__(self, examples, tokenizer, max_length=2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Format prompt (using the 'input' field which contains the image filename)
        # Ensure the 'input' field is just the filename as intended by generate_example_json.py
        image_ref = example['input'] # Should be like "002.png"
        prompt = f"""### Instruction:
{example['instruction']}

### Input Reference:
Image corresponding to embedding for: {image_ref}

### Response:
"""
        response = example['output']

        # Tokenize text (prompt + response)
        # Important: Account for the visual token potentially pushing text out
        # Leave some buffer, maybe max_length - num_visual_tokens (here, max_length - 1)
        # However, HF Trainer usually handles padding/truncation well based on `max_length`
        tokenized_input = self.tokenizer(
            prompt + response + self.tokenizer.eos_token, # Add EOS token explicitly
            truncation=True,
            max_length=self.max_length, # Max length for text part
            padding="max_length", # Pad text part to max_length
            return_tensors="pt"
        )

        # Create labels: -100 for prompt tokens to ignore them in loss
        labels = tokenized_input["input_ids"].clone().squeeze() # Squeeze to 1D

        # Tokenize prompt *separately* to find length accurately
        prompt_tokens = self.tokenizer(
            prompt,
            add_special_tokens=False # Don't add BOS/EOS here if they are part of the main tokenization
        )["input_ids"]
        prompt_len = len(prompt_tokens)

        # Use prompt_len, but be careful with special tokens (like BOS) added by default
        # A safer way might be tokenizing prompt+response and prompt, then finding the difference
        # Or, simpler: find the start of the response after tokenization
        full_text = prompt + response + self.tokenizer.eos_token
        tokenized_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length)["input_ids"]
        tokenized_prompt_only = self.tokenizer(prompt, truncation=True, max_length=self.max_length)["input_ids"]

        # Assuming prompt doesn't contain EOS/PAD tokens internally that match the actual ones
        # This calculation needs care depending on tokenizer specifics (BOS tokens etc.)
        # Let's stick to the simpler method for now, assuming it's roughly correct
        # Adjust index carefully if BOS token is added automatically
        prompt_token_length = len(tokenized_prompt_only)
        if self.tokenizer.bos_token_id is not None and tokenized_full[0] == self.tokenizer.bos_token_id:
            # If tokenizer adds BOS, the prompt length might seem off by 1 relative to combined sequence
             pass # Adjust logic if needed based on specific tokenizer behavior

        # Mask prompt tokens
        labels[:prompt_token_length] = -100

        # Handle potential truncation where response gets cut off
        # Ensure label isn't -100 for the very last *actual* token if it wasn't padding
        input_ids_squeezed = tokenized_input["input_ids"].squeeze()
        # Find the index of the first padding token (if any)
        pad_token_indices = (input_ids_squeezed == self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        first_pad_index = pad_token_indices[0].item() if len(pad_token_indices) > 0 else len(labels)
        # Make sure labels beyond the first pad token are also -100
        if first_pad_index < len(labels):
             labels[first_pad_index:] = -100

        # Ensure EOS token has a valid label if it wasn't masked or padded out
        eos_token_indices = (input_ids_squeezed == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_token_indices) > 0:
            last_eos_index = eos_token_indices[-1].item()
            if last_eos_index < first_pad_index and last_eos_index >= prompt_token_length:
                 # If the last EOS is part of the response and not masked/padded, keep its label
                 pass # Label should already be the EOS token ID
            # If EOS was part of prompt or padding, it should remain -100


        # Load pre-computed embedding and convert to tensor
        clip_embedding_list = example["clip_embeddings"]
        clip_embedding_tensor = torch.tensor(clip_embedding_list, dtype=torch.float)

        return {
            "input_ids": input_ids_squeezed,
            "attention_mask": tokenized_input["attention_mask"].squeeze(), # Squeeze to 1D
            "labels": labels,
            "clip_embeddings": clip_embedding_tensor # Return the embedding tensor
        }

# Modified data collator
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    # Stack clip embeddings instead of images
    clip_embeddings = torch.stack([item["clip_embeddings"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "clip_embeddings": clip_embeddings # Pass embeddings in the batch
    }

# Modified Trainer
class EmbeddingLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract clip_embeddings instead of images
        clip_embeddings = inputs.pop("clip_embeddings", None)
        labels = inputs.get("labels") # Keep labels in inputs for the model

        if clip_embeddings is not None:
            # Ensure embeddings are on the correct device (matching the model)
            # Check if model has parameters before accessing device
            if list(model.parameters()):
                model_device = next(model.parameters()).device
                clip_embeddings = clip_embeddings.to(model_device)
            else:
                # Handle case where model might have no parameters (unlikely but safe)
                logger.warning("Model has no parameters, cannot determine device for embeddings.")
                # You might default to CPU or raise an error depending on your setup
                clip_embeddings = clip_embeddings.to('cpu')

            # Forward pass with embeddings
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                clip_embeddings=clip_embeddings # Pass embeddings to the model
            )
        else:
            # Text-only forward pass (should ideally not happen if data is prepared correctly)
            logger.warning("Trainer compute_loss: clip_embeddings not found in inputs. Performing text-only forward pass.")
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )

        loss = outputs.loss if hasattr(outputs, "loss") else None
        if loss is None:
             logger.error("Model output did not contain 'loss'.")
             # Handle this case, maybe return a dummy loss or raise error
             # For now, returning None which might break training loop.
             # Check model's return structure if this happens.
             # return (torch.tensor(0.0, device=model_device, requires_grad=True), outputs) if return_outputs else torch.tensor(0.0, device=model_device, requires_grad=True)
             raise ValueError("Model output missing 'loss' attribute.")


        return (loss, outputs) if return_outputs else loss

# --- Main Training Script ---
if __name__ == "__main__":
    # Load data
    logger.info("Loading training data...")
    train_examples = load_examples_with_embeddings(TRAIN_DATA_DIR)
    logger.info("Loading validation data...")
    val_examples = load_examples_with_embeddings(VAL_DATA_DIR)

    if not train_examples:
        raise ValueError(f"No valid training examples found in {TRAIN_DATA_DIR}. Check JSON files and embedding generation.")
    if not val_examples:
        # Optional: Allow training without validation data, but issue a warning
        logger.warning(f"No valid validation examples found in {VAL_DATA_DIR}. Proceeding without validation.")
        # raise ValueError(f"No valid validation examples found in {VAL_DATA_DIR}. Check JSON files.")

    # Initialize model - Pass embedding dim, remove vision model name
    logger.info("Initializing model...")
    # Note: No AutoModel or AutoImageProcessor needed here anymore for the *training* script's model definition
    model = EmbeddingAugmentedLLM(
        llm_model_name=LLM_MODEL_NAME,
        clip_embedding_dim=CLIP_EMBEDDING_DIM
        )

    # Ensure model's tokenizer is accessible for dataset creation
    tokenizer = model.tokenizer

    # Create datasets - Remove image_processor
    logger.info("Creating datasets...")
    train_dataset = ShaderEmbeddingDataset(
        train_examples,
        tokenizer,
        max_length=MAX_LENGTH
    )

    val_dataset = None
    if val_examples:
         val_dataset = ShaderEmbeddingDataset(
             val_examples,
             tokenizer,
             max_length=MAX_LENGTH
         )
    else:
         logger.warning("Validation dataset is empty, evaluation will be skipped.")


    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2, # Can potentially increase slightly now
        gradient_accumulation_steps=4, # Effective batch size = 2 * 4 = 8
        per_device_eval_batch_size=2,
        eval_strategy="steps" if val_dataset else "no", # Evaluate only if val_dataset exists
        eval_steps=100 if val_dataset else 0, # Evaluate only if val_dataset exists
        save_steps=100,
        warmup_steps=50, # Adjust warmup steps based on total steps
        logging_steps=10,
        learning_rate=2e-5,
        #bf16=torch.cuda.is_bf16_supported(), # Use BF16 if available
        #fp16=not torch.cuda.is_bf16_supported(), # Otherwise use FP16 if available
        # --- MODIFIED for MPS compatibility ---
    # Disable mixed precision as fp16/bf16 support on MPS via accelerate can be problematic
        bf16=False,
        fp16=False,
        gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={'use_reentrant': False}, # Often needed for newer models
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False, # Load best only if evaluating
        metric_for_best_model="loss" if val_dataset else None, # Metric only if evaluating
        greater_is_better=False if val_dataset else None,
        report_to="tensorboard",
        # Disable safetensors due to shared weight issue with Qwen model + safetensors library
        # Fall back to saving as pytorch_model.bin
        save_safetensors=False,
        logging_dir=f"{OUTPUT_DIR}/logs", # Separate logs dir
    )

    # Ensure gradient_checkpointing_kwargs is set correctly if needed
    if training_args.gradient_checkpointing:
         training_args.gradient_checkpointing_kwargs={'use_reentrant': False}


    # Initialize trainer - Use the modified trainer class
    logger.info("Initializing trainer...")
    trainer = EmbeddingLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Pass None if not available
        data_collator=collate_fn,
        tokenizer=tokenizer # Pass tokenizer for saving purposes
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