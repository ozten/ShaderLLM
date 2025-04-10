# train_with_vision.py
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
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from torch import nn

# Configuration
#  or  "Qwen/Qwen2.5-1.5B-Instruct"  too big w/o some tweaks -> "Qwen/Qwen2.5-7B-Instruct"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Coder model
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP vision encoder
OUTPUT_DIR = "./shader_model"
TRAIN_DATA_DIR = "./data/train"
VAL_DATA_DIR = "./data/val"
MAX_LENGTH = 2048
IMAGE_SIZE = (224, 224)  # Standard size for vision models

# Custom model that combines vision encoder with LLM
# Note: This model doesn't use Vision, so we aren't using it...
class VisionLLMForShaderGeneration(nn.Module):
    def __init__(self, llm_model_name, vision_model_name):
        super().__init__()
        # Load vision encoder (e.g., CLIP)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        
        # Freeze vision encoder parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Get correct vision feature dimensions for CLIP model
        # CLIP uses vision_config for the vision encoder part
        vision_hidden_size = self.vision_model.config.vision_config.hidden_size
        llm_hidden_size = self.llm.config.hidden_size
        
        # Projection layer to match dimensions
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        
    def encode_images(self, images):
        # Process images through vision encoder
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model.get_image_features(**inputs)  # Get image features directly
        
        # Project to LLM hidden size
        projected_features = self.projection(outputs)
        
        return projected_features
        
    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None):
        # For MVP, we'll use a simple approach:
        # 1. Encode the image description in the prompt
        # 2. Use the LLM to generate shader code
        
        # Note: This implementation doesn't directly use image_embeddings in the LLM
        # A production version would modify the LLM to incorporate visual features
        
        # Just pass through to the LLM
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# Custom model that combines vision encoder with LLM
class VisionAugmentedLLM(nn.Module):
    def __init__(self, llm_model_name, vision_model_name, num_visual_tokens=16):
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
             # Important: Resize token embeddings if adding pad token like this
             # self.llm.resize_token_embeddings(len(self.tokenizer))
             # Note: Qwen2 usually has a pad token defined, check tokenizer_config.json
             # If it has pad_token_id set but tokenizer.pad_token is None, just setting it might be okay.
             # If resizing, ensure optimizer handles new parameters if training just started.

        # Vision features projection
        vision_dim = self.vision_model.config.projection_dim
        llm_dim = self.llm.config.hidden_size

        # Create learnable visual tokens (Initialization could be improved)
        self.num_visual_tokens = num_visual_tokens
        # self.visual_token_embeddings = nn.Parameter(
        #     torch.zeros(num_visual_tokens, llm_dim)
        # )

        # Project vision features to token space
        self.vision_projection = nn.Linear(vision_dim, llm_dim) # Project CLIP output (512) to LLM dim

        # --- ADDED: Delegate config ---
        # Make the wrapper look more like a PreTrainedModel for the Trainer
        self.config = self.llm.config

    # --- ADDED: Delegate gradient checkpointing methods ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enables gradient checkpointing for the underlying LLM."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            print("Enabling gradient checkpointing for LLM...")
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            print("WARNING: Underlying LLM does not support gradient_checkpointing_enable.")

    def gradient_checkpointing_disable(self):
        """Disables gradient checkpointing for the underlying LLM."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            print("Disabling gradient checkpointing for LLM...")
            self.llm.gradient_checkpointing_disable()
        else:
            print("WARNING: Underlying LLM does not support gradient_checkpointing_disable.")


    def forward(self, input_ids, attention_mask, images=None, labels=None):
        batch_size = input_ids.shape[0]
        embed_layer = self.llm.get_input_embeddings()
        llm_device = embed_layer.weight.device # Get device from LLM embedding layer

        if images is not None:
            # Ensure images are on the correct device
            pixel_values = images.to(self.vision_model.device)

            # Get image features from CLIP
            with torch.no_grad(): # Freeze vision model during training
                 vision_outputs = self.vision_model.get_image_features(
                     pixel_values=pixel_values
                 ) # Shape: [batch_size, vision_dim] (e.g., [B, 512])

            # Project vision features to LLM embedding dimension
            # Shape: [batch_size, llm_dim]
            projected_features = self.vision_projection(vision_outputs)

            # --- Simpler Approach: Prepend ONE projected visual token ---
            # Unsqueeze to add sequence dimension: [batch_size, 1, llm_dim]
            projected_features = projected_features.unsqueeze(1).to(llm_device)

            # Get text embeddings
            input_ids = input_ids.to(llm_device)
            inputs_embeds = embed_layer(input_ids) # Shape: [batch_size, seq_len, llm_dim]

            # Combine visual feature and text embeddings
            # Shape: [batch_size, 1 + seq_len, llm_dim]
            combined_embeds = torch.cat([projected_features, inputs_embeds], dim=1)

            # Extend attention mask
            attention_mask = attention_mask.to(llm_device)
            visual_attention = torch.ones(batch_size, 1, device=llm_device) # Attention for the single visual token
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

            # --- Modification: Ensure labels match the extended sequence length ---
            # The original code might have sliced labels assuming fixed length.
            # We need labels for the combined sequence length.
            # Check dimensions before passing to LLM
            expected_seq_len = combined_embeds.shape[1]
            if new_attention_mask.shape[1] != expected_seq_len:
                 raise ValueError(f"Attention mask length mismatch: Expected {expected_seq_len}, got {new_attention_mask.shape[1]}")
            if new_labels is not None and new_labels.shape[1] != expected_seq_len:
                 # This can happen if original labels were padded/truncated differently than input_ids
                 # Pad/truncate new_labels to match combined_embeds length
                 pad_len = expected_seq_len - new_labels.shape[1]
                 if pad_len > 0:
                     padding = torch.full((batch_size, pad_len), -100, dtype=torch.long, device=llm_device)
                     new_labels = torch.cat([new_labels, padding], dim=1)
                 elif pad_len < 0:
                     new_labels = new_labels[:, :expected_seq_len]

                 if new_labels.shape[1] != expected_seq_len: # Final check after adjustment
                     raise ValueError(f"Label length mismatch after adjustment: Expected {expected_seq_len}, got {new_labels.shape[1]}")


            # Forward pass using inputs_embeds
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=new_attention_mask,
                labels=new_labels,
                # `use_cache=False` is often implied when gradient_checkpointing=True, but can be explicit
                use_cache=not self.training or not self.llm.is_gradient_checkpointing
            )
            return outputs
        else:
            # Text-only forward pass (ensure gradient checkpointing status is respected)
             input_ids = input_ids.to(llm_device)
             attention_mask = attention_mask.to(llm_device)
             labels = labels.to(llm_device) if labels is not None else None
             return self.llm(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 labels=labels,
                 use_cache=not self.training or not self.llm.is_gradient_checkpointing
             )

# Data preparation functions
def load_examples_with_images(data_dir):
    examples = []
    for json_path in glob(f"{data_dir}/example_*.json"):
        try:
            with open(json_path, 'r') as f:
                example = json.load(f)
            
            # Extract image path from input field - using safer method
            # Assuming the input field contains a reference to the image filename
            input_text = example.get('input', '')
            image_filename = None
            
            # Try to extract image filename from input text
            if ':' in input_text:
                # Split by colon and get the last part (assumed to be the filename)
                image_filename = input_text.split(':')[-1].strip()
            
            if not image_filename:
                # If we couldn't extract a filename, skip this example
                continue
                
            # Construct full image path
            image_path = os.path.join(data_dir, image_filename)
            
            if os.path.exists(image_path):
                examples.append({
                    **example,
                    "image_path": image_path
                })
        except Exception as e:
            print(f"Error loading example from {json_path}: {e}")
    
    print(f"Loaded {len(examples)} valid examples with images from {data_dir}")
    return examples

# Custom dataset that includes images
class ShaderImageDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, image_processor, max_length=2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format prompt
        prompt = f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
"""
        response = example['output']
        
        # Tokenize text
        tokenized_input = self.tokenizer(
            prompt + response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels: -100 for prompt tokens to ignore them in loss
        labels = tokenized_input["input_ids"].clone()[0]
        prompt_tokens = self.tokenizer(
            prompt, 
            return_tensors="pt"
        )["input_ids"].shape[1]
        
        labels[:prompt_tokens] = -100
        
        # Load and process image
        try:
            image = Image.open(example["image_path"]).convert("RGB")
            image = image.resize(IMAGE_SIZE)
            
            # Process image
            image_inputs = self.image_processor(
                images=image, 
                return_tensors="pt"
            )
            image_tensor = image_inputs["pixel_values"][0]
        except Exception as e:
            print(f"Error processing image {example['image_path']}: {e}")
            # Create a blank tensor if image loading fails
            image_tensor = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        
        return {
    "input_ids": tokenized_input["input_ids"][0],
    "attention_mask": tokenized_input["attention_mask"][0],
    "labels": labels,
    "images": image_tensor 
}

# Data collator function
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    images = torch.stack([item["images"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images
    }

# Custom trainer
# Modify VisionLLMTrainer to work with VisionAugmentedLLM
class VisionLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract images
        images = inputs.pop("images", None)
        
        if images is not None:            
            # Ensure images are on the correct device (matching the model)
            # Infer device from a model parameter
            model_device = next(model.parameters()).device
            images = images.to(model_device)
            
            # Forward pass with images for VisionAugmentedLLM
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                images=images
            )
        else:
            # Text-only forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# Load data
train_examples = load_examples_with_images(TRAIN_DATA_DIR)
val_examples = load_examples_with_images(VAL_DATA_DIR)

if not train_examples:
    raise ValueError(f"No valid training examples found in {TRAIN_DATA_DIR}")

if not val_examples:
    raise ValueError(f"No valid validation examples found in {VAL_DATA_DIR}")

# Initialize model
print("Initializing model...")
# model = VisionLLMForShaderGeneration(LLM_MODEL_NAME, VISION_MODEL_NAME)
model = VisionAugmentedLLM(LLM_MODEL_NAME, VISION_MODEL_NAME)

# Create datasets
print("Creating datasets...")
train_dataset = ShaderImageDataset(
    train_examples,
    model.tokenizer,
    model.image_processor
)

val_dataset = ShaderImageDataset(
    val_examples,
    model.tokenizer,
    model.image_processor
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1, # Keep batch size small if using smaller model
    gradient_accumulation_steps=8, # Adjust as needed
    per_device_eval_batch_size=1,  # Keep eval batch size small
    eval_steps=100,
    save_steps=100,
    warmup_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    # Use bf16 or fp16 if possible, gradient checkpointing recommended
    bf16=True, # Or fp16=True
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    eval_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    save_safetensors=False,  # <--- ADD THIS LINE
)

# Initialize trainer
print("Initializing trainer...")
trainer = VisionLLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# Train model
print("Starting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)



# Example usage
if __name__ == "__main__":
    # After training, you can use the model for inference
    # model = VisionLLMForShaderGeneration.from_pretrained(OUTPUT_DIR)
    # model = VisionAugmentedLLM.from_pretrained(OUTPUT_DIR)
    # shader_code = generate_shader_from_image(model, "path/to/test/image.jpg")
    # print(shader_code)
    pass