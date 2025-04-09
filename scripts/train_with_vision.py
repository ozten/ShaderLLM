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
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Coder model
VISION_MODEL_NAME = "openai/clip-vit-base-patch32"  # CLIP vision encoder
OUTPUT_DIR = "./shader_model"
TRAIN_DATA_DIR = "./data/train"
VAL_DATA_DIR = "./data/val"
MAX_LENGTH = 2048
IMAGE_SIZE = (224, 224)  # Standard size for vision models

# Custom model that combines vision encoder with LLM
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
        
        # Vision features to text projection
        vision_hidden_size = self.vision_model.config.hidden_size
        llm_hidden_size = self.llm.config.hidden_size
        
        # Projection layer to match dimensions
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        
    def encode_images(self, images):
        # Process images through vision encoder
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.vision_model.device)
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        
        # Extract image features (using pooled output)
        image_features = outputs.pooler_output
        
        # Project to LLM hidden size
        projected_features = self.projection(image_features)
        
        return projected_features
        
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # Standard LLM inputs
        inputs = self.llm.prepare_inputs_for_generation(
            input_ids, 
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Add image embeddings if available
        if 'image_embeddings' in kwargs:
            # This would require modifying the LLM's forward pass
            # to incorporate the image embeddings
            inputs['image_embeddings'] = kwargs['image_embeddings']
            
        return inputs
    
    def forward(self, input_ids, attention_mask=None, labels=None, image_embeddings=None):
        # For MVP, we'll use a simple approach:
        # 1. Encode the image description in the prompt
        # 2. Use the LLM to generate shader code
        
        # This is a simplified approach - a production version would need
        # to modify the LLM architecture to properly incorporate visual features
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

# Data preparation functions
def load_examples_with_images(data_dir):
    examples = []
    for json_path in glob(f"{data_dir}/example_*.json"):
        with open(json_path, 'r') as f:
            example = json.load(f)
        
        # Extract image path from input field
        image_path = os.path.join(
            data_dir, 
            example['input'].split(": ")[1]
        )
        
        if os.path.exists(image_path):
            examples.append({
                **example,
                "image_path": image_path
            })
    
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
        inputs = self.tokenizer(
            prompt + response,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels: -100 for prompt tokens to ignore them in loss
        labels = inputs["input_ids"].clone()[0]
        prompt_tokens = self.tokenizer(
            prompt, 
            return_tensors="pt"
        )["input_ids"].shape[1]
        
        labels[:prompt_tokens] = -100
        
        # Load and process image
        image = Image.open(example["image_path"]).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        
        # Convert image to tensor
        image_tensor = self.image_processor(
            images=image, 
            return_tensors="pt"
        )["pixel_values"][0]
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "image": image_tensor
        }

# Data collator function
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    images = torch.stack([item["image"] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images
    }

# Load data
train_examples = load_examples_with_images(TRAIN_DATA_DIR)
val_examples = load_examples_with_images(VAL_DATA_DIR)

# Initialize model
model = VisionLLMForShaderGeneration(LLM_MODEL_NAME, VISION_MODEL_NAME)

# Create datasets
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

# Custom trainer
class VisionLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Process images and get embeddings
        images = inputs.pop("images").to(model.vision_model.device)
        image_embeddings = model.encode_images(images)
        
        # Regular LLM forward pass with labels
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            image_embeddings=image_embeddings
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduce batch size due to images
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=2,
    eval_steps=100,
    save_steps=100,
    warmup_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    fp16=True,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
)

# Initialize trainer
trainer = VisionLLMTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

# Train model
trainer.train()
trainer.save_model(OUTPUT_DIR)

# Generation function for evaluation/inference
def generate_shader_from_image(model, image_path, instruction="Generate an HLSL shader that reproduces this image"):
    # Load and process image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    
    # Get image embeddings
    inputs = model.image_processor(images=image, return_tensors="pt").to(model.vision_model.device)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        image_features = vision_outputs.pooler_output
        image_embeddings = model.projection(image_features)
    
    # Generate prompt
    prompt = f"""### Instruction:
{instruction}

### Input:
Reference image provided

### Response:
"""
    
    # Tokenize prompt
    inputs = model.tokenizer(prompt, return_tensors="pt").to(model.llm.device)
    
    # Generate shader code
    with torch.no_grad():
        output_ids = model.llm.generate(
            inputs["input_ids"],
            image_embeddings=image_embeddings,  # Custom arg that would need handling
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    # Decode output
    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    shader_code = output_text[len(prompt):]
    
    return shader_code