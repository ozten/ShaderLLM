# This could replace VisionAugmentedLLM in train_with_vision.py

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoImageProcessor
import copy

class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout=0.1):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query from text, key/value from vision
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, text_hidden_states, visual_hidden_states):
        # Project queries from text features
        mixed_query_layer = self.query(text_hidden_states)
        
        # Project keys and values from visual features
        mixed_key_layer = self.key(visual_hidden_states)
        mixed_value_layer = self.value(visual_hidden_states)
        
        # Reshape for attention computation
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        
        # Apply softmax and dropout
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Get context vectors
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Final projection
        output = self.out(context_layer)
        
        return output

class VisionAugmentedTransformerBlock(nn.Module):
    def __init__(self, original_block, hidden_size):
        super().__init__()
        # Copy the original transformer block
        self.original_block = original_block
        
        # Add cross-attention layer
        self.cross_attention = CrossAttention(hidden_size)
        self.cross_attention_layernorm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states, attention_mask=None, visual_features=None, **kwargs):
        # Original block forward pass
        outputs = self.original_block(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]
        
        # Apply cross-attention with visual features
        if visual_features is not None:
            # Cross attention
            cross_attention_output = self.cross_attention(hidden_states, visual_features)
            # Residual connection and layer norm
            hidden_states = self.cross_attention_layernorm(hidden_states + cross_attention_output)
        
        # Return the same structure as the original block, but with updated hidden states
        outputs = (hidden_states,) + outputs[1:]
        return outputs

class VisionAugmentedLLM(nn.Module):
    def __init__(self, llm_model_name, vision_model_name, 
                 augment_layers=None, freeze_llm=True):
        super().__init__()
        # Load vision encoder
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        
        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze parameters if requested
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Vision features projection
        vision_dim = self.vision_model.config.vision_config.hidden_size
        llm_dim = self.llm.config.hidden_size
        self.vision_projection = nn.Linear(vision_dim, llm_dim)
        
        # Modify the LLM architecture: add cross-attention layers
        # Determine which transformer layers to augment with cross-attention
        if augment_layers is None:
            # Default: augment every third layer
            num_layers = self.llm.config.num_hidden_layers
            augment_layers = [i for i in range(0, num_layers, 3)]
        
        # Replace selected transformer blocks with augmented versions
        for layer_idx in augment_layers:
            # This is a simplified approach - the actual layer access depends on the specific model architecture
            if hasattr(self.llm, "transformer"):
                # For models like GPT-2
                orig_block = self.llm.transformer.h[layer_idx]
                self.llm.transformer.h[layer_idx] = VisionAugmentedTransformerBlock(
                    orig_block, self.llm.config.hidden_size
                )
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                # For models like Llama
                orig_block = self.llm.model.layers[layer_idx]
                self.llm.model.layers[layer_idx] = VisionAugmentedTransformerBlock(
                    orig_block, self.llm.config.hidden_size
                )
        
        # Store the augmented layer indices
        self.augmented_layers = augment_layers
    
    def process_images(self, images):
        """Process images through vision encoder and project to LLM dimension"""
        with torch.no_grad():
            vision_outputs = self.vision_model.get_image_features(
                self.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.vision_model.device)
            )
        
        # Project to LLM hidden dimension
        projected_features = self.vision_projection(vision_outputs)
        
        # Expand dimensions to create a sequence of visual features
        # This creates a sequence of identical visual features (could be improved)
        batch_size = projected_features.shape[0]
        expanded_features = projected_features.unsqueeze(1).expand(batch_size, 16, -1)
        
        return expanded_features
    
    def forward(self, input_ids, attention_mask=None, images=None, labels=None):
        """Forward pass with optional visual features"""
        # Process images if provided
        visual_features = self.process_images(images) if images is not None else None
        
        # This would require modifying the forward methods of the transformer layers
        # to accept and use the visual_features
        
        # For now, we'll use a simple approach to inject visual features
        # through the custom VisionAugmentedTransformerBlock
        
        # Store visual features for access during layer forward passes
        self._visual_features = visual_features
        
        # Hook to inject visual features
        def inject_visual_features(module, input_args, output):
            # Only modify output for augmented blocks
            if isinstance(module, VisionAugmentedTransformerBlock) and self._visual_features is not None:
                # The original output is (hidden_states, attention_probs, ...)
                hidden_states = output[0]
                
                # Apply cross-attention with visual features
                cross_attn_output = module.cross_attention(hidden_states, self._visual_features)
                new_hidden_states = module.cross_attention_layernorm(hidden_states + cross_attn_output)
                
                # Return modified output
                return (new_hidden_states,) + output[1:]
            return output
        
        # Register forward hooks for augmented layers
        hooks = []
        for layer_idx in self.augmented_layers:
            if hasattr(self.llm, "transformer"):
                layer = self.llm.transformer.h[layer_idx]
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                layer = self.llm.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(inject_visual_features))
        
        # Forward pass through LLM
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Clean up
        self._visual_features = None
        
        return outputs
        
    def generate(self, input_ids, images=None, attention_mask=None, **generate_kwargs):
        """Generation with visual context"""
        # Process images if provided
        visual_features = self.process_images(images) if images is not None else None
        
        # Store visual features for layer access
        self._visual_features = visual_features
        
        # Similar hook approach as in forward
        def inject_visual_features(module, input_args, output):
            if isinstance(module, VisionAugmentedTransformerBlock) and self._visual_features is not None:
                hidden_states = output[0]
                cross_attn_output = module.cross_attention(hidden_states, self._visual_features)
                new_hidden_states = module.cross_attention_layernorm(hidden_states + cross_attn_output)
                return (new_hidden_states,) + output[1:]
            return output
        
        # Register hooks
        hooks = []
        for layer_idx in self.augmented_layers:
            if hasattr(self.llm, "transformer"):
                layer = self.llm.transformer.h[layer_idx]
            elif hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
                layer = self.llm.model.layers[layer_idx]
            hooks.append(layer.register_forward_hook(inject_visual_features))
        
        # Generate with the LLM
        generated_ids = self.llm.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Clean up
        self._visual_features = None
        
        return generated_ids

# Example training loop
def train_vision_augmented_llm(model, train_dataloader, optimizer, num_epochs=3):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(model.llm.device)
            attention_mask = batch["attention_mask"].to(model.llm.device)
            labels = batch["labels"].to(model.llm.device)
            images = batch["images"].to(model.vision_model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_dataloader)}")