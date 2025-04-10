# Inside forward method...
if images is not None:
    pixel_values = images.to(self.vision_model.device)

    # Get sequence of patch features + [CLS] token hidden states
    with torch.no_grad():
         # Request the output before pooling
         vision_outputs = self.vision_model(
             pixel_values=pixel_values,
             output_hidden_states=False # Or True if needed, but last_hidden_state is usually enough
         )
         # Shape typically [batch_size, num_patches + 1, vision_hidden_dim]
         # The +1 is often the [CLS] token. We usually want the patch tokens.
         image_features_sequence = vision_outputs.last_hidden_state[:, 1:, :] # Exclude [CLS] token at index 0

    # Project the sequence of features
    # vision_projection is Linear(vision_dim, llm_dim)
    # Shape: [batch_size, num_patches, llm_dim]
    projected_features = self.vision_projection(image_features_sequence)
    projected_features = projected_features.to(llm_device) # Move to LLM device AFTER projection

    # Get text embeddings
    input_ids = input_ids.to(llm_device)
    inputs_embeds = embed_layer(input_ids) # Shape: [batch_size, seq_len, llm_dim]

    # Combine: [Vis, Vis, ..., Vis, Text, Text, ...]
    # Shape: [batch_size, num_patches + seq_len, llm_dim]
    combined_embeds = torch.cat([projected_features, inputs_embeds], dim=1)

    # Extend attention mask
    attention_mask = attention_mask.to(llm_device)
    num_visual_tokens = projected_features.shape[1]
    # Shape: [batch_size, num_visual_tokens]
    visual_attention = torch.ones(batch_size, num_visual_tokens, device=llm_device)
    # Shape: [batch_size, num_visual_tokens + seq_len]
    new_attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

    # Adjust labels
    if labels is not None:
        labels = labels.to(llm_device)
        # Shape: [batch_size, num_visual_tokens] filled with -100
        visual_labels = torch.full((batch_size, num_visual_tokens), -100, dtype=torch.long, device=llm_device)
        # Shape: [batch_size, num_visual_tokens + seq_len]
        new_labels = torch.cat([visual_labels, labels], dim=1)
        # --- Add the label length adjustment logic from your original code here ---
        expected_seq_len = combined_embeds.shape[1]
        if new_labels.shape[1] != expected_seq_len:
             # Pad/truncate new_labels logic...
             pass # (Use your existing padding/truncating logic)

    else:
        new_labels = None

    # Forward pass using inputs_embeds
    outputs = self.llm(
        inputs_embeds=combined_embeds,
        attention_mask=new_attention_mask,
        labels=new_labels,
        use_cache=not self.training or not self.llm.is_gradient_checkpointing # Use appropriate cache setting
    )
    return outputs
else:
    # Text-only forward pass...
    # ... (rest of your code)