# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ShaderLLM is a machine learning project that trains language models to generate GLSL shader code from input images. The system uses a vision-augmented LLM architecture combining CLIP vision embeddings with a Qwen2.5 coding model.

## Key Architecture Components

### Model Architecture
- **Vision Model**: CLIP (openai/clip-vit-base-patch32) for image feature extraction
- **LLM**: Qwen2.5-0.5B-Instruct for shader code generation
- **Vision Projection**: Linear layer mapping CLIP embeddings (512D) to LLM hidden dimension
- **Training Strategy**: Pre-computed CLIP embeddings + supervised fine-tuning with ROUGE evaluation

### Data Pipeline
1. **Data Generation**: Template shaders in `data/shaders/` expanded using `comb-synth-gen`
2. **Image Rendering**: `glslViewer` renders 224x224 images from .frag files
3. **Embedding Generation**: CLIP processes images to create embeddings stored in JSON
4. **Training Data**: JSON files contain instruction/input/output/clip_embeddings

### Training Process
- **Supervised Fine-Tuning**: Vision projection layer + LoRA-style training
- **Evaluation**: ROUGE scores for text similarity, SSIM for image comparison
- **Hardware**: CPU inference (MPS compatibility issues noted)

## Development Commands

### Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Data Preparation
```bash
# Generate training examples with embeddings
python scripts/generate_example_json.py

# Process shader files and generate images
python scripts/prepare_data.py
```

### Training
```bash
# Train the vision-augmented model
python scripts/train_with_vision.py
```

### Inference
```bash
# Generate shader code from an image
python scripts/run_model.py
```

### Image Generation from Shaders
```bash
# Generate screenshot from shader (single file)
export NUM=010
glslViewer data/train/${NUM}.frag -e "screenshot,data/train/${NUM}.png" -e "quit" --headless
sips -z 224 224 data/train/${NUM}.png --out data/train/${NUM}.png
```

## Important Implementation Notes

### Model Classes
- `VisionAugmentedLLM` (run_model.py): Full model for inference including vision components
- `EmbeddingAugmentedLLM` (train_with_vision.py): Training model using pre-computed embeddings
- Custom trainer `EmbeddingLLMTrainer` handles vision embedding integration

### Known Issues
- MPS (Apple Silicon) compatibility problems require CPU inference
- safetensors saving disabled due to shared weight issues with Qwen model
- Vision model weights not saved (uses pre-trained CLIP)

### Data Format
Training examples use this JSON structure:
```json
{
  "instruction": "Generate an HLSL shader that reproduces this image",
  "input": "filename.png",
  "output": "shader code...",
  "clip_embeddings": [512-dimensional array]
}
```

### Future Architecture Improvements
- Patch-level features instead of global CLIP embeddings
- Cross-attention mechanisms between vision and text
- Reinforcement learning with shader compilation + SSIM rewards