# A LLM that can write Shader code

* glslViewer
* ollama run qwen2.5-coder:7b-instruct
* Training: Hugging Face Transformers + LoRA (efficient fine-tuning)
* Evaluation: SSIM (structural similarity) + syntax checker

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```
## Pipeline steps

1. Supervised Fine-Tuning (SFT)
2. RL Setup (Simplified)
3. Training Loop (SFT and RL)
4. Evaluation Metrics


### RL Algorithm

* REINFORCE

### Evalaution Metrics

* SSIM Score: Compare generated vs. target images.
* Compilation Rate: % of generated shaders that compile.
* Code Similarity: BLEU score between generated and ground-truth HLSL.


## Input questions

Should we input a target image or a textual description?

Can we train with both an image and a textual description?

Maybe text can be added later by going text -> prompt (2D, flat, graphical) -> image -> our input

## LLM

7B or 32B?

Which one? https://huggingface.co/spaces/mike-ravkine/can-ai-code-results


## Python Dependencies

    pip install requests Pillow scikit-image numpy

(We'll use requests to talk to the ollama API, Pillow and numpy for image loading, scikit-image for comparison).

## Data Preperation

We will work with 224x224 images (openai/clip-vit-base-patch32 preferred)

* `glslViewer data/train/003.frag  -e "screenshot,data/train/003.png" -e "quit" --headless`
* `sips -z 224 224 data/train/003.png --out data/train/003.png`

    export NUM=010
    glslViewer data/train/${NUM}.frag  -e "screenshot,data/train/${NUM}.png" -e "quit" --headless && \
    sips -z 224 224 data/train/${NUM}.png --out data/train/${NUM}.png


## References
### High level research

* https://aistudio.google.com/prompts/1-tPn7SCiajf52keGaacR-166H3oipD4W
* https://chat.deepseek.com/a/chat/s/5e91697c-31d0-4040-912e-530ebfb95886
* https://claude.ai/chat/bd8b198a-6065-46a1-9599-acdb1f868ee8
* https://chatgpt.com/c/67f6be98-434c-8010-8d9d-073c64332aec