# KGK AI — Fine-Tuning Pipeline

> **Important:** Fine-tuning is optional. Do not fine-tune until the baseline system works.
> Fine-tuning does NOT automatically give the model new factual knowledge. Use RAG for frequently changing knowledge.

## Overview

The fine-tuning pipeline allows you to customize the KGK AI foundation model using LoRA/QLoRA techniques. This is for adjusting the model's behavior, tone, or domain-specific patterns — not for injecting new facts.

## Prerequisites

- A working KGK AI baseline (all v1 phases complete)
- GPU access (local or cloud) — minimum 16 GB VRAM for 4B model with QLoRA
- Training dataset in the correct format

## Dataset Preparation

### Format

Training datasets use the Hugging Face chat format:

```json
{
  "messages": [
    {"role": "system", "content": "You are KGK AI..."},
    {"role": "user", "content": "Explain machine learning simply."},
    {"role": "assistant", "content": "Machine learning is..."}
  ]
}
```

### Creating a Dataset

1. Collect conversation pairs (user → assistant)
2. Ensure responses reflect KGK AI personality
3. Include diverse topics and difficulty levels
4. Minimum recommended: 500-1000 examples
5. Split: 90% train, 10% validation

Place datasets in `training/datasets/`.

## Training

### LoRA Fine-Tuning

```bash
python training/scripts/train.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --dataset training/datasets/kgk_dataset.json \
    --output_dir training/outputs \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

### QLoRA Fine-Tuning (4-bit quantization)

```bash
python training/scripts/train.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --dataset training/datasets/kgk_dataset.json \
    --output_dir training/outputs \
    --quantization 4bit \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-4
```

## Evaluation

After training, evaluate the fine-tuned model:

```bash
python scripts/evaluate.py --model training/outputs/final
```

Compare results against the baseline model to ensure improvements.

## Model Export

Export the fine-tuned model (merge LoRA weights with base model):

```bash
python training/scripts/export.py \
    --base_model Qwen/Qwen3-4B-Instruct-2507 \
    --lora_path training/outputs/final \
    --output_dir training/outputs/merged
```

## Hugging Face Upload

Upload the fine-tuned model to Hugging Face Hub:

```bash
python training/scripts/upload.py \
    --model_dir training/outputs/merged \
    --repo_id KGK/kgk-ai-v1 \
    --private
```

> **Security:** Never upload private user conversations, secrets, or private documents to Hugging Face.

## Configuration

Training configurations are in `training/configs/`. See `default_config.yaml` for all options.

## Notes

- LoRA/QLoRA only trains a small subset of parameters — efficient and cost-effective
- Training does NOT give the model new factual knowledge
- Use RAG for knowledge that changes frequently
- Always evaluate before deploying a fine-tuned model
- Keep the foundation model configurable so fine-tuned models can be swapped in
