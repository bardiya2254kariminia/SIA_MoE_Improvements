# Visual Analogy — Selective Image Analogy (SIA)

Controllable image editing via visual analogies using Selective Token LoRA (STLoRA) on FLUX.2-Klein.

Given images **A**, **A'**, and **B**, the model generates **B'** by transferring the transformation A→A' onto B — with per-edit controllability through selective token masking. Supports datasets with **2, 3, or 4 simultaneous edits** and difference-of-means (DoM) text-embedding steering vectors.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (≥24 GB VRAM recommended)
- A HuggingFace account with access to [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B)

### Environment variables

```bash
export HF_TOKEN="your_huggingface_token"
export WANDB_API_KEY="your_wandb_key"   # optional, for logging
```

### Installation

```bash
    conda env create -f environment.yml
conda activate visual_analogy
pip install -e .
```

Or download the dataset and install in one step:

```bash
bash setup.sh
```

---

## Data Structure

The dataset is organized by concept, with each sample containing multi-edit pairs:

```
data/
├── 135-close-eye/
│   └── 02/
│       └── 1/
│           ├── input.png           # Source image (A / B)
│           ├── total_changes.png   # All edits applied (A')
│           ├── pose_only.png       # Edit 1 only applied
│           ├── style_only.png      # Edit 2 only applied
│           └── prompt.json         # {"edit1": "...", "edit2": "..."}
├── 136-hat/
└── ...
```

Each concept folder contains view subfolders, each with numbered sample folders. Two samples from the same view form an analogy pair:

- **Sample 1** → A (input) / A' (total_changes)
- **Sample 2** → B (input) / B' (partial edit — pose_only or style_only)

The training script randomly selects which edit to suppress, enabling STLoRA to learn per-edit gating.

---

## Project Structure

```
visual_analogy/              # Core library (pip-installable)
├── models/
│   └── selective_lora.py       # BaseLoRALinear & SelectiveLoRALinear
└── utils/
    ├── hf_utils.py             # HF cache resolution
    └── selective_lora.py       # Injection, masking, save/load utilities

training/                    # Training & inference scripts
├── train_simple_lora_flux2_klein.py   # Simple LoRA (Stage 1)
├── train_stlora_flux2_klein.py        # STLoRA with PPS (Stage 2)
├── infer_single.py                    # Inference with all suppression modes
├── steering_dom.py                    # DoM steering vector generation
├── ablation.py                        # CSV × STLoRA 2×2 ablation runner
├── _check_token_alignment.py          # CPU regression test for token positions
└── configs/
    ├── train_simple_lora_flux2_klein.yaml
    └── train_stlora_flux2_klein.yaml

evaluation/                  # Evaluation metrics
├── evaluators.py            # CLIP, SigLIP, BLIP, LPIPS, DINOv2, FaceID
├── utils.py
└── example_evaluation.ipynb
```

---

## Training

All training is configured via YAML files in `training/configs/`.

### Stage 1 — Simple LoRA (Global Analogy)

Trains a standard LoRA on **FLUX.2-Klein** using concat-mode analogy (A, A', B as condition images → denoise B'). The loss is computed on the full B' target.

```bash
accelerate launch --mixed_precision bf16 \
    training/train_simple_lora_flux2_klein.py \
    --config training/configs/train_simple_lora_flux2_klein.yaml
```

### Stage 2 — STLoRA (Selective Token LoRA)

Builds on the Stage 1 checkpoint. Injects `SelectiveLoRALinear` modules into the text-stream (context) MM-DiT layers and trains with Partial Prompt Suppression (PPS) loss — enabling per-edit control at inference.

The training loop alternates between:

- **Base LoRA steps** (55%): maintain global analogy quality
- **STLoRA steps** (35%): learn selective suppression via token masks
- **All-suppressed steps** (10%): identity regularisation (output should match B)

```bash
accelerate launch --mixed_precision bf16 \
    training/train_stlora_flux2_klein.py \
    --config training/configs/train_stlora_flux2_klein.yaml
```

Set in the YAML config:

```yaml
base_lora_path: "path/to/stage1/checkpoint"
```

---

## Inference

```bash
python training/infer_single.py \
    --config training/configs/train_stlora_flux2_klein.yaml \
    --image_a /path/to/A.png \
    --image_a_prime /path/to/A_prime.png \
    --image_b /path/to/B.png \
    --prompt "Image 1 is the original ... Edit 1: X. Edit 2: Y. Apply ..." \
    --edits "X" "Y" \
    --output /path/to/result.png
```

This generates a comparison strip showing: A | A' | B | base_lora | stlora (no suppression) | stlora (suppress e1) | stlora (suppress e2) | stlora (suppress all).

---

## Steering Vectors (DoM)

`steering_dom.py` implements difference-of-means text-embedding steering for fine-grained control:

1. Uses the Qwen3 text encoder to generate N contrastive pos/neg sentence pairs per edit
2. Encodes each via the same pipeline path as diffusion inference
3. Computes `v = normalize(mean(pos) − mean(neg))` — a unit-norm steering direction
4. Caches both the pairs (JSONL) and the vector (`.pt`) for reuse

The steering vector can be added to prompt embeddings at edit-token positions during inference for additional controllability beyond STLoRA masking.

---

## CSV × STLoRA Ablation

`training/ablation.py` runs a 2×2 grid varying the **Concept Steering Vector (CSV)** and **STLoRA** independently — same noise seed in every cell, so any visual difference is attributable to the intervention.

```bash
python training/ablation.py \
    --config        training/configs/train_stlora_flux2_klein.yaml \
    --image_a       /path/to/A.png \
    --image_a_prime /path/to/A_prime.png \
    --image_b       /path/to/B.png \
    --edits "change pose to running" "change style to oil painting" \
    --suppress 1 \
    --csv_alpha 5.0 \
    --output_dir   /tmp/ablation_run
```

Outputs in `--output_dir`:

- `ablation_grid.png` — A | A' | B reference strip + 4-cell grid (rows: STLoRA off/on, cols: CSV off/on)
- `csv{0,1}_stlora{0,1}.png` — annotated per-cell PNGs
- `summary.txt` — prompt, suppressed indices, decoded edit-token positions, seed
- `metrics.csv` — CLIP / LPIPS / DINOv2 scores per cell (skipped if evaluators unavailable)

---

## Token-Position Alignment (important fix)

Both STLoRA's token mask and the CSV are applied to specific token positions of the prompt. Because `Flux2KleinPipeline._get_qwen3_prompt_embeds` wraps every prompt in Qwen's chat template (`<|im_start|>user\n…`) **before** tokenizing, those positions must be computed against the **chat-templated** sequence — not the raw prompt — or they will not line up with the rows of `prompt_embeds` the transformer actually sees.

This was previously mis-handled (callers passed a `use_chat_template` kwarg the underlying utility did not accept; the resulting `TypeError` was silently swallowed and STLoRA's mask defaulted to all-False, making STLoRA a no-op throughout training and inference). The unified helpers `klein_find_substring_token_indices` and `klein_templated_prompt_token_positions` in `visual_analogy/utils/selective_lora.py` are now the single source of truth, and `build_token_mask` raises loudly on alignment failures.

**Any STLoRA checkpoint trained before this fix is effectively a base-LoRA-only checkpoint and should be retrained from scratch.**

A CPU-only regression test is provided:

```bash
python training/_check_token_alignment.py
```

It validates that each edit's positions decode back to the edit text, that per-edit positions are a subset of the all-prompt positions, and that `build_token_mask` raises on missing edits instead of returning a zero mask.

---

## Config Options

| Key                             | Description                                                    |
| ------------------------------- | -------------------------------------------------------------- |
| `pretrained_model_name_or_path` | Base model (default: `black-forest-labs/FLUX.2-klein-base-9B`) |
| `data_root`                     | Path to the dataset                                            |
| `output_dir`                    | Where checkpoints and logs are saved                           |
| `base_lora_path`                | Stage 1 checkpoint path (STLoRA only)                          |
| `lora_rank`                     | LoRA rank (default: `16`)                                      |
| `max_train_steps`               | Total training steps                                           |
| `learning_rate`                 | Learning rate (default: `5e-5`)                                |
| `base_lora_learning_rate`       | LR for frozen base LoRA (STLoRA only)                          |
| `base_lora_anchor_weight`       | L2 anchor weight to prevent base LoRA drift                    |
| `train_batch_size`              | Batch size per GPU                                             |
| `checkpointing_steps`           | Save a checkpoint every N steps                                |
| `validation_steps`              | Run validation every N steps                                   |
| `resolution`                    | Training image resolution (default: `512`)                     |

---

## Evaluation

See `evaluation/example_evaluation.ipynb` for usage of the evaluation metrics:

- **CLIP / SigLIP / BLIP** — text-image alignment scores
- **LPIPS** — perceptual distance
- **DINOv2** — structural similarity
- **FaceID** — identity preservation
