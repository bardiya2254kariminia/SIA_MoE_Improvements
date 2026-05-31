"""
train_stlora_flux2_klein.py

STLoRA (Selective Token LoRA) trainer for the Image Analogy project,
built on top of FLUX.2-Klein with PPS (Partial Prompt Suppression) loss.

Architecture:
  - FLUX.2-Klein model stack (Qwen3 text encoder, Flux2 VAE, Flux2 Transformer)
  - STLoRA injected into context (text-stream) MM-DiT layers
  - Concat-mode analogy: A, A', B as conditions → denoise B'
  - PPS loss: frozen model + suppressed prompt vs LoRA model + full prompt + token mask
  - Ground truth is pose_only.png (edit1 only); edit2 is suppressed

Usage:
  python training/train_stlora_flux2_klein.py \
      --config training/configs/train_stlora_flux2_klein.yaml
"""

# ── Offline mode for Compute Canada GPU nodes (no internet access) ─────────
import argparse
import inspect as _inspect
import json
import math
import os
import pickle
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

# Ensure the visual_analogy package (parent of this training/ dir) is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import wandb
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from tqdm.auto import tqdm

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Now safe to import HF libraries ───────────────────────────────────────
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from diffusers.models import AutoencoderKLFlux2, Flux2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import make_image_grid
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from visual_analogy.models.selective_lora import SelectiveLoRALinear, BaseLoRALinear
from visual_analogy.models.moe_lora import TokenWiseGatedMoELoraLinear
from visual_analogy.utils import (
    resolve_hf_snapshot_path,
    save_selective_lora_state_dict,
    save_base_lora_state_dict,
    load_base_lora_state_dict,
    stlora_token_mask_ctx,
    inject_moe_lora_modules,
    save_moe_lora_state_dict,
    load_moe_lora_state_dict,
    collect_moe_lora_params,
    collect_moe_aux_losses,
    set_moe_lora_requires_grad,
)
from visual_analogy.utils.selective_lora import (
    inject_selective_lora_modules,
    inject_base_lora_modules,
    load_base_lora_from_peft_checkpoint,
    collect_base_lora_params,
    collect_stlora_params,
    set_base_lora_requires_grad,
    set_stlora_requires_grad,
    klein_find_substring_token_indices,
)

# ── Compatibility shim: strip unknown kwargs from set_module_tensor_to_device
try:
    from accelerate.utils import modeling as _accelerate_modeling
    _orig_smttd = _accelerate_modeling.set_module_tensor_to_device
    _known_smttd_params = set(_inspect.signature(_orig_smttd).parameters)
    _VAR_KEYWORD = _inspect.Parameter.VAR_KEYWORD
    _has_var_kw = any(
        p.kind == _VAR_KEYWORD
        for p in _inspect.signature(_orig_smttd).parameters.values()
    )
    if not _has_var_kw:
        def _smttd_compat(*args, **kwargs):
            filtered = {k: v for k, v in kwargs.items()
                        if k in _known_smttd_params}
            return _orig_smttd(*args, **filtered)
        _accelerate_modeling.set_module_tensor_to_device = _smttd_compat
        try:
            import diffusers.models.model_loading_utils as _dml
            _dml.set_module_tensor_to_device = _smttd_compat
        except Exception:
            pass
except Exception:
    pass
# ───────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# STLoRA target modules: context (text-stream) layers only.
# ---------------------------------------------------------------------------
STLORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_to_match(img, target_w, target_h):
    """Resize image to (target_w, target_h), both divisible by 16."""
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def resize_keep_ratio(img, max_side=512):
    """Resize so the longer side equals max_side, rounding both sides to multiples of 16."""
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def harmonize_images(a, a_prime, b, b_prime, max_side=512):
    """Normalise all four images for training.

    * A is resized to max_side (long-side) while preserving its aspect ratio,
      rounded to multiples of 16.
    * A' is snapped to A's exact pixel dimensions (same subject, same crop).
    * B is resized independently with the same max_side / ratio rule so its
      own aspect ratio is respected.
    * B' is snapped to B's exact pixel dimensions so that the denoising target
      always has the same spatial size as the B condition image.
    """
    a       = resize_keep_ratio(a, max_side)
    a_w, a_h = a.size
    a_prime = resize_to_match(a_prime, a_w, a_h)

    b       = resize_keep_ratio(b, max_side)
    b_w, b_h = b.size
    b_prime = resize_to_match(b_prime, b_w, b_h)

    return a, a_prime, b, b_prime


# ---------------------------------------------------------------------------
# Klein-specific helpers
# ---------------------------------------------------------------------------

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None,
                       timesteps=None, sigmas=None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        return scheduler.timesteps, num_inference_steps


# ---------------------------------------------------------------------------
# Base LoRA target modules (same as simple_lora training)
# ---------------------------------------------------------------------------
BASE_LORA_TARGET_MODULES = [
    "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj", "ff.net.2",
    "ff_context.net.0.proj", "ff_context.net.2",
]

# ---------------------------------------------------------------------------
# MoE LoRA target modules: image-stream MLP output layers.
# Analogous to VIRAL's img_mlp.net.2 on Qwen-Image-Edit.
# When use_MoE=True these get TokenWiseGatedMoELoraLinear instead of
# BaseLoRALinear, and are excluded from the BASE_LORA injection.
# Override via YAML key "moe_target_modules" if needed.
# ---------------------------------------------------------------------------
MOE_LORA_TARGET_MODULES_DEFAULT = ["ff.net.2"]

# ---------------------------------------------------------------------------
# Dataset — A'=total_changes.png, B'=pose_only.png, loads prompt.json
# ---------------------------------------------------------------------------

class ImageAnalogyDataset(Dataset):
    """Image analogy dataset for STLoRA training.

    Each stem folder has:
      - input.png           (before image)
      - total_changes.png   (after image with both edit1 + edit2)
      - pose_only.png       (after image with edit1 only)
      - prompt.json         (with edit1, edit2 fields)

    Sampling: stem1 → A=input, A'=total_changes (shows both edits)
              stem2 → B=input, B'=pose_only     (edit1 only, edit2 suppressed)
    """

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.styles = []

        for concept_dir in sorted(self.data_root.iterdir()):
            if not concept_dir.is_dir():
                continue
            edit_name = (
                re.sub(r'^\d+-', '', concept_dir.name)
                .replace('_', ' ').replace('-', ' ').lower().strip()
            )
            for style_dir in sorted(concept_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                stems = []
                for stem_dir in sorted(style_dir.iterdir()):
                    if not stem_dir.is_dir():
                        continue
                    input_path        = stem_dir / 'input.png'
                    total_changes_path = stem_dir / 'total_changes.png'
                    pose_only_path    = stem_dir / 'pose_only.png'
                    style_only_path   = stem_dir / 'style_only.png'
                    prompt_file       = stem_dir / 'prompt.json'
                    if not (input_path.exists() and total_changes_path.exists()
                            and (pose_only_path.exists() or style_only_path.exists())):
                        continue
                    edit1, edit2 = "", ""
                    if prompt_file.exists():
                        try:
                            data = json.loads(prompt_file.read_text())
                            edit1 = data.get("edit1", "")
                            edit2 = data.get("edit2", "")
                        except Exception:
                            pass
                    stems.append((
                        str(input_path), str(total_changes_path),
                        str(style_only_path) if style_only_path.exists() else None,
                        str(pose_only_path) if pose_only_path.exists() else None,
                        edit1, edit2,
                    ))

                if len(stems) >= 2:
                    self.styles.append((edit_name, stems))

        print(f"ImageAnalogyDataset: {len(self.styles)} style groups loaded.")

    def __len__(self):
        return len(self.styles)

    def __getitem__(self, idx):
        edit_name, stems = random.choice(self.styles)
        stem_a, stem_b = random.sample(stems, 2)
        # stem = (input, total_changes, style_only_or_None, pose_only_or_None, edit1, edit2)

        # A pair: input → (total_changes; partial matched to B side when available)
        a       = Image.open(stem_a[0]).convert('RGB')   # input.png
        a_total = Image.open(stem_a[1]).convert('RGB')   # total_changes.png
        a_edit1, a_edit2 = stem_a[4], stem_a[5]

        # B pair: input → pose_only or style_only (randomly chosen)
        # Preserve the original training mapping (unchanged from prior version):
        #   stem[2]  (style_only)  → suppress_index = 1
        #   stem[3]  (pose_only)   → suppress_index = 0
        b       = Image.open(stem_b[0]).convert('RGB')   # input.png
        b_total = Image.open(stem_b[1]).convert('RGB')   # total_changes.png
        b_edit1, b_edit2 = stem_b[4], stem_b[5]

        # Build {suppress_index: partial_path} using the same mapping for A and B.
        a_partials = {}  # {suppress_idx: path}
        if stem_a[2] is not None:
            a_partials[1] = stem_a[2]   # style_only → suppress_index 1
        if stem_a[3] is not None:
            a_partials[0] = stem_a[3]   # pose_only  → suppress_index 0
        b_partials = {}
        if stem_b[2] is not None:
            b_partials[1] = stem_b[2]
        if stem_b[3] is not None:
            b_partials[0] = stem_b[3]

        # Prefer a suppress_index available on BOTH stems so the base-LoRA
        # partial branch can use a matched partial A'.  If the intersection is
        # empty, fall back to any B-side option (A' partial will be None →
        # training loop falls back to A_total for that branch).
        common = sorted(set(a_partials) & set(b_partials))
        if common:
            suppress_index = random.choice(common)
        else:
            suppress_index = random.choice(list(b_partials))

        b_prime_path = b_partials[suppress_index]
        b_prime = Image.open(b_prime_path).convert('RGB')

        a_partial_path = a_partials.get(suppress_index)
        a_partial = (Image.open(a_partial_path).convert('RGB')
                     if a_partial_path is not None else None)

        pair_a_edits = [e for e in [a_edit1, a_edit2] if e]
        pair_b_edits = [e for e in [b_edit1, b_edit2] if e]

        combined_prompt = build_analogy_prompt(pair_a_edits, pair_b_edits)

        # Harmonize to A's canonical (max_side=512) size; a_partial uses the
        # same target so it is a drop-in replacement for a_total.
        a, a_total, b, b_prime = harmonize_images(a, a_total, b, b_prime, max_side=512)
        b_total = resize_to_match(b_total, *b.size)
        if a_partial is not None:
            a_partial = resize_to_match(a_partial, *a.size)

        return {
            'A':                a,
            'A_prime':          a_total,   # default A' (total_changes), kept for back-compat
            'A_prime_total':    a_total,
            'A_prime_partial':  a_partial, # matched partial or None
            'B':                b,
            'B_prime':          b_prime,
            'B_total':          b_total,
            'edit_name':        edit_name,
            'prompt':           combined_prompt,
            'pair_a_edits':     pair_a_edits,
            'pair_b_edits':     pair_b_edits,
            'suppress_index':   suppress_index,
        }


def collate_fn(batch):
    return {
        'A':               [item['A']            for item in batch],
        'A_prime':         [item['A_prime']         for item in batch],
        'A_prime_total':   [item['A_prime_total']   for item in batch],
        'A_prime_partial': [item['A_prime_partial'] for item in batch],
        'B':               [item['B']               for item in batch],
        'B_prime':         [item['B_prime']         for item in batch],
        'B_total':         [item['B_total']         for item in batch],
        'prompts':         [item['prompt']          for item in batch],
        'pair_a_edits':    [item['pair_a_edits']    for item in batch],
        'pair_b_edits':    [item['pair_b_edits']    for item in batch],
        'suppress_indices': [item['suppress_index'] for item in batch],
    }


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_analogy_prompt(pair_a_edits, pair_b_edits, suppress_indices=None):
    """Build the concat-mode prompt, removing edits at suppress_indices (0-based).

    Follows the same format as train_simple_lora_flux2_klein.py: only the A pair
    edits are listed (the model infers B→B' by analogy).

    Args:
        pair_a_edits: list of edit strings for A→A'
        pair_b_edits: list of edit strings for B→B' (unused in prompt, kept for API compat)
        suppress_indices: set/list of indices to omit (None = keep all)

    Returns:
        Prompt string.
    """
    suppress = set(suppress_indices or [])
    kept = [e for i, e in enumerate(pair_a_edits) if i not in suppress and e]
    edits_part = " ".join(f"Edit {n + 1}: {e}." for n, e in enumerate(kept))
    return (
        "Image 1 is the original and image 2 is the edited version. "
        f"{edits_part} "
        "Apply the same edits to image 3 to produce the output."
    )


# ---------------------------------------------------------------------------
# Token mask helpers (Qwen/Flux2 tokenizer)
# ---------------------------------------------------------------------------

def build_token_mask(full_prompt, edit_texts, tokenizer, device,
                     max_seq_len=512, use_chat_template: bool = True):
    """Build a (1, max_seq_len) bool mask covering tokens for each edit text.

    The mask is aligned against the **chat-templated** Qwen prompt — the
    exact sequence ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` feeds
    into the text encoder. Mask positions therefore line up with rows of
    ``prompt_embeds`` and with the ``encoder_hidden_states`` consumed by
    ``SelectiveLoRALinear``.

    Args:
        full_prompt:  full prompt string (raw, pre-template)
        edit_texts:   list of edit strings whose tokens should be masked
        tokenizer:    Qwen2TokenizerFast
        device:       target device
        max_seq_len:  must match max_length used in prompt encoding
        use_chat_template: kept as an escape hatch for forensic runs only.
            ``True`` (default) is the only correct setting for Flux2-Klein.

    Returns:
        torch.BoolTensor of shape (1, max_seq_len). If a particular edit
        cannot be located the function raises — silent zero-masks were the
        original STLoRA bug and must never recur.
    """
    mask = torch.zeros((1, max_seq_len), device=device, dtype=torch.bool)
    for text in edit_texts:
        if not text:
            continue
        try:
            indices = klein_find_substring_token_indices(
                full_prompt, text, tokenizer,
                max_length=max_seq_len,
                use_chat_template=use_chat_template,
            )
        except AssertionError as exc:
            # Substring not found in the prompt — almost always an edit-text
            # vs. prompt-builder mismatch. Loud failure is mandatory because
            # an empty mask silently disables STLoRA.
            raise RuntimeError(
                f"[TokenMask] Could not locate edit text in prompt:\n"
                f"  edit:   {text!r}\n"
                f"  prompt: {full_prompt!r}\n"
                f"  underlying: {exc}"
            ) from exc
        for idx in indices:
            if idx < max_seq_len:
                mask[0, idx] = True

    if edit_texts and not bool(mask.any()):
        raise RuntimeError(
            "[TokenMask] All edit texts produced empty positions — STLoRA "
            "would be a no-op. Check tokenizer / prompt format."
        )
    return mask


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    local_path = resolve_hf_snapshot_path(args.pretrained_model_name_or_path)
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        local_path, subfolder="tokenizer", local_files_only=True)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        local_path, subfolder="text_encoder", local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(
        local_path, subfolder="vae", local_files_only=True)
    transformer = Flux2Transformer2DModel.from_pretrained(
        local_path, subfolder="transformer", local_files_only=True)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        local_path, subfolder="scheduler", local_files_only=True)
    return tokenizer, text_encoder, vae, transformer, noise_scheduler


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_text_embeddings(prompts, tokenizer, text_encoder):
    prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=prompts,
    )
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(prompt_embeds.device)
    return prompt_embeds, text_ids


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ---------------------------------------------------------------------------
# Latent helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_single_image(pipe, img, resolution=512):
    """VAE-encode a PIL image with aspect ratio preserved.

    Scales so the longer side equals ``resolution``, rounds both sides to the
    nearest multiple of 16 (VAE requirement).  For the training data
    (384×512) this yields 384×512 instead of the distorted 512×512.
    """
    w, h = img.size
    scale = resolution / max(w, h)
    new_w = max(16, round(w * scale / 16) * 16)
    new_h = max(16, round(h * scale / 16) * 16)
    if (w, h) != (new_w, new_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    tensor = pipe.image_processor.preprocess(img, new_h, new_w)
    tensor = tensor.to(device=pipe._execution_device, dtype=torch.bfloat16)
    return pipe._encode_vae_image(tensor, generator=None)


@torch.no_grad()
def prepare_analogy_latents(pipe, batch, resolution=512):
    """Encode A, A', B, B' through VAE and prepare concat-mode latents + IDs.

    Returns:
        target_latents:    (B, N, C) packed B' latents
        condition_latents: (B, 3*N, C) packed [A, A', B] condition latents
        target_ids:        position IDs for target
        condition_ids:     position IDs for conditions
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(
        batch['A'], batch['A_prime'], batch['B'], batch['B_prime']
    ):
        lat_a       = encode_single_image(pipe, a,       resolution)
        lat_a_prime = encode_single_image(pipe, a_prime, resolution)
        lat_b       = encode_single_image(pipe, b,       resolution)
        lat_b_prime = encode_single_image(pipe, b_prime, resolution)

        target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
        cond_a  = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
        cond_arr.append(
            torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])

    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr,   dim=0),
        target_ids,
        condition_ids,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_peft_ckpt_into_stlora(transformer, ckpt_dir, device):
    """Initialize STLoRA A/B weights from a PEFT safetensors checkpoint."""
    ckpt_path = os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(ckpt_path):
        print(f"[PEFT Init] No file at {ckpt_path}, skipping checkpoint init.")
        return

    print(f"[PEFT Init] Loading PEFT weights from {ckpt_path}")
    peft_sd = load_file(ckpt_path)
    print(f"[PEFT Init] Found {len(peft_sd)} keys in PEFT checkpoint.")

    stlora_map = {
        name: mod
        for name, mod in transformer.named_modules()
        if isinstance(mod, SelectiveLoRALinear)
    }
    print(f"[PEFT Init] STLoRA has {len(stlora_map)} injected modules.")

    matched, skipped, mismatched = [], [], []

    for key, weight in peft_sd.items():
        norm = key
        for pfx in ("transformer.", "base_model.model."):
            if norm.startswith(pfx):
                norm = norm[len(pfx):]

        norm = re.sub(r'\.lora_([AB])\.[\w\-]+\.weight$', r'.lora_\1.weight', norm)

        if ".lora_A.weight" in norm:
            mod_name = norm.replace(".lora_A.weight", "")
            ab = "lora_A"
        elif ".lora_B.weight" in norm:
            mod_name = norm.replace(".lora_B.weight", "")
            ab = "lora_B"
        else:
            skipped.append(key)
            continue

        if mod_name not in stlora_map:
            skipped.append(key)
            continue

        target = getattr(stlora_map[mod_name], ab)
        if target.weight.shape != weight.shape:
            mismatched.append(f"{key}: peft={weight.shape} stlora={target.weight.shape}")
            continue

        with torch.no_grad():
            target.weight.copy_(weight.to(device))
        matched.append(key)

    print(
        f"[PEFT Init] matched={len(matched)}, "
        f"skipped={len(skipped)}, shape_mismatch={len(mismatched)}"
    )
    if mismatched:
        print("[PEFT Init] Shape mismatches:")
        for m in mismatched:
            print(f"  {m}")


def restore_training_state(optimizer, lr_scheduler, ckpt_dir, device):
    """Restore optimizer, lr_scheduler, and RNG state from an Accelerate checkpoint dir."""
    opt_path = os.path.join(ckpt_dir, "optimizer.bin")
    sch_path = os.path.join(ckpt_dir, "scheduler.bin")
    rng_path = os.path.join(ckpt_dir, "random_states_0.pkl")

    if os.path.exists(opt_path):
        optimizer.load_state_dict(torch.load(opt_path, map_location=device))
        print(f"[Ckpt] Restored optimizer from {opt_path}")
    if os.path.exists(sch_path):
        lr_scheduler.load_state_dict(torch.load(sch_path, map_location=device))
        print(f"[Ckpt] Restored lr_scheduler from {sch_path}")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng_state = pickle.load(f)
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        print(f"[Ckpt] Restored RNG state from {rng_path}")


# ---------------------------------------------------------------------------
# Accelerate save / load hooks
# ---------------------------------------------------------------------------

def get_save_model_hook(accelerator):
    def hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, Flux2Transformer2DModel):
                    save_selective_lora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "selective_lora.pt"),
                    )
                    save_base_lora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "base_lora.pt"),
                    )
                    # Save MoE weights if any MoE modules are present
                    if any(isinstance(m, TokenWiseGatedMoELoraLinear)
                           for m in unwrapped.modules()):
                        save_moe_lora_state_dict(
                            unwrapped,
                            os.path.join(output_dir, "moe_lora.pt"),
                        )
                    weights.pop()
    return hook


def get_load_model_hook(accelerator):
    def hook(models, input_dir):
        assert len(models) == 1, "Only transformer should be in the models list."
        transformer = models.pop()

        # Load STLoRA weights
        stlora_path = os.path.join(input_dir, "selective_lora.pt")
        state = torch.load(stlora_path, map_location=accelerator.device)
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys when loading STLoRA: {unexpected}")
        print(f"[Resume] Loaded STLoRA weights ({len(missing)} missing keys).")

        # Load base LoRA weights
        base_lora_path = os.path.join(input_dir, "base_lora.pt")
        if os.path.exists(base_lora_path):
            load_base_lora_state_dict(transformer, base_lora_path, accelerator.device)

        # Load MoE LoRA weights (present when use_MoE=True)
        moe_lora_path = os.path.join(input_dir, "moe_lora.pt")
        if os.path.exists(moe_lora_path):
            load_moe_lora_state_dict(transformer, moe_lora_path, accelerator.device)

        cast_training_params([transformer], dtype=torch.float32)
    return hook


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def log_analogy_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with STLoRA and log a comparison strip."""
    print(f"Running STLoRA analogy validation at step {global_step}...")

    pipe   = pipeline
    device = accelerator.device
    dtype  = torch.bfloat16

    # Pick a random sample
    edit_name, stems = random.choice(train_dataset.styles)
    stem_a, stem_b = random.sample(stems, 2)
    # stem = (input, total_changes, pose_only_or_None, style_only_or_None, edit1, edit2)
    pair_a_edits = [e for e in [stem_a[4], stem_a[5]] if e]
    pair_b_edits = [e for e in [stem_b[4], stem_b[5]] if e]

    # Randomly choose B' type:
    # stem_b[2] = style_only.png (edit2/style applied, edit1/pose suppressed → suppress index 1)
    # stem_b[3] = pose_only.png  (edit1/pose applied, edit2/style suppressed → suppress index 0)
    val_b_prime_options = []
    if stem_b[2] is not None:
        val_b_prime_options.append((stem_b[2], 1, 'style_only'))
    if stem_b[3] is not None:
        val_b_prime_options.append((stem_b[3], 0, 'pose_only'))
    val_b_prime_path, val_suppress_index, val_b_prime_type = random.choice(val_b_prime_options)

    prompt = build_analogy_prompt(pair_a_edits, pair_b_edits)
    suppressed_prompt = build_analogy_prompt(pair_a_edits, pair_b_edits,
                                             suppress_indices=[val_suppress_index])

    val_sample = {
        'A':       Image.open(stem_a[0]).convert('RGB'),   # input
        'A_prime': Image.open(stem_a[1]).convert('RGB'),   # total_changes
        'B':       Image.open(stem_b[0]).convert('RGB'),   # input
        'B_prime': Image.open(val_b_prime_path).convert('RGB'),
        'B_total': Image.open(stem_b[1]).convert('RGB'),   # B with both edits — GT for base-LoRA-total
        'edit_name': edit_name,
        'prompt':    prompt,
    }

    resolution = args.get("resolution", 512)

    # Encode images through VAE
    lat_a       = encode_single_image(pipe, val_sample['A'],       resolution)
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'], resolution)
    lat_b       = encode_single_image(pipe, val_sample['B'],       resolution)

    # Text embeddings
    prompt_embeds, text_ids = compute_text_embeddings(
        [prompt], tokenizer, text_encoder
    )
    guidance = None  # Klein: no guidance embeds

    patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

    def decode_latents(packed, h, w):
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def run_denoising(stlora_scale=1.0, suppress_indices=None,
                      prompt_embeds_override=None, text_ids_override=None,
                      mask_prompt_override=None):
        """Run denoising loop with STLoRA at the given uniform scale.

        suppress_indices: list of edit indices to suppress via the token mask.
            None / empty → use training default (val_suppress_index only).
            Pass all indices to suppress every edit (expected output = B).
        When stlora_scale == 0 the mask is unused (STLoRA is fully disabled).
        prompt_embeds_override / text_ids_override: optional precomputed
            embeddings to use instead of the default full-prompt embeddings.
        mask_prompt_override: the raw prompt text corresponding to the override
            embeddings (required for correct token-mask alignment when the
            override is used).
        """
        emb     = prompt_embeds_override if prompt_embeds_override is not None else prompt_embeds
        tids    = text_ids_override      if text_ids_override      is not None else text_ids
        raw_pt  = mask_prompt_override   if mask_prompt_override   is not None else prompt

        # Build the proper selective token mask (or None when STLoRA is off)
        if stlora_scale != 0.0:
            indices = suppress_indices if suppress_indices is not None \
                      else [val_suppress_index]
            texts_to_mask = [
                pair_a_edits[i]
                for i in indices
                if i < len(pair_a_edits) and pair_a_edits[i]
            ]
            max_seq_len = emb.shape[1]
            stlora_mask = build_token_mask(
                raw_pt, texts_to_mask, tokenizer, device,
                max_seq_len=max_seq_len,
            )  # (1, max_seq_len)
        else:
            stlora_mask = None

        for m in pipe.transformer.modules():
            if isinstance(m, SelectiveLoRALinear):
                m.set_scaling(stlora_scale)

        cond_a  = Flux2KleinPipeline._pack_latents(lat_a)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
        cond_b  = Flux2KleinPipeline._pack_latents(lat_b)
        condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
        gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

        target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
        condition_ids = Flux2KleinPipeline._prepare_image_ids(
            [lat_a, lat_a_prime, lat_b]).to(device)
        combined_ids = torch.cat([target_ids, condition_ids], dim=1)

        image_seq_len = gen_latents.shape[1]
        num_steps = args.num_val_inference_steps
        sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
        mu = compute_empirical_mu(image_seq_len, num_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu)

        with stlora_token_mask_ctx(pipe.transformer, stlora_mask, disable_mask_after=False):
            for t in timesteps:
                latent_model_input = torch.cat(
                    [gen_latents, condition_latents], dim=1)
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(1).to(dtype) / 1000,
                    guidance=guidance,
                    encoder_hidden_states=emb,
                    txt_ids=tids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0][:, :gen_latents.shape[1]]
                gen_latents = pipe.scheduler.step(
                    noise_pred, t, gen_latents, return_dict=False)[0]

        for m in pipe.transformer.modules():
            if isinstance(m, SelectiveLoRALinear):
                m.reset_scaling()

        return decode_latents(gen_latents, patch_h, patch_w)

    # Precompute embeddings for the suppressed prompt (used by Base LoRA
    # partial: no STLoRA, no token mask — the suppressed edit is just stripped
    # from the prompt text, mirroring the training protocol).
    suppressed_prompt_embeds, suppressed_text_ids = compute_text_embeddings(
        [suppressed_prompt], tokenizer, text_encoder
    )

    # Run inference:
    #   1. base LoRA total           — full prompt, STLoRA off, no mask (mirrors base-LoRA-total training)
    #   2. base LoRA, partial prompt — suppressed edit stripped from text, STLoRA off (mirrors base-LoRA-partial training)
    #   3. STLoRA, one edit          — training objective (full prompt + mask)
    #   4. STLoRA, all edits         — sanity check: output should resemble B
    pred_base_total     = run_denoising(stlora_scale=0.0)
    pred_base_partial   = run_denoising(
        stlora_scale=0.0,
        prompt_embeds_override=suppressed_prompt_embeds,
        text_ids_override=suppressed_text_ids,
        mask_prompt_override=suppressed_prompt,
    )
    pred_with_lora      = run_denoising(stlora_scale=1.0)
    all_indices         = list(range(len(pair_a_edits)))
    pred_all_suppressed = run_denoising(stlora_scale=1.0, suppress_indices=all_indices)

    # Resize all images with aspect ratio preserved (max side = resolution).
    a_img       = resize_keep_ratio(val_sample['A'],       resolution)
    a_prime_img = resize_keep_ratio(val_sample['A_prime'], resolution)
    b_img       = resize_keep_ratio(val_sample['B'],       resolution)
    b_prime_gt  = resize_keep_ratio(val_sample['B_prime'], resolution)
    b_total_img = resize_keep_ratio(val_sample['B_total'], resolution)

    # Cell dimensions from the first image (all training images share one AR).
    cell_w, cell_h = a_img.size

    if accelerator.is_main_process:
        vis_dir = Path(args.output_dir) / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Three-row layout:
        #   row 1 — inputs + global ground truth
        #   row 2 — per-column ground truth (what each row-3 prediction targets)
        #   row 3 — model outputs
        gt_label = f"B' GT ({val_b_prime_type})"
        row1_imgs   = [a_img, a_prime_img, b_img, b_prime_gt]
        row1_labels = ["A", "A'", "B", gt_label]
        # Row 2: the GT each row-3 prediction is trying to reproduce.
        #   col 1 (base LoRA total prompt)   → B_total (both edits)
        #   col 2 (base LoRA partial prompt) → B' GT  (kept edit only)
        #   col 3 (STLoRA)                   → B' GT  (kept edit only)
        #   col 4 (all suppressed)           → B      (unchanged input)
        row2_imgs   = [b_total_img, b_prime_gt, b_prime_gt, b_img]
        row2_labels = ["GT: B_total\n(both edits)",
                       f"GT: B' ({val_b_prime_type})",
                       f"GT: B' ({val_b_prime_type})",
                       "GT: B\n(unchanged)"]
        row3_imgs   = [pred_base_total, pred_base_partial,
                       pred_with_lora, pred_all_suppressed]
        row3_labels = ["B' (base LoRA\ntotal prompt)",
                       "B' (base LoRA\npartial prompt)",
                       "B' (STLoRA)",
                       "B' (all suppressed\n→ expect B)"]
        n_cols = max(len(row1_imgs), len(row2_imgs), len(row3_imgs))

        import textwrap
        from PIL import ImageFont

        # Try to load a legible TrueType font; fall back to PIL default
        font_size   = 20
        label_font_size = 22
        _font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        ]
        font = None
        label_font = None
        for _fp in _font_candidates:
            if os.path.exists(_fp):
                font       = ImageFont.truetype(_fp, font_size)
                label_font = ImageFont.truetype(_fp, label_font_size)
                break

        strip_w      = cell_w * n_cols
        label_height = 28   # per-row column-label banner

        # Estimate chars per line across the full strip width
        # TrueType: ~0.55 × font_size px per char; default bitmap: ~6 px/char
        approx_char_w = (font_size * 0.55) if font else 6
        chars_per_line = max(60, int((strip_w - 8) / approx_char_w))

        def wrap(text):
            return "\n".join(textwrap.wrap(text, chars_per_line))

        # Show the suppressed edit as it appears in the prompt, with its prompt position number
        _supp_edit_text = pair_a_edits[val_suppress_index] if val_suppress_index < len(pair_a_edits) else ""
        _supp_edits_str = f"Edit {val_suppress_index + 1}: {_supp_edit_text}." if _supp_edit_text else ""
        # All-suppressed run: every edit in the prompt is masked.
        _all_supp_str = " | ".join(
            f"Edit {i + 1}: {e}." for i, e in enumerate(pair_a_edits) if e
        )
        prompt_lines      = wrap(f"Full prompt:       {prompt}")
        partial_lines     = wrap(f"Partial prompt:    {suppressed_prompt}")
        supp_lines        = wrap(f"Suppressed:        {_supp_edits_str}")
        all_supp_lines    = wrap(f"All suppressed:    {_all_supp_str}")

        # Count lines to size text area dynamically
        line_h      = font_size + 4   # line height with spacing
        n_lines     = (prompt_lines.count("\n") + 1
                       + partial_lines.count("\n") + 1
                       + supp_lines.count("\n") + 1
                       + all_supp_lines.count("\n") + 1)
        text_height = n_lines * line_h + 16   # +16 for top/bottom padding

        # One banner + image per row, three rows total
        n_rows      = 3
        row_block_h = label_height + cell_h
        strip_h     = n_rows * row_block_h + text_height
        strip       = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw        = ImageDraw.Draw(strip)

        def _paste_row(row_imgs, row_labels, y0):
            """Draw one row of images with column labels at y=y0."""
            for i, (img, label) in enumerate(zip(row_imgs, row_labels)):
                cell_img = resize_keep_ratio(img, resolution)
                cw, ch = cell_img.size
                x_off = i * cell_w + (cell_w - cw) // 2
                y_off = y0 + label_height + (cell_h - ch) // 2
                strip.paste(cell_img, (x_off, y_off))
                draw.text((i * cell_w + 6, y0 + 4), label, fill=(0, 0, 0),
                          font=label_font if label_font else None)

        _paste_row(row1_imgs, row1_labels, y0=0)
        _paste_row(row2_imgs, row2_labels, y0=row_block_h)
        _paste_row(row3_imgs, row3_labels, y0=2 * row_block_h)

        # Draw prompts below the images — full-width, wrapped
        y_text = n_rows * row_block_h + 8
        draw.multiline_text(
            (8, y_text), prompt_lines, fill=(0, 0, 160),
            font=font if font else None, spacing=4,
        )
        y_partial = y_text + (prompt_lines.count("\n") + 1) * line_h + 4
        draw.multiline_text(
            (8, y_partial), partial_lines, fill=(0, 110, 0),
            font=font if font else None, spacing=4,
        )
        y_supp = y_partial + (partial_lines.count("\n") + 1) * line_h + 4
        draw.multiline_text(
            (8, y_supp), supp_lines, fill=(160, 0, 0),
            font=font if font else None, spacing=4,
        )
        y_all_supp = y_supp + (supp_lines.count("\n") + 1) * line_h + 4
        draw.multiline_text(
            (8, y_all_supp), all_supp_lines, fill=(120, 0, 120),
            font=font if font else None, spacing=4,
        )

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

        # Save prompts to text file alongside the strip
        prompt_path = vis_dir / f"step_{global_step:06d}_prompt.txt"
        prompt_path.write_text(
            f"edit: {edit_name}\n\n"
            f"full prompt:\n{prompt}\n\n"
            f"suppressed prompt (one edit):\n{suppressed_prompt}\n\n"
            f"all-suppressed edits:\n{_all_supp_str}\n"
        )

        # Optional slider sweep — commented out for now
        # slider_scales = getattr(args, "validation_lora_scales", [0.0, 0.5, 1.0, 1.5])
        # ...


        # wandb.log({
        #     "val/strip":        wandb.Image(strip, caption=f"step {global_step} | {edit_name}"),
        #     "val/slider_sweep": wandb.Image(slider_strip, caption="slider sweep"),
        #     "val/prompt":       wandb.Html(f"<b>edit:</b> {edit_name}<br><b>prompt:</b> {prompt}"),
        #     "train/step":       global_step,
        # })

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(print_args=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--use_MoE",
        action="store_true",
        default=False,
        help="Override config: use Mixture-of-Experts LoRA on image-stream MLP layers.",
    )
    cli = parser.parse_args()
    args = OmegaConf.load(cli.config)
    # CLI --use_MoE overrides the YAML value (if YAML omits it, default False)
    if cli.use_MoE:
        args.use_MoE = True
    elif not hasattr(args, "use_MoE"):
        args.use_MoE = False
    if print_args:
        print("=" * 40, "Arguments", "=" * 40)
        for k in args:
            print(f"  {k}: {args[k]}")
        print("=" * 91)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        dataloader_config=dataloader_config,
    )

    if accelerator.is_main_process:
        wandb_key = os.environ.get("WANDB_API_KEY", "")
        if wandb_key:
            wandb.login(key=wandb_key)
        wandb.init(
            project=args.tracker_name,
            name=args.wandb_run_name,
            config=OmegaConf.to_container(args, resolve=True),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*",   step_metric="train/step")

    set_seed(args.seed)

    # ---- Load models ----
    tokenizer, text_encoder, vae, transformer, noise_scheduler = load_models(args)

    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)

    weight_dtype = torch.bfloat16
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)

    # ---- Resolve MoE config ----
    use_moe: bool = bool(getattr(args, "use_MoE", False))
    moe_target_modules: list[str] = list(
        getattr(args, "moe_target_modules", MOE_LORA_TARGET_MODULES_DEFAULT)
    )
    moe_lora_rank: int = int(getattr(args, "moe_lora_rank", args.lora_rank))
    moe_lora_alpha: float = float(getattr(args, "moe_lora_alpha", moe_lora_rank))
    num_experts: int = int(getattr(args, "num_experts", 4))
    moe_top_k: int = int(getattr(args, "moe_top_k", 1))
    moe_aux_loss_weight: float = float(getattr(args, "moe_aux_loss_weight", 0.005))
    moe_lora_lr: float = float(
        getattr(args, "moe_lora_learning_rate",
                getattr(args, "learning_rate", 5e-5))
    )

    if use_moe:
        print(f"[MoE LoRA] Enabled — targets: {moe_target_modules}, "
              f"experts={num_experts}, rank={moe_lora_rank}, top_k={moe_top_k}")
        # Base LoRA covers all modules NOT handled by MoE
        base_only_modules = [
            m for m in BASE_LORA_TARGET_MODULES
            if not any(m.endswith(sfx) for sfx in moe_target_modules)
        ]
    else:
        base_only_modules = BASE_LORA_TARGET_MODULES

    # ---- Inject BaseLoRALinear on ALL base LoRA target modules ----
    base_replaced = inject_base_lora_modules(
        transformer, base_only_modules,
        r=args.lora_rank, alpha=args.lora_rank, dropout=0.0,
    )
    print(f"[Base LoRA] Injected {len(base_replaced)} BaseLoRALinear modules")

    # ---- Inject MoE LoRA on image-stream MLP layers (use_MoE only) ----
    if use_moe:
        moe_replaced = inject_moe_lora_modules(
            transformer, moe_target_modules,
            num_experts=num_experts,
            r=moe_lora_rank,
            lora_alpha=moe_lora_alpha,
            lora_dropout=getattr(args, "lora_dropout", 0.0),
            top_k=moe_top_k,
        )
        print(f"[MoE LoRA] Injected {len(moe_replaced)} TokenWiseGatedMoELoraLinear modules:")
        for name in moe_replaced:
            print(f"  {name}")

    # ---- Load pre-trained base LoRA weights from PEFT checkpoint ----
    #      When Stage-1 ran with use_MoE=True the checkpoint dir also
    #      contains moe_lora.pt alongside base_lora.pt.
    load_base_lora_from_peft_checkpoint(
        transformer, args.base_lora_path, device="cpu")

    if use_moe:
        base_moe_ckpt = os.path.join(args.base_lora_path, "moe_lora.pt")
        if os.path.exists(base_moe_ckpt):
            print(f"[MoE LoRA] Loading pre-trained MoE weights from {base_moe_ckpt}")
            load_moe_lora_state_dict(transformer, base_moe_ckpt, device="cpu")
        else:
            print("[MoE LoRA] No pre-trained MoE checkpoint — starting from random init.")

    # ---- Inject STLoRA on context targets (wraps BaseLoRALinear where applicable) ----
    replaced = inject_selective_lora_modules(
        transformer, STLORA_TARGET_MODULES,
        r=args.lora_rank, alpha=args.lora_rank, dropout=args.lora_dropout,
    )
    print(f"[STLoRA] Injected {len(replaced)} SelectiveLoRALinear modules:")
    for name in replaced:
        print(f"  {name}")

    # ---- Optionally initialise STLoRA from a PEFT checkpoint ----
    init_ckpt = getattr(args, "init_from_lora_ckpt", None)
    if init_ckpt:
        load_peft_ckpt_into_stlora(transformer, init_ckpt, accelerator.device)

    transformer.enable_gradient_checkpointing()

    # ---- Register Accelerate save/load hooks ----
    accelerator.register_save_state_pre_hook(get_save_model_hook(accelerator))
    accelerator.register_load_state_pre_hook(get_load_model_hook(accelerator))

    # ---- Keep LoRA params in float32 ----
    cast_training_params([transformer], dtype=torch.float32)

    # ---- Snapshot base LoRA weights as L2 anchor (prevents drift) ----
    base_lora_anchor = {}
    for mod_name, mod in transformer.named_modules():
        if isinstance(mod, BaseLoRALinear):
            base_lora_anchor[(mod_name, 'A')] = mod.lora_A.weight.detach().clone()
            base_lora_anchor[(mod_name, 'B')] = mod.lora_B.weight.detach().clone()
    base_lora_anchor_weight = getattr(args, "base_lora_anchor_weight", 1e-4)
    print(f"[L2 Anchor] Saved {len(base_lora_anchor)} base LoRA param tensors, "
          f"λ = {base_lora_anchor_weight}")

    # ---- Two optimizer param groups: STLoRA (normal LR) + base LoRA (low LR) ----
    base_lora_lr = getattr(args, "base_lora_learning_rate", 5e-6)
    stlora_params = collect_stlora_params(transformer)
    base_lora_params = collect_base_lora_params(transformer)
    print(f"[Optimizer] STLoRA params: {len(stlora_params)}, "
          f"base LoRA params: {len(base_lora_params)}")

    param_groups = [
        {"params": stlora_params,    "lr": args.learning_rate},
        {"params": base_lora_params, "lr": base_lora_lr},
    ]

    if use_moe:
        moe_params = collect_moe_lora_params(transformer)
        print(f"[Optimizer] MoE LoRA params: {len(moe_params)}")
        param_groups.append({"params": moe_params, "lr": moe_lora_lr})

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # ---- Dataset & dataloader ----
    train_dataset = ImageAnalogyDataset(args.data_root)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        shuffle=True,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # ---- Optionally restore optimizer/scheduler/RNG from init checkpoint ----
    # init_ckpt = getattr(args, "init_from_lora_ckpt", None)
    if init_ckpt and getattr(args, "init_optimizer_from_ckpt", False):
        restore_training_state(optimizer, lr_scheduler, init_ckpt, accelerator.device)

    len_train_dataset          = len(train_dataset)
    len_train_dataloader       = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs           = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running STLoRA training (Flux2-Klein) *****")
    print(f"  Num examples             = {len_train_dataset}")
    print(f"  Num epochs               = {num_train_epochs}")
    print(f"  Batch size per device    = {args.train_batch_size}")
    print(f"  Gradient accumulation    = {args.gradient_accumulation_steps}")
    print(f"  Total optimisation steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step  = 0
    first_epoch  = 0

    # ---- Resume from STLoRA checkpoint ----
    if args.resume_from_checkpoint:
        dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if dirs else None
        if path is None:
            print(f"No checkpoint found in {args.output_dir}. Starting fresh.")
            args.resume_from_checkpoint = False
        else:
            print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step         = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch         = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # ---- Build pipeline for VAE encoding & validation ----
    pipeline = Flux2KleinPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(transformer, keep_fp32_wrapper=False),
    )
    unwrapped_transformer = accelerator.unwrap_model(transformer)

    transformer.train()

    base_lora_step_ratio = getattr(args, "base_lora_step_ratio", 0.333)
    print(f"[Training] base_lora_step_ratio = {base_lora_step_ratio}")

    # ====================================================================
    # Training loop
    # ====================================================================
    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                device = accelerator.device
                pair_a_edits = batch['pair_a_edits']  # List[List[str]]
                pair_b_edits = batch['pair_b_edits']

                # ---- Decide step type ----
                # Distribution (single draw):
                #   [0.00, 0.25) → base LoRA partial  (25 %)
                #   [0.25, 0.55) → base LoRA total    (30 %)
                #   [0.55, 0.90) → STLoRA partial     (35 %)
                #   [0.90, 1.00) → STLoRA all-supp    (10 %)
                _r = random.random()
                is_all_suppressed_step = _r >= 0.90
                is_base_lora_step      = _r < 0.55           # covers both base-LoRA cases
                use_total_change       = 0.25 <= _r < 0.55   # base LoRA total only
                is_base_lora_partial   = is_base_lora_step and not use_total_change

                # ---- Toggle trainability ----
                if is_base_lora_step:
                    set_stlora_requires_grad(unwrapped_transformer, False)
                    set_base_lora_requires_grad(unwrapped_transformer, True)
                    if use_moe:
                        # MoE (on image-stream layers) trains alongside base LoRA
                        set_moe_lora_requires_grad(unwrapped_transformer, True)
                else:
                    # Both STLoRA steps (partial + all-suppressed) train STLoRA.
                    set_stlora_requires_grad(unwrapped_transformer, True)
                    set_base_lora_requires_grad(unwrapped_transformer, False)
                    if use_moe:
                        # MoE routing must also adapt during STLoRA steps so that
                        # experts remain consistent with the selective token masking.
                        set_moe_lora_requires_grad(unwrapped_transformer, True)

                # ---- Choose target image ----
                # Base-LoRA-partial also swaps A_prime to the matched partial
                # (same "kept edit" as B_prime).  Falls back to A_total where
                # the matched partial is unavailable for a given sample.
                # STLoRA branches keep A_prime = total_changes unchanged.
                if use_total_change:
                    target_batch = {**batch, 'B_prime': batch['B_total']}
                elif is_all_suppressed_step:
                    # All edits suppressed → model should reproduce B unchanged.
                    target_batch = {**batch, 'B_prime': batch['B']}
                elif is_base_lora_partial:
                    a_prime_matched = [
                        ap if ap is not None else at
                        for ap, at in zip(
                            batch['A_prime_partial'], batch['A_prime_total']
                        )
                    ]
                    target_batch = {**batch, 'A_prime': a_prime_matched}
                else:
                    target_batch = batch

                # ---- Encode images (concat mode) ----
                target_latents, condition_latents, target_ids, condition_ids = \
                    prepare_analogy_latents(pipeline, target_batch,
                                           resolution=args.get("resolution", 512))

                target_ids    = target_ids.to(device)
                condition_ids = condition_ids.to(device)
                bsz = target_latents.shape[0]

                target_ids    = target_ids.expand(bsz, -1, -1)
                condition_ids = condition_ids.expand(bsz, -1, -1)
                combined_ids  = torch.cat([target_ids, condition_ids], dim=1)

                # ---- Sample timestep and add noise ----
                u = compute_density_for_timestep_sampling("none", batch_size=bsz)
                indices = (
                    u * (args.max_train_timesteps - args.min_train_timesteps)
                    + args.min_train_timesteps
                ).long().clamp(0, len(noise_scheduler.timesteps) - 1)
                timesteps = noise_scheduler.timesteps[indices].to(device)
                sigmas = get_sigmas(
                    noise_scheduler, timesteps, device,
                    n_dim=target_latents.ndim, dtype=target_latents.dtype,
                )
                noise = torch.randn_like(target_latents)
                noisy_model_input = (1.0 - sigmas) * target_latents + sigmas * noise

                # Klein: no guidance embeds
                guidance = None

                orig_shape = noisy_model_input.shape
                latent_model_input = torch.cat(
                    [noisy_model_input, condition_latents], dim=1)

                if is_base_lora_step:
                    # ==== Base LoRA step: standard flow matching loss ====
                    # Base LoRA has no selective suppression mechanism, so the
                    # suppressed edit is removed from the prompt text entirely
                    # (instead of masking tokens inside a full prompt).  STLoRA
                    # is fully disabled (mask=None) for the whole forward.
                    if use_total_change:
                        # GT=total_change → keep every edit in the prompt.
                        prompts_to_encode = [
                            build_analogy_prompt(ae, be)
                            for ae, be in zip(pair_a_edits, pair_b_edits)
                        ]
                    else:
                        # GT=style_only/pose_only → strip the suppressed edit
                        # from the prompt so only the retained edit remains.
                        prompts_to_encode = [
                            build_analogy_prompt(
                                ae, be,
                                suppress_indices=[idx],
                            )
                            for ae, be, idx in zip(
                                pair_a_edits, pair_b_edits,
                                batch['suppress_indices'],
                            )
                        ]
                    prompt_embeds, text_ids = compute_text_embeddings(
                        prompts_to_encode, tokenizer, text_encoder,
                    )

                    with stlora_token_mask_ctx(transformer, None):
                        model_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_ids,
                            return_dict=False,
                        )[0][:, :orig_shape[1]]

                    flow_target = noise - target_latents
                    weighting = compute_loss_weighting_for_sd3("none", sigmas=sigmas)
                    loss = torch.mean(
                        (weighting.float()
                         * (model_pred.float() - flow_target.float()) ** 2
                         ).reshape(bsz, -1),
                        dim=1,
                    ).mean()

                else:
                    # ==== STLoRA step: flow matching loss with token mask ====
                    # GT = style_only/pose_only (partial edit)  OR  B (all suppressed)
                    if is_all_suppressed_step:
                        # Mask every edit token — model must output B unchanged.
                        suppress_per_sample = [
                            list(range(len(ae))) for ae in pair_a_edits
                        ]
                    else:
                        suppress_per_sample = [
                            [idx] for idx in batch['suppress_indices']
                        ]

                    full_prompts = [
                        build_analogy_prompt(ae, be)
                        for ae, be in zip(pair_a_edits, pair_b_edits)
                    ]

                    prompt_embeds, text_ids = compute_text_embeddings(
                        full_prompts, tokenizer, text_encoder,
                    )

                    # ---- Build (bsz, max_seq_len) token mask for suppressed edits ----
                    max_seq_len = prompt_embeds.shape[1]
                    tokens_mask = torch.zeros((bsz, max_seq_len), device=device, dtype=torch.bool)
                    for i in range(bsz):
                        texts_to_mask = []
                        for j in suppress_per_sample[i]:
                            if j < len(pair_a_edits[i]) and pair_a_edits[i][j]:
                                texts_to_mask.append(pair_a_edits[i][j])
                        single_mask = build_token_mask(
                            full_prompts[i], texts_to_mask, tokenizer, device,
                            max_seq_len=max_seq_len,
                        )
                        tokens_mask[i] = single_mask[0]

                    # ---- Forward with STLoRA active + token mask ----
                    with stlora_token_mask_ctx(transformer, tokens_mask, disable_mask_after=False):
                        model_pred = transformer(
                            hidden_states=latent_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_ids,
                            return_dict=False,
                        )[0][:, :orig_shape[1]]

                    flow_target = noise - target_latents
                    weighting = compute_loss_weighting_for_sd3("none", sigmas=sigmas)
                    loss = torch.mean(
                        (weighting.float()
                         * (model_pred.float() - flow_target.float()) ** 2
                         ).reshape(bsz, -1),
                        dim=1,
                    ).mean()

                # ---- L2 anchor: prevent base LoRA drift on base_lora steps ----
                if is_base_lora_step and base_lora_anchor_weight > 0:
                    anchor_loss = torch.tensor(0.0, device=device)
                    for mod_name, mod in unwrapped_transformer.named_modules():
                        if isinstance(mod, BaseLoRALinear):
                            for ab, param in [('A', mod.lora_A.weight),
                                              ('B', mod.lora_B.weight)]:
                                anchor = base_lora_anchor[(mod_name, ab)]
                                anchor_loss = anchor_loss + torch.sum(
                                    (param.float() - anchor.to(device).float()) ** 2)
                    print(f"[L2 Anchor] base LoRA anchor loss: {anchor_loss.item():.4f}")
                    print("loss before anchor:", loss.item())
                    loss = loss + base_lora_anchor_weight * anchor_loss

                # ---- MoE load-balancing aux loss ----
                if use_moe:
                    moe_aux = collect_moe_aux_losses(unwrapped_transformer)
                    if moe_aux is not None:
                        loss = loss + moe_aux_loss_weight * moe_aux

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---- After gradient sync ----
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    log_dict = {
                        "train/loss":  loss.detach().item(),
                        "train/lr":    lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step":  global_step,
                    }
                    if use_moe:
                        moe_aux_log = collect_moe_aux_losses(
                            accelerator.unwrap_model(transformer))
                        if moe_aux_log is not None:
                            log_dict["train/moe_aux_loss"] = moe_aux_log.detach().item()
                    wandb.log(log_dict)
                    if (global_step % args.checkpointing_steps == 0
                            or global_step == args.max_train_steps):
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if (global_step % args.validation_steps == 0
                            or global_step == args.max_train_steps):
                        pipeline.transformer = accelerator.unwrap_model(
                            transformer, keep_fp32_wrapper=False)
                        log_analogy_validation(
                            pipeline, train_dataset,
                            tokenizer, text_encoder,
                            args, accelerator, global_step,
                        )
                        transformer.train()

            if is_all_suppressed_step:
                step_type = "stlora_all_supp"
            elif use_total_change:
                step_type = "base_total"
            elif is_base_lora_step:
                step_type = "base_partial"
            else:
                step_type = "stlora"
            logs = {
                "loss": loss.detach().item(),
                "lr":   lr_scheduler.get_last_lr()[0],
                "step_type": step_type,
            }
            if is_base_lora_step and base_lora_anchor_weight > 0:
                logs["anchor_loss"] = (base_lora_anchor_weight * anchor_loss).detach().item()
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
