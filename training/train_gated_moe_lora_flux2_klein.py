# ── Offline mode for Compute Canada GPU nodes (no internet access) ─────────
# These env vars MUST be set BEFORE any HuggingFace / transformers / diffusers
# imports, because huggingface_hub reads them at import time.
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig
from accelerate.utils import set_seed
from accelerate.utils import DataLoaderConfiguration
from omegaconf import OmegaConf
from contextlib import nullcontext
import re
import random
import wandb
from tqdm.auto import tqdm
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import make_image_grid
import math
from diffusers.optimization import get_scheduler
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.models import Flux2Transformer2DModel, AutoencoderKLFlux2
from diffusers import Flux2KleinPipeline
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import json
import inspect
import argparse
import os
import sys
import inspect as _inspect

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ── Now safe to import HF libraries ───────────────────────────────────────

# ── Compatibility shim: strip unknown kwargs from set_module_tensor_to_device
# diffusers>=0.30 passes kwargs (non_blocking, clear_cache, …) that older
# accelerate builds don't accept. Strip any unknown ones instead of hardcoding.
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

from visual_analogy.models.selective_lora import BaseLoRALinear
from visual_analogy.models.moe_lora import TokenWiseGatedMoELoraLinear
from visual_analogy.utils.selective_lora import (
    inject_base_lora_modules,
    save_base_lora_state_dict,
    collect_base_lora_params,
)
from visual_analogy.utils.moe_lora import (
    inject_moe_lora_modules,
    save_moe_lora_state_dict,
    collect_moe_lora_params,
    collect_moe_aux_losses,
)

# ---------------------------------------------------------------------------
# MoE LoRA target modules: image-stream MLP output (analogous to VIRAL's
# img_mlp.net.2).  Override via YAML key "moe_target_modules".
# ---------------------------------------------------------------------------
MOE_LORA_TARGET_MODULES_DEFAULT = ["ff.net.2"]

LORA_TARGET_MODULES_ALL = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


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


def retrieve_timesteps(scheduler, num_inference_steps=None, device=None, timesteps=None, sigmas=None, **kwargs):
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


def resize_to_match(img, target_w, target_h):
    """Resize image to (target_w, target_h), ensuring both are divisible by 16
    (required for VAE 8x compression + 2x packing)."""
    target_w = (target_w // 16) * 16
    target_h = (target_h // 16) * 16
    return img.resize((target_w, target_h), Image.LANCZOS)


def harmonize_images(a, a_prime, b, b_prime):
    """Resize all 4 images to the same size (use A's dimensions as reference).
    Ensures the 2x2 grid has consistent cell sizes."""
    target_w, target_h = a.size
    a_prime = resize_to_match(a_prime, target_w, target_h)
    b = resize_to_match(b,       target_w, target_h)
    b_prime = resize_to_match(b_prime, target_w, target_h)
    return a, a_prime, b, b_prime


class ImageAnalogyDataset(Dataset):
    """Dataset for image analogy training.

    Each stem folder contains exactly one before→after pair: a.png (before)
    and a_prime.png (after). To form an analogy (A:A'::B:B'), we pick 2 different
    stems from the same style subdir: first stem → A/A', second stem → B/B'.
    """

    def __init__(self, data_root, mode="concat"):
        self.mode = mode
        self.data_root = Path(data_root)
        self.styles = []  # list of (edit_name, stems_list)

        for concept_dir in sorted(self.data_root.iterdir()):
            if not concept_dir.is_dir():
                continue
            edit_name = re.sub(
                r'^\d+-', '', concept_dir.name).replace('_', ' ').replace('-', ' ').lower().strip()

            for style_dir in sorted(concept_dir.iterdir()):
                if not style_dir.is_dir():
                    continue
                stems = []
                for stem_dir in sorted(style_dir.iterdir()):
                    if not stem_dir.is_dir():
                        continue
                    a = stem_dir / 'input.png'
                    a_prime = stem_dir / 'total_changes.png'
                    prompt_file = stem_dir / 'prompt.json'
                    if a.exists() and a_prime.exists():
                        edit1, edit2 = "edit the image", "edit the image"

                        # TODO make sure to add prompt.json later
                        # TODO code baraye inke Qwen tooye har folder prompt.json besaze
                        prompt_file = stem_dir / 'prompt.json'
                        if torch.rand(1).item() < 0.8 and prompt_file.exists():
                            try:
                                prompt_data = json.loads(
                                    prompt_file.read_text())
                                edit1 = prompt_data.get("edit1", "")
                                edit2 = prompt_data.get("edit2", "")
                            except Exception:
                                pass
                        stems.append((str(a), str(a_prime), edit1, edit2))

                if len(stems) >= 2:
                    self.styles.append((edit_name, stems))

        print(f"ImageAnalogyDataset: {len(self.styles)} style groups")

    def __len__(self):
        return len(self.styles)

    def __getitem__(self, idx):
        edit_name, stems = random.choice(self.styles)
        # Pick 2 different stems: first → A/A', second → B/B'
        (a_path, a_prime_path, a_edit1, a_edit2), (b_path,
                                                   b_prime_path, b_edit1, b_edit2) = random.sample(stems, 2)
        pair_a = {"edit1": a_edit1, "edit2": a_edit2}
        pair_b = {"edit1": b_edit1, "edit2": b_edit2}
        if self.mode == "grid":
            combined_prompt = (
                "This is a 2x2 grid image. "
                "Top-left is A, top-right is A', bottom-left is B, bottom-right is B'. "
                f"Edit 1 (A to A'): {pair_a['edit1']}. {pair_a['edit2']}. "
                f"Edit 1 (B to B'): {pair_b['edit1']}. {pair_b['edit2']}. "
                "Treat the grid as two edit pairs: A to A' and B to B'."
            )
        else:
            combined_prompt = (
                "Image 1 is the original and image 2 is the edited version. "
                f"Edit 1: {pair_a['edit1']}. Edit 2: {pair_a['edit2']}. "
                "Apply the same edits to image 3 to produce the output."
            )
        a = Image.open(a_path).convert('RGB')
        a_prime = Image.open(a_prime_path).convert('RGB')
        b = Image.open(b_path).convert('RGB')
        b_prime = Image.open(b_prime_path).convert('RGB')
        a, a_prime, b, b_prime = harmonize_images(a, a_prime, b, b_prime)
        return {
            'A': a,
            'A_prime': a_prime,
            'B': b,
            'B_prime': b_prime,
            'edit_name': edit_name,
            'prompt': combined_prompt,
        }


def create_grid_condition_ids(packed_h, packed_w):
    """Create condition IDs for a 2x2 grid with per-quadrant T values.

    T=10 (top-left, A), T=20 (top-right, A'), T=30 (bottom-left, B),
    T=0 (bottom-right, B' generation target). H,W are global grid coordinates.
    Returns (1, packed_h*packed_w, 4).
    """
    half_h, half_w = packed_h // 2, packed_w // 2
    rows = torch.arange(packed_h).float()
    cols = torch.arange(packed_w).float()
    grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing='ij')
    t_map = torch.zeros(packed_h, packed_w)
    t_map[:half_h, :half_w] = 10   # A
    t_map[:half_h, half_w:] = 20   # A'
    t_map[half_h:, :half_w] = 30   # B
    t_map[half_h:, half_w:] = 0    # B' (generation target)
    l_map = torch.zeros(packed_h, packed_w)
    ids = torch.stack([t_map, grid_rows, grid_cols, l_map], dim=-1)
    return ids.reshape(1, packed_h * packed_w, 4)


def create_b_prime_mask(packed_h=64, packed_w=64):
    """Boolean mask for B' (bottom-right quadrant) in packed latent token space.

    For 1024x1024 grid → VAE 128x128 → packed 64x64.
    Each quadrant is 32x32 packed tokens. B' = bottom-right 32x32.
    """
    mask = torch.zeros(packed_h, packed_w, dtype=torch.bool)
    mask[packed_h // 2:, packed_w // 2:] = True
    return mask.reshape(-1)


def collate_fn(batch):
    return {
        'A': [item['A'] for item in batch],
        'A_prime': [item['A_prime'] for item in batch],
        'B': [item['B'] for item in batch],
        'B_prime': [item['B_prime'] for item in batch],
        'prompts': [item['prompt'] for item in batch],
    }


def resolve_hf_snapshot_path(repo_id: str, cache_dir: str | None = None) -> str:
    """Resolve a HuggingFace repo ID to its local snapshot directory.

    Converts e.g. 'black-forest-labs/FLUX.2-klein-base-9B' to
    '{cache_dir}/models--black-forest-labs--FLUX.2-klein-base-9B/snapshots/{hash}'.
    If repo_id is already a local directory, returns it unchanged.
    """
    if os.path.isdir(repo_id):
        return repo_id

    if cache_dir is None:
        cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.path.expanduser("~"),
                         ".cache", "huggingface", "hub"),
        )

    folder_name = "models--" + repo_id.replace("/", "--")
    model_dir = os.path.join(cache_dir, folder_name)
    refs_file = os.path.join(model_dir, "refs", "main")

    if not os.path.isfile(refs_file):
        raise FileNotFoundError(
            f"Cannot find cached model for '{repo_id}'.\n"
            f"Expected refs file at: {refs_file}\n"
            f"Download the model on a login node first:\n"
            f"  huggingface-cli download {repo_id}"
        )

    commit_hash = open(refs_file).read().strip()
    snapshot_dir = os.path.join(model_dir, "snapshots", commit_hash)

    if not os.path.isdir(snapshot_dir):
        raise FileNotFoundError(
            f"Snapshot directory missing for '{repo_id}'.\n"
            f"Expected: {snapshot_dir}\n"
            f"Re-download on a login node: huggingface-cli download {repo_id}"
        )

    print(f"Resolved '{repo_id}' → {snapshot_dir}")
    return snapshot_dir


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
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma


@torch.no_grad()
def encode_single_image(pipe, img, cell_size=512):
    """Resize, preprocess, and VAE-encode a single PIL image. Returns (1, C, h, w) latent."""
    img = resize_to_match(img, cell_size, cell_size)
    tensor = pipe.image_processor.preprocess(img, cell_size, cell_size)
    tensor = tensor.to(device=pipe._execution_device, dtype=torch.bfloat16)
    return pipe._encode_vae_image(tensor, generator=None)


@torch.no_grad()
def prepare_analogy_latents(pipe, batch, mode="concat"):
    """Encode A, A', B, B' through VAE and prepare latents + IDs.

    Args:
        mode: "concat" (default) - separate images with T=10/20/30 IDs
              "grid" - 2x2 spatial grid with B' mask

    Returns:
        target_latents, condition_latents, target_ids, condition_ids, b_prime_mask
        b_prime_mask is None in concat mode, a boolean tensor in grid mode.
    """
    target_arr, cond_arr = [], []

    for a, a_prime, b, b_prime in zip(batch['A'], batch['A_prime'], batch['B'], batch['B_prime']):
        lat_a = encode_single_image(pipe, a)
        lat_a_prime = encode_single_image(pipe, a_prime)
        lat_b = encode_single_image(pipe, b)
        lat_b_prime = encode_single_image(pipe, b_prime)

        if mode == "grid":
            # Assemble single 2x2 grid: A, A', B are clean conditions; B' is target
            # No separate condition grid — T-coordinates distinguish roles
            top_row = torch.cat([lat_a, lat_a_prime], dim=3)
            bot_target = torch.cat([lat_b, lat_b_prime], dim=3)
            target_grid = torch.cat(
                [top_row, bot_target], dim=2)  # (1, C, 2h, 2w)

            target_arr.append(Flux2KleinPipeline._pack_latents(target_grid))
        else:
            # Concat mode: B' is target, A/A'/B are separate conditions
            target_arr.append(Flux2KleinPipeline._pack_latents(lat_b_prime))
            cond_a = Flux2KleinPipeline._pack_latents(lat_a).squeeze(0)
            cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime).squeeze(0)
            cond_b = Flux2KleinPipeline._pack_latents(lat_b).squeeze(0)
            cond_arr.append(
                torch.cat([cond_a, cond_ap, cond_b], dim=0).unsqueeze(0))

    if mode == "grid":
        packed_h, packed_w = target_grid.shape[2], target_grid.shape[3]
        # Single grid IDs: B' at T=0 (target), A/A'/B at T=10/20/30 (conditions)
        target_ids = create_grid_condition_ids(packed_h, packed_w)
        b_prime_mask = create_b_prime_mask(packed_h, packed_w)
        return (
            torch.cat(target_arr, dim=0),
            None,
            target_ids,
            None,
            b_prime_mask,
        )

    target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_b_prime)
    condition_ids = Flux2KleinPipeline._prepare_image_ids(
        [lat_a, lat_a_prime, lat_b])

    return (
        torch.cat(target_arr, dim=0),
        torch.cat(cond_arr, dim=0),
        target_ids,
        condition_ids,
        None,
    )


def add_lora(transformer, args, lora_target_modules):
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=lora_target_modules,
    )
    transformer.add_adapter(transformer_lora_config,
                            adapter_name=args.lora_name)


def get_save_model_hook(accelerator, args):
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                print("Type of model in save_model_hook:", type(model))
                if isinstance(accelerator.unwrap_model(model), Flux2Transformer2DModel):
                    Flux2KleinPipeline.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=get_peft_model_state_dict(
                            accelerator.unwrap_model(model), adapter_name=args.lora_name),
                    )
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                weights.pop()  # make sure to pop weight so that corresponding model is not saved again

    return save_model_hook


def unpack_latents(latents, packed_h, packed_w):
    """Inverse of Flux2 _pack_latents: (B, H*W, C) → (B, C, H, W) in patchified space."""
    batch_size, num_patches, channels = latents.shape
    latents = latents.permute(0, 2, 1).reshape(
        batch_size, channels, packed_h, packed_w)
    return latents


@torch.no_grad()
def log_analogy_validation(pipeline, train_dataset, tokenizer, text_encoder,
                           args, accelerator, global_step):
    """Run analogy inference with LoRA scale=1 and log a comparison strip to wandb."""
    print(f"Running analogy validation at step {global_step}...")

    pipe = pipeline
    device = accelerator.device
    dtype = torch.bfloat16
    mode = getattr(args, 'analogy_mode', 'concat')

    # Pick a fresh random sample each validation
    val_sample = random.choice(train_dataset.styles)
    edit_name, stems = val_sample
    (a_path, a_prime_path, a_edit1, a_edit2), (b_path,
                                               b_prime_path, b_edit1, b_edit2) = random.sample(stems, 2)
    pair_a = {"edit1": a_edit1, "edit2": a_edit2}
    pair_b = {"edit1": b_edit1, "edit2": b_edit2}

    if mode == "grid":
        combined_prompt = (
            "This is a 2x2 grid image. "
            "Top-left is A, top-right is A', bottom-left is B, bottom-right is B'. "
            f"Edit 1 (A to A'): {pair_a['edit1']}. {pair_a['edit2']}. "
            f"Edit 1 (B to B'): {pair_b['edit1']}. {pair_b['edit2']}. "
            "Treat the grid as two edit pairs: A to A' and B to B'."
        )
    else:
        combined_prompt = (
            "Image 1 is the original and image 2 is the edited version. "
            f"Edit 1: {pair_a['edit1']}. Edit 2: {pair_a['edit2']}. "
            "Apply the same edits to image 3 to produce the output."
        )

    val_sample = {
        'A': Image.open(a_path).convert('RGB'),
        'A_prime': Image.open(a_prime_path).convert('RGB'),
        'B': Image.open(b_path).convert('RGB'),
        'B_prime': Image.open(b_prime_path).convert('RGB'),
        'edit_name': edit_name,
        'prompt': combined_prompt,
    }

    # Encode images through VAE
    lat_a = encode_single_image(pipe, val_sample['A'])
    lat_a_prime = encode_single_image(pipe, val_sample['A_prime'])
    lat_b = encode_single_image(pipe, val_sample['B'])

    # Text embeddings
    prompt_embeds, text_ids = compute_text_embeddings(
        [val_sample['prompt']], tokenizer, text_encoder
    )
    # Klein: no guidance embeds, pass guidance=None
    guidance = None

    def decode_latents(packed, h, w):
        """Decode packed latents (1, N, C) → PIL image, given patchified spatial dims h, w."""
        unpacked = packed.permute(0, 2, 1).reshape(1, -1, h, w)
        bn_mean = pipe.vae.bn.running_mean.view(
            1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(
            1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * bn_std + bn_mean
        unpacked = Flux2KleinPipeline._unpatchify_latents(unpacked)
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    pipe.transformer.set_adapters([args.lora_name], [1])
    num_inference_steps = args.num_val_inference_steps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    if mode == "grid":
        # Single grid: A, A', B are clean conditions; B' starts as noise
        top_row = torch.cat([lat_a, lat_a_prime], dim=3)
        bot_row = torch.cat([lat_b, lat_b.clone()], dim=3)
        grid = torch.cat([top_row, bot_row], dim=2)
        packed_h, packed_w = grid.shape[2], grid.shape[3]

        gen_latents = Flux2KleinPipeline._pack_latents(grid)
        bp_mask = create_b_prime_mask(packed_h, packed_w).to(device)
        clean_non_bp = gen_latents[:, ~bp_mask].clone()
        gen_latents[:, bp_mask] = torch.randn_like(gen_latents[:, bp_mask])

        combined_ids = create_grid_condition_ids(packed_h, packed_w).to(device)

        image_seq_len = gen_latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        for t in timesteps:
            noise_pred = pipe.transformer(
                hidden_states=gen_latents,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0]
            gen_latents = pipe.scheduler.step(
                noise_pred, t, gen_latents, return_dict=False)[0]
            gen_latents[:, ~bp_mask] = clean_non_bp  # keep non-B' clean

        # Decode full grid, crop B' (bottom-right quadrant)
        decoded_grid = decode_latents(gen_latents, packed_h, packed_w)
        grid_w, grid_h = decoded_grid.size
        pred_b_prime = decoded_grid.crop(
            (grid_w // 2, grid_h // 2, grid_w, grid_h))
    else:
        # Concat mode: denoise single B' image from noise
        patch_h, patch_w = lat_a.shape[2], lat_a.shape[3]

        cond_a = Flux2KleinPipeline._pack_latents(lat_a)
        cond_ap = Flux2KleinPipeline._pack_latents(lat_a_prime)
        cond_b = Flux2KleinPipeline._pack_latents(lat_b)
        condition_latents = torch.cat([cond_a, cond_ap, cond_b], dim=1)

        gen_latents = torch.randn(lat_a.shape, device=device, dtype=dtype)
        gen_latents = Flux2KleinPipeline._pack_latents(gen_latents)

        target_ids = Flux2KleinPipeline._prepare_latent_ids(lat_a).to(device)
        condition_ids = Flux2KleinPipeline._prepare_image_ids(
            [lat_a, lat_a_prime, lat_b]).to(device)
        combined_ids = torch.cat([target_ids, condition_ids], dim=1)

        image_seq_len = gen_latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)

        for t in timesteps:
            latent_model_input = torch.cat(
                [gen_latents, condition_latents], dim=1)
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(1).to(dtype) / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0][:, :gen_latents.shape[1]]
            gen_latents = pipe.scheduler.step(
                noise_pred, t, gen_latents, return_dict=False)[0]

        pred_b_prime = decode_latents(gen_latents, patch_h, patch_w)

    # Condition images from val_sample (shown once, not repeated)
    cell_size = 512
    a_img = val_sample['A'].resize((cell_size, cell_size), Image.LANCZOS)
    a_prime_img = val_sample['A_prime'].resize(
        (cell_size, cell_size), Image.LANCZOS)
    b_img = val_sample['B'].resize((cell_size, cell_size), Image.LANCZOS)
    b_prime_gt_img = val_sample['B_prime'].resize(
        (cell_size, cell_size), Image.LANCZOS)

    # Log to wandb: individual images + full strip
    if accelerator.is_main_process:
        from PIL import ImageDraw
        vis_dir = Path("./experiments") / "validation_images"
        vis_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = [a_img, a_prime_img, b_img, b_prime_gt_img, pred_b_prime]
        labels = ["A", "A'", "B", "B' GT", "B' pred"]

        cell_w, cell_h = cell_size, cell_size
        strip_w = cell_w * len(all_imgs)
        strip_h = cell_h + 30
        strip = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
        draw = ImageDraw.Draw(strip)
        for i, (img, label) in enumerate(zip(all_imgs, labels)):
            strip.paste(img.resize((cell_w, cell_h)), (i * cell_w, 30))
            draw.text((i * cell_w + 4, 4), label, fill=(0, 0, 0))

        strip_path = vis_dir / f"step_{global_step:06d}.png"
        strip.save(strip_path)
        print(f"Saved validation strip to {strip_path}")

        # Log everything to wandb directly
        # log_dict = {
        #     "val/strip": wandb.Image(strip, caption=f"step {global_step} | {val_sample['edit_name']}"),
        #     "val/A": wandb.Image(a_img, caption="A"),
        #     "val/A_prime": wandb.Image(a_prime_img, caption="A'"),
        #     "val/B": wandb.Image(b_img, caption="B"),
        #     "val/B_prime_GT": wandb.Image(b_prime_gt_img, caption="B' GT"),
        #     "val/B_prime_pred": wandb.Image(
        #         pred_b_prime, caption=f"B' pred | {val_sample['edit_name']}"
        #     ),
        # }
        # log_dict["val/prompt"] = wandb.Html(
        #     f"<b>edit:</b> {val_sample['edit_name']}<br><b>prompt:</b> {val_sample['prompt']}"
        # )
        # log_dict["train/step"] = global_step  # keep x-axis aligned
        # wandb.log(log_dict)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args(print_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    cli = parser.parse_args()
    args = OmegaConf.load(cli.config)

    # CLI --use_MoE overrides the YAML value
    if cli.use_MoE:
        args.use_MoE = True
    elif not hasattr(args, "use_MoE"):
        args.use_MoE = False

    if print_args:
        print("="*40, "Arguments", "="*40)
        for arg in args:
            print(f"{arg}: {getattr(args, arg)}")
        print("="*91)

    return args


def main():
    args = parse_args(print_args=True)

    dataloader_config = DataLoaderConfiguration(dispatch_batches=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[DistributedDataParallelKwargs(
            find_unused_parameters=True)],
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
        wandb.define_metric("val/*", step_metric="train/step")

    set_seed(args.seed)

    tokenizer, text_encoder, vae, transformer, noise_scheduler = load_models(
        args)

    # Only train the LoRA layers
    for m in [text_encoder, vae, transformer]:
        m.requires_grad_(False)

    weight_dtype = torch.bfloat16
    for m in [text_encoder, vae, transformer]:
        m.to(accelerator.device, dtype=weight_dtype)

    transformer.enable_gradient_checkpointing()

    # Resolve MoE config 
    use_moe = bool(getattr(args, "use_MoE", False))
    moe_target_modules: list[str] = list(
        getattr(args, "moe_target_modules", MOE_LORA_TARGET_MODULES_DEFAULT)
    )
    moe_lora_rank= int(getattr(args, "moe_lora_rank", args.lora_rank))
    moe_lora_alpha= float(getattr(args, "moe_lora_alpha", moe_lora_rank))
    num_experts= int(getattr(args, "num_experts", 4))
    moe_top_k= int(getattr(args, "moe_top_k", 1))
    moe_aux_loss_weight = float(getattr(args, "moe_aux_loss_weight", 0.005))


    print(f"[MoE LoRA] Enabled — targets: {moe_target_modules}, "
            f"experts={num_experts}, rank={moe_lora_rank}, top_k={moe_top_k}")

    # ---- Base LoRA: inject BaseLoRALinear on all modules EXCEPT MoE targets ----
    base_only_modules = [
        m for m in LORA_TARGET_MODULES_ALL
        if not any(m.endswith(sfx) for sfx in moe_target_modules)
    ]
    base_replaced = inject_base_lora_modules(
        transformer, base_only_modules,
        r=args.lora_rank, alpha=args.lora_rank,
        dropout=getattr(args, "lora_dropout", 0.0),
    )
    print(f"[Base LoRA] Injected {len(base_replaced)} BaseLoRALinear modules")

    # ---- MoE LoRA: injected on the remaining target layers ----
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

    # ---- Save hook: base_lora.pt + moe_lora.pt ----
    def _save_moe_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                unwrapped = accelerator.unwrap_model(model)
                if isinstance(unwrapped, Flux2Transformer2DModel):
                    save_base_lora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "base_lora.pt"),
                    )
                    save_moe_lora_state_dict(
                        unwrapped,
                        os.path.join(output_dir, "moe_lora.pt"),
                    )
                    weights.pop()

    accelerator.register_save_state_pre_hook(_save_moe_hook)

    cast_training_params([transformer], dtype=torch.float32)

    # ---- Optimizer: BaseLoRALinear params + MoE gate/expert params ----
    base_lora_params = collect_base_lora_params(transformer)
    moe_params       = collect_moe_lora_params(transformer)
    all_trainable    = base_lora_params + moe_params
    print(f"[Optimizer] base LoRA params: {len(base_lora_params)}, MoE params: {len(moe_params)}")

    optimizer = torch.optim.AdamW(
        [{"params": all_trainable, "lr": args.learning_rate}],
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8,
    )

    # Making Dataset and Loaders 
    analogy_mode = getattr(args, 'analogy_mode', 'concat')
    args.analogy_mode = analogy_mode
    train_dataset = ImageAnalogyDataset(args.data_root, mode=analogy_mode)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, num_workers=2, shuffle=True)

    lr_scheduler = get_scheduler(
        "constant",  # TODO: make it argument
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    len_train_dataset = len(train_dataset)
    len_train_dataloader = math.ceil(len_train_dataset / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(
        len_train_dataloader / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    print("***** Running training *****")
    print(f"  Num examples = {len_train_dataset}")
    print(f"  Num batches each epoch = {len_train_dataloader}")
    print(f"  Num Epochs = {num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = 0
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        # Get latest checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            print(
                f"No checkpoint found in {args.output_dir}. Starting a new training run.")
            args.resume_from_checkpoint = None
        else:
            print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    pipeline = Flux2KleinPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=accelerator.unwrap_model(
            transformer, keep_fp32_wrapper=False),
    )

    transformer.train()

    for epoch in range(first_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                prompts = batch["prompts"]

                target_latents, condition_latents, target_ids, condition_ids, b_prime_mask = prepare_analogy_latents(
                    pipeline, batch, mode=args.analogy_mode
                )

                device = target_latents.device
                target_ids = target_ids.to(device)

                bsz = target_latents.shape[0]

                # Expand IDs to batch size
                target_ids = target_ids.expand(bsz, -1, -1)

                if condition_ids is not None:
                    # Concat mode: target (T=0) + conditions (T=10/20/30)
                    condition_ids = condition_ids.to(
                        device).expand(bsz, -1, -1)
                    combined_ids = torch.cat(
                        [target_ids, condition_ids], dim=1)
                else:
                    # Grid mode: single grid with per-quadrant T values
                    combined_ids = target_ids

                prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, tokenizer, text_encoder)

                noise = torch.randn_like(target_latents)

                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=bsz)
                indices = (u * (args.max_train_timesteps -
                           args.min_train_timesteps) + args.min_train_timesteps).long()
                indices = indices.clamp(0, len(noise_scheduler.timesteps) - 1)
                timesteps = noise_scheduler.timesteps[indices].to(
                    device=target_latents.device)

                # Add noise via flow matching (grid: only B' quadrant; concat: all tokens)
                sigmas = get_sigmas(noise_scheduler, timesteps, device=target_latents.device,
                                    n_dim=target_latents.ndim, dtype=target_latents.dtype)
                if b_prime_mask is not None:
                    b_prime_mask_dev = b_prime_mask.to(device)
                    noisy_model_input = target_latents.clone()
                    noisy_model_input[:, b_prime_mask_dev] = (
                        1.0 - sigmas) * target_latents[:, b_prime_mask_dev] + sigmas * noise[:, b_prime_mask_dev]
                else:
                    noisy_model_input = (1.0 - sigmas) * \
                        target_latents + sigmas * noise

                # Klein: no guidance embeds, always guidance=None
                guidance = None

                if condition_latents is not None:
                    # Concat mode: append condition tokens
                    latent_model_input = torch.cat(
                        [noisy_model_input, condition_latents], dim=1)
                else:
                    # Grid mode: single grid, conditions embedded spatially
                    latent_model_input = noisy_model_input

                # Get the models ouput for B'(in the grid mode, the whole grid but loss only on B' via masking)
                model_pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_ids,
                    return_dict=False,
                )[0]

                if condition_latents is not None:
                    model_pred = model_pred[:, :noisy_model_input.shape[1]]

                # Flow matching loss (grid: only B' quadrant; concat: all tokens)
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=sigmas)
                flow_target = noise - target_latents
                if b_prime_mask is not None:
                    pred_masked = model_pred[:, b_prime_mask_dev]
                    target_masked = flow_target[:, b_prime_mask_dev]
                else:
                    pred_masked = model_pred
                    target_masked = flow_target
                loss = torch.mean(
                    (weighting.float() * (pred_masked.float() -
                     target_masked.float()) ** 2).reshape(bsz, -1),
                    1,
                )
                loss = loss.mean()

                # MoE load-balancing aux loss
                if use_moe:
                    _unwrapped = accelerator.unwrap_model(transformer)
                    moe_aux = collect_moe_aux_losses(_unwrapped)
                    if moe_aux is not None:
                        loss = loss + moe_aux_loss_weight * moe_aux

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss":  loss.detach().item(),
                        "train/lr":    lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/step":  global_step,
                    })

                    if global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0 or global_step == args.max_train_steps:
                        pipeline.transformer = accelerator.unwrap_model(
                            transformer, keep_fp32_wrapper=False)
                        log_analogy_validation(
                            pipeline, train_dataset,
                            tokenizer, text_encoder,
                            args, accelerator, global_step,
                        )
                        transformer.train()

            logs = {"loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    main()
