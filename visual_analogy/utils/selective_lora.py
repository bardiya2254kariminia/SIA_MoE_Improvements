from contextlib import contextmanager
from collections import OrderedDict
import re
import torch
import torch.nn as nn
from visual_analogy.models.selective_lora import SelectiveLoRALinear, BaseLoRALinear
from visual_analogy.models.moe_lora import TokenWiseGatedMoELoraLinear

def _apply_klein_chat_template(prompt: str, tokenizer) -> str:
    """Wrap ``prompt`` in the same Qwen chat template that
    ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` applies before tokenizing.

    This is the **only** sequence the diffusion transformer ever sees, so
    every token-position computation (STLoRA mask, steering vector,
    embedding pooling) MUST be done against this string.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def klein_find_substring_token_indices(
    prompt: str,
    substr: str,
    tokenizer,
    max_length: int = 512,
    use_chat_template: bool = True,
):
    """Return Qwen token indices of ``substr`` inside ``prompt``.

    With ``use_chat_template=True`` (the default and the **only correct**
    option for Flux2-Klein inference / training), the substring is aligned
    against the chat-templated text — which is exactly what
    ``Flux2KleinPipeline._get_qwen3_prompt_embeds`` encodes. This means the
    returned indices are directly usable to index ``prompt_embeds`` (for
    steering vectors) or to build a token mask consumed by
    ``SelectiveLoRALinear`` (which sees ``encoder_hidden_states`` produced
    from the same templated sequence).

    With ``use_chat_template=False`` the substring is aligned against the
    raw prompt instead. This is **only** kept for forensic comparisons
    against pre-fix runs that mistakenly used raw-prompt offsets — those
    runs trained STLoRA with an effectively empty mask (see commit history /
    README "Token-position alignment" section). New code must keep the
    default.
    """
    text = _apply_klein_chat_template(prompt, tokenizer) if use_chat_template else prompt

    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"][0].tolist()  # list of (char_start, char_end)

    # Find the character span of the substring in the (possibly templated) text
    char_start = text.find(substr)
    if char_start == -1:
        raise AssertionError(
            f"substr not found in {'templated ' if use_chat_template else ''}"
            f"prompt text: '{substr[:80]}'"
        )
    char_end = char_start + len(substr)

    # Collect token indices whose character span overlaps with the substring.
    # Padding tokens and special tokens have offset (0, 0); skip them after
    # position 0 so we don't accidentally pick up the leading <|im_start|>.
    token_indices = []
    for tok_idx, (cs, ce) in enumerate(offsets):
        if cs == ce == 0 and tok_idx > 0:
            continue
        if ce > char_start and cs < char_end:
            token_indices.append(tok_idx)

    if not token_indices:
        raise AssertionError(
            f"No tokens overlap with substr chars [{char_start}:{char_end}]: "
            f"'{substr[:80]}'"
        )

    return token_indices


def klein_templated_prompt_token_positions(
    prompt: str,
    tokenizer,
    max_length: int = 512,
):
    """Return the list of non-padding token indices in the chat-templated
    prompt — i.e. every "real" text-token position the diffusion encoder
    actually sees (preamble, edits, end-of-turn marker, etc.).

    Used to support the ``scope='prompt'`` steering mode in inference and
    for sanity checks / regression tests.
    """
    text = _apply_klein_chat_template(prompt, tokenizer)
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    attn = enc["attention_mask"][0].tolist()
    return [i for i, m in enumerate(attn) if m]


@contextmanager
def stlora_token_mask_ctx(model, mask: torch.Tensor, disable_mask_after=True):  # mask: (B, S) bool
    """
    Context manager to set token mask for all SelectiveLoRALinear modules in the model.

    Args:
        model: The model containing SelectiveLoRALinear modules.
        mask: A boolean tensor of shape (B, S) indicating which tokens to apply LoRA to.
        disable_mask_after: If True, the token mask will be cleared after exiting the context.
    """
    loras = [m for m in model.modules() if isinstance(m, SelectiveLoRALinear)]
    for m in loras:
        m.set_token_mask(mask)
    try:
        yield
    finally:
        if disable_mask_after:
            for m in loras:
                m.set_token_mask(None)


def _get_parent(model, dotted_name):
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_selective_lora_modules(model, target_linear_name_suffixes, r, alpha, dropout):
    replaced = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == "SelectiveLoRALinear":  # Already loaded (Useful for debugging)
            module = module.base

        is_target = isinstance(module, (nn.Linear, BaseLoRALinear, TokenWiseGatedMoELoraLinear))
        if is_target and any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            parent, attr = _get_parent(model, name)
            wrapped = SelectiveLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            replaced.append(name)

    return replaced


def delete_selective_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, SelectiveLoRALinear):
            parent, attr = _get_parent(model, name)
            setattr(parent, attr, module.base)


def save_selective_lora_state_dict(unwraped_model, output_file_path):
    ckpt = OrderedDict()
    for name, module in unwraped_model.named_modules():
        if isinstance(module, SelectiveLoRALinear):
            ckpt[f"{name}.lora_A.weight"] = module.lora_A.weight.cpu()
            ckpt[f"{name}.lora_B.weight"] = module.lora_B.weight.cpu()

    torch.save(ckpt, output_file_path)


# ---------------------------------------------------------------------------
# Base LoRA injection, loading, saving, and param management
# ---------------------------------------------------------------------------

def inject_base_lora_modules(model, target_linear_name_suffixes, r, alpha, dropout=0.0):
    """Wrap target nn.Linear modules with BaseLoRALinear (always-active LoRA)."""
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            parent, attr = _get_parent(model, name)
            wrapped = BaseLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr, wrapped)
            replaced.append(name)
    return replaced


def load_base_lora_from_peft_checkpoint(model, ckpt_path, device="cpu"):
    """Load a PEFT-format LoRA checkpoint into BaseLoRALinear modules.

    Handles key format: "transformer.{module_path}.lora_A.weight"
    or "{module_path}.lora_A.weight" (with or without adapter name in key).
    """
    from pathlib import Path
    from safetensors.torch import load_file

    ckpt_dir = Path(ckpt_path)
    sf_file = ckpt_dir / "pytorch_lora_weights.safetensors"
    pt_file = ckpt_dir / "pytorch_lora_weights.bin"

    if sf_file.exists():
        raw_sd = load_file(sf_file, device=str(device))
    elif pt_file.exists():
        raw_sd = torch.load(pt_file, map_location=device, weights_only=True)
    else:
        raise FileNotFoundError(
            f"No LoRA weights found in {ckpt_path}. "
            "Expected 'pytorch_lora_weights.safetensors' or '.bin'."
        )

    # Build a lookup: module_path → BaseLoRALinear
    base_lora_map = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseLoRALinear):
            base_lora_map[name] = module

    matched, skipped, mismatched = [], [], []

    for key, weight in raw_sd.items():
        norm = key
        # Strip common prefixes
        for pfx in ("transformer.", "base_model.model."):
            if norm.startswith(pfx):
                norm = norm[len(pfx):]

        # Strip adapter name: "lora_A.{adapter_name}.weight" → "lora_A.weight"
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

        if mod_name not in base_lora_map:
            skipped.append(key)
            continue

        target = getattr(base_lora_map[mod_name], ab)
        if target.weight.shape != weight.shape:
            mismatched.append(f"{key}: ckpt={weight.shape} model={target.weight.shape}")
            continue

        with torch.no_grad():
            target.weight.copy_(weight.to(target.weight.device))
        matched.append(key)

    print(
        f"[Base LoRA Load] matched={len(matched)}, "
        f"skipped={len(skipped)}, shape_mismatch={len(mismatched)}"
    )
    if mismatched:
        for m in mismatched:
            print(f"  [Shape mismatch] {m}")

    # Sanity check
    for name, mod in base_lora_map.items():
        if mod.lora_B.weight.data.abs().sum().item() > 0:
            print(f"[Base LoRA Load] Weights OK (sample: {name}, "
                  f"B norm={mod.lora_B.weight.data.norm().item():.4f})")
            break
    else:
        print("[Base LoRA Load] WARNING — all lora_B weights are zero!")


def save_base_lora_state_dict(model, output_file_path):
    """Save BaseLoRALinear weights. Strips '.base' suffix for compatibility."""
    ckpt = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, BaseLoRALinear):
            # If inside a SelectiveLoRALinear, name ends with ".base" — strip it
            save_name = name[:-5] if name.endswith(".base") else name
            ckpt[f"{save_name}.lora_A.weight"] = module.lora_A.weight.cpu()
            ckpt[f"{save_name}.lora_B.weight"] = module.lora_B.weight.cpu()
    torch.save(ckpt, output_file_path)


def load_base_lora_state_dict(model, ckpt_path, device="cpu"):
    """Load base LoRA weights saved by save_base_lora_state_dict."""
    raw_sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    base_lora_map = {}
    for name, module in model.named_modules():
        if isinstance(module, BaseLoRALinear):
            save_name = name[:-5] if name.endswith(".base") else name
            base_lora_map[save_name] = module

    loaded = 0
    for key, weight in raw_sd.items():
        if ".lora_A.weight" in key:
            mod_name = key.replace(".lora_A.weight", "")
            ab = "lora_A"
        elif ".lora_B.weight" in key:
            mod_name = key.replace(".lora_B.weight", "")
            ab = "lora_B"
        else:
            continue

        if mod_name in base_lora_map:
            target = getattr(base_lora_map[mod_name], ab)
            with torch.no_grad():
                target.weight.copy_(weight.to(target.weight.device))
            loaded += 1

    print(f"[Base LoRA Resume] Loaded {loaded} params from {ckpt_path}")


def collect_stlora_params(model):
    """Collect SelectiveLoRALinear's own lora_A/B parameters (not base's)."""
    params = []
    for module in model.modules():
        if isinstance(module, SelectiveLoRALinear):
            params.append(module.lora_A.weight)
            params.append(module.lora_B.weight)
    return params


def collect_base_lora_params(model):
    """Collect BaseLoRALinear's lora_A/B parameters."""
    params = []
    seen = set()
    for module in model.modules():
        if isinstance(module, BaseLoRALinear):
            for p in [module.lora_A.weight, module.lora_B.weight]:
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params


def set_stlora_requires_grad(model, requires_grad: bool):
    """Toggle requires_grad on all STLoRA parameters."""
    for p in collect_stlora_params(model):
        p.requires_grad_(requires_grad)


def set_base_lora_requires_grad(model, requires_grad: bool):
    """Toggle requires_grad on all base LoRA parameters."""
    for p in collect_base_lora_params(model):
        p.requires_grad_(requires_grad)
