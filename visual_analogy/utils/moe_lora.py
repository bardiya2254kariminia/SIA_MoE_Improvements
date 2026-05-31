"""
utils/moe_lora.py

Utilities for injecting, saving, loading, and collecting parameters from
TokenWiseGatedMoELoraLinear modules inside a FLUX.2-Klein transformer.

Public API
----------
inject_moe_lora_modules        – replace target nn.Linear modules with MoE LoRA
save_moe_lora_state_dict       – checkpoint gate + expert weights
load_moe_lora_state_dict       – restore from that checkpoint
collect_moe_lora_params        – list of all trainable MoE parameters
collect_moe_aux_losses         – average aux loss across all MoE modules
set_moe_lora_requires_grad     – toggle requires_grad on MoE parameters
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from visual_analogy.models.moe_lora import TokenWiseGatedMoELoraLinear
from visual_analogy.models.selective_lora import BaseLoRALinear
from visual_analogy.utils.selective_lora import _get_parent


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_moe_lora_modules(
    model: nn.Module,
    target_linear_name_suffixes: list[str],
    num_experts: int,
    r: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    top_k: int = 1,
) -> list[str]:
    """Replace target nn.Linear (or BaseLoRALinear) modules with MoE LoRA.

    Replacement is done deepest-first to avoid replacing a parent before
    its children have been scanned.

    Args:
        model:                        the transformer to modify in-place
        target_linear_name_suffixes:  list of module name suffixes to match
        num_experts:                  number of expert LoRA pairs
        r:                            LoRA rank
        lora_alpha:                   LoRA alpha (scaling = lora_alpha / r)
        lora_dropout:                 dropout applied to x before each expert
        top_k:                        experts selected per token

    Returns:
        List of replaced module names.
    """
    candidates: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not any(name.endswith(sfx) for sfx in target_linear_name_suffixes):
            continue
        if isinstance(module, nn.Linear):
            candidates.append((name, module))
        elif isinstance(module, BaseLoRALinear):
            # If a BaseLoRALinear is already there, wrap its inner base linear
            candidates.append((name, module.base))

    # Sort deepest first to prevent double-replacement
    candidates.sort(key=lambda x: x[0].count("."), reverse=True)

    replaced: list[str] = []
    for name, linear in candidates:
        parent, attr = _get_parent(model, name)
        moe_module = TokenWiseGatedMoELoraLinear(
            base_linear=linear,
            num_experts=num_experts,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            top_k=top_k,
            full_name=name,
        )
        setattr(parent, attr, moe_module)
        replaced.append(name)

    return replaced


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_moe_lora_state_dict(model: nn.Module, output_file_path: str) -> OrderedDict:
    """Save gate weights and all expert lora_A / lora_B weights.

    Key format:
        "{module_name}.gate.weight"
        "{module_name}.lora_A.expert_{i}.weight"
        "{module_name}.lora_B.expert_{i}.weight"
    """
    ckpt = OrderedDict()
    for name, module in model.named_modules():
        if not isinstance(module, TokenWiseGatedMoELoraLinear):
            continue
        ckpt[f"{name}.gate.weight"] = module.gate.weight.cpu()
        for expert_key in module.lora_A:
            ckpt[f"{name}.lora_A.{expert_key}.weight"] = (
                module.lora_A[expert_key].weight.cpu()
            )
            ckpt[f"{name}.lora_B.{expert_key}.weight"] = (
                module.lora_B[expert_key].weight.cpu()
            )
    torch.save(ckpt, output_file_path)
    return ckpt


def load_moe_lora_state_dict(
    model: nn.Module,
    ckpt_path: str,
    device: str = "cpu",
) -> None:
    """Restore MoE LoRA weights saved by :func:`save_moe_lora_state_dict`."""
    raw_sd: dict = torch.load(ckpt_path, map_location=device, weights_only=True)

    moe_map: dict[str, TokenWiseGatedMoELoraLinear] = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, TokenWiseGatedMoELoraLinear)
    }

    loaded = 0
    skipped = 0

    for key, weight in raw_sd.items():
        matched = False
        for mod_name, mod in moe_map.items():
            if not key.startswith(mod_name + "."):
                continue
            suffix = key[len(mod_name) + 1:]  # e.g. "gate.weight"

            if suffix == "gate.weight":
                with torch.no_grad():
                    mod.gate.weight.copy_(weight.to(mod.gate.weight.device))
                loaded += 1
                matched = True

            elif suffix.startswith("lora_A."):
                # "lora_A.expert_0.weight"
                expert_key = suffix[len("lora_A."):-len(".weight")]
                if expert_key in mod.lora_A:
                    with torch.no_grad():
                        mod.lora_A[expert_key].weight.copy_(
                            weight.to(mod.lora_A[expert_key].weight.device)
                        )
                    loaded += 1
                    matched = True

            elif suffix.startswith("lora_B."):
                expert_key = suffix[len("lora_B."):-len(".weight")]
                if expert_key in mod.lora_B:
                    with torch.no_grad():
                        mod.lora_B[expert_key].weight.copy_(
                            weight.to(mod.lora_B[expert_key].weight.device)
                        )
                    loaded += 1
                    matched = True

            if matched:
                break

        if not matched:
            skipped += 1

    print(f"[MoE LoRA Load] loaded={loaded}, skipped={skipped} from {ckpt_path}")


# ---------------------------------------------------------------------------
# Parameter collection
# ---------------------------------------------------------------------------

def collect_moe_lora_params(model: nn.Module) -> list[torch.nn.Parameter]:
    """Return a deduplicated list of all trainable MoE LoRA parameters.

    Includes: gate.weight, all lora_A[expert_i].weight, all lora_B[expert_i].weight.
    """
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in model.modules():
        if not isinstance(module, TokenWiseGatedMoELoraLinear):
            continue
        for p in [module.gate.weight]:
            if id(p) not in seen:
                params.append(p)
                seen.add(id(p))
        for expert_key in module.lora_A:
            for p in [
                module.lora_A[expert_key].weight,
                module.lora_B[expert_key].weight,
            ]:
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params


def collect_moe_aux_losses(model: nn.Module) -> torch.Tensor | None:
    """Average the Switch-Transformer load-balancing losses across MoE modules.

    Returns None when no MoE modules are present or when all aux losses are
    scalar 0.0 (eval mode).
    """
    losses: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, TokenWiseGatedMoELoraLinear):
            loss = module.current_aux_loss
            if torch.is_tensor(loss):
                losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def set_moe_lora_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Toggle requires_grad on every MoE LoRA parameter."""
    for p in collect_moe_lora_params(model):
        p.requires_grad_(requires_grad)
