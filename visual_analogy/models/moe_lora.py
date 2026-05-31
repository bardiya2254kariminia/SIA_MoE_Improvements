"""
moe_lora.py

Token-wise Gated Mixture-of-Experts LoRA for FLUX.2-Klein.

Inspired by VIRAL's TokenWiseGatedMoELoraLinear but ported cleanly into
SIA's architecture with the expert-accumulation bug fixed: each expert's
output is now correctly weighted and summed into lora_delta before the
final base_layer + lora_delta * scaling return.

Key differences vs. VIRAL moe_lora.py:
  - Bug fixed: `lora_delta += expert_out * mask` (was missing in VIRAL)
  - Consistent dtype casting through the entire forward pass
  - top_k==1 uses straight-through estimator; top_k>1 uses normalised probs
  - Load-balancing aux loss (Switch-Transformer style) stored per forward in
    `self.current_aux_loss` (tensor during training, scalar 0.0 at eval)
  - `in_features` / `out_features` exposed for compatibility with SIA utils
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenWiseGatedMoELoraLinear(nn.Module):
    """Token-wise gated MoE LoRA layer (drop-in replacement for nn.Linear).

    For each token independently, a learned gate routes the token to the
    top-k experts. Each expert is a rank-r LoRA pair (lora_A, lora_B).
    The final output is:

        y = base_layer(x) + sum_i(route_weight_i * lora_B_i(lora_A_i(x))) * scaling

    The base_layer is frozen; only the gate, lora_A, and lora_B parameters
    are trainable.

    Args:
        base_linear: the nn.Linear to wrap (frozen in-place)
        num_experts:  number of LoRA expert pairs
        r:            LoRA rank (same for every expert)
        lora_alpha:   LoRA scaling alpha; effective scaling = lora_alpha / r
        lora_dropout: dropout rate applied to x before each expert
        top_k:        number of experts selected per token (1 or 2)
        full_name:    dotted module name, stored for debugging / checkpointing
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 4,
        r: int = 16,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        top_k: int = 1,
        full_name: str = "",
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear), (
            f"base_linear must be nn.Linear, got {type(base_linear)}"
        )

        self.base_layer = base_linear
        self.num_experts = num_experts
        self.r = r
        self.top_k = top_k
        self.full_name = full_name

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.alpha = lora_alpha
        self.scaling = lora_alpha / r

        # Accumulates the Switch-Transformer load-balancing loss each forward.
        # Collected externally via collect_moe_aux_losses(); reset each forward.
        self.current_aux_loss: torch.Tensor | float = 0.0

        # Freeze the pre-trained base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.ModuleDict({
            f"expert_{i}": nn.Linear(self.in_features, r, bias=False)
            for i in range(num_experts)
        })
        self.lora_B = nn.ModuleDict({
            f"expert_{i}": nn.Linear(r, self.out_features, bias=False)
            for i in range(num_experts)
        })
        self.gate = nn.Linear(self.in_features, num_experts, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Initialisation
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[f"expert_{i}"].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[f"expert_{i}"].weight)

    # ------------------------------------------------------------------
    # Inference helpers (used by infer_single.py ablations)
    # ------------------------------------------------------------------

    def set_scaling(self, scaling):
        self.scaling = scaling

    def reset_scaling(self):
        self.scaling = self.alpha / self.r

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x shape: [Batch, Tokens, Dim]
        """
        orig_dtype = x.dtype

        result = self.base_layer(x)  # [B, N, out_features]

        # Cast x to gate/lora weight dtype for mixed-precision safety
        lora_dtype = self.gate.weight.dtype
        if x.dtype != lora_dtype:
            x = x.to(lora_dtype)

        # ---- Routing ----
        route_logits = self.gate(x)                          # [B, N, num_experts]
        all_probs = F.softmax(route_logits, dim=-1,
                              dtype=torch.float32)           # keep fp32 for routing

        top_k_probs, top_k_indices = torch.topk(all_probs, k=self.top_k, dim=-1)

        if self.top_k == 1:
            # Straight-through estimator: gradient flows through the winner
            top_k_probs = 1.0 + (top_k_probs - top_k_probs.detach())
        else:
            # Re-normalise top-k probabilities so they sum to 1
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        top_k_probs = top_k_probs.to(orig_dtype)

        # ---- Load-balancing aux loss (Switch-Transformer) ----
        if self.training:
            me = torch.mean(all_probs, dim=(0, 1))          # [num_experts] mean probability
            expert_mask = torch.zeros_like(route_logits, dtype=torch.float32)
            expert_mask.scatter_(-1, top_k_indices, 1.0)
            ce = torch.mean(expert_mask, dim=(0, 1))        # [num_experts] mean dispatch fraction
            self.current_aux_loss = self.num_experts * torch.sum(me * ce)
        else:
            self.current_aux_loss = 0.0

        # ---- Build sparse routing weight ----
        route_weight = torch.zeros_like(route_logits, dtype=orig_dtype)
        route_weight.scatter_(-1, top_k_indices, top_k_probs)  # [B, N, num_experts]

        # ---- Expert computation ----
        x_dn = self.dropout(x)
        lora_delta = torch.zeros_like(result, dtype=orig_dtype)

        for i in range(self.num_experts):
            mask = route_weight[:, :, i].unsqueeze(-1)      # [B, N, 1]
            expert_key = f"expert_{i}"
            expert_out = self.lora_B[expert_key](
                self.lora_A[expert_key](x_dn)
            ).to(orig_dtype)
            lora_delta = lora_delta + expert_out * mask     # weighted accumulation

        return (result + lora_delta * self.scaling).to(orig_dtype)

    def __repr__(self):
        return (
            f"TokenWiseGatedMoELoraLinear("
            f"in={self.in_features}, out={self.out_features}, "
            f"experts={self.num_experts}, r={self.r}, "
            f"top_k={self.top_k}, name={self.full_name!r})"
        )
