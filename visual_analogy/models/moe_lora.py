import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenWiseGatedMoELoraLinear(nn.Module):
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
        self.base_layer = base_linear
        self.num_experts = num_experts
        self.r = r
        self.top_k = top_k
        self.full_name = full_name

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.current_aux_loss = 0.0

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
        """
            x: (Batch, Tokens, Dim)
        """
        orig_dtype = x.dtype

        result = self.base_layer(x)  
        lora_dtype = self.gate.weight.dtype
        if x.dtype != lora_dtype:
            x = x.to(lora_dtype)

        # Routing 
        route_logits = self.gate(x)                          # [B, N, num_experts]
        all_probs = F.softmax(route_logits, dim=-1,
                              dtype=torch.float32)           # keep fp32 for routing

        top_k_probs, top_k_indices = torch.topk(all_probs, k=self.top_k, dim=-1)

        if self.top_k == 1:
            top_k_probs = 1.0 + (top_k_probs - top_k_probs.detach())
        else:
            # Re-normalise top-k probabilities so they sum to 1
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        top_k_probs = top_k_probs.to(orig_dtype)

        # Load-balancing aux loss (Enforce to Switch-Transformer) 
        if self.training:
            me = torch.mean(all_probs, dim=(0, 1))          # mean probability ussage for each expert [num_experts]
            expert_mask = torch.zeros_like(route_logits, dtype=torch.float32)
            expert_mask.scatter_(-1, top_k_indices, 1.0)
            ce = torch.mean(expert_mask, dim=(0, 1))        # mean ussage fraction for each expert [num_experts]
            self.current_aux_loss = self.num_experts * torch.sum(me * ce)
        else:
            self.current_aux_loss = 0.0

        # Build sparse routing weight
        route_weight = torch.zeros_like(route_logits, dtype=orig_dtype)
        route_weight.scatter_(-1, top_k_indices, top_k_probs)  # [B, N, num_experts]

        # Calculate MoE output
        x_dn = self.dropout(x)
        lora_delta = torch.zeros_like(result, dtype=orig_dtype)

        for i in range(self.num_experts):
            mask = route_weight[:, :, i].unsqueeze(-1)      # [B, N, 1]
            expert_key = f"expert_{i}"
            expert_out = self.lora_B[expert_key](
                self.lora_A[expert_key](x_dn)
            ).to(orig_dtype)
            lora_delta = lora_delta + expert_out * mask

        return (result + lora_delta * self.scaling).to(orig_dtype)

    def __repr__(self):
        return (
            f"TokenWiseGatedMoELoraLinear("
            f"in={self.in_features}, out={self.out_features}, "
            f"experts={self.num_experts}, r={self.r}, "
            f"top_k={self.top_k}, name={self.full_name!r})"
        )
