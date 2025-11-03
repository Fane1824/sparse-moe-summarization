from typing import List, Tuple

import torch
import torch.nn as nn

from .experts import FeedForwardExpert, LoRAFeedForwardExpert
from .gating import TopKRouter, HashRouter, load_balancer_loss


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int,
        k: int,
        router: str = "topk",
        dropout: float = 0.1,
        use_lora_experts: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router_type = router

        if router == "topk":
            self.router = TopKRouter(d_model, num_experts, k)
        elif router == "hash":
            self.router = HashRouter(d_model, num_experts, k)
        else:
            raise ValueError(f"Unknown router: {router}")

        ExpertCls = LoRAFeedForwardExpert if use_lora_experts else FeedForwardExpert
        self.experts = nn.ModuleList(
            [
                ExpertCls(
                    d_model=d_model,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    r=lora_r,
                    alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                if use_lora_experts
                else ExpertCls(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)
                for _ in range(num_experts)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, seq, d_model)
        b, s, d = x.shape
        flat = x.reshape(b * s, d)
        expert_idx, gates = self.router(flat)  # (tokens, k), (tokens, k)

        # Accumulate outputs
        out = torch.zeros_like(flat)
        # Track usage for visualization - reshape to (batch, seq, k)
        assignment = expert_idx.detach().reshape(b, s, -1)

        # Process each of the k expert choices
        for choice in range(expert_idx.shape[1]):
            idx = expert_idx[:, choice]  # (tokens)
            # Process per expert
            for e_id in range(self.num_experts):
                sel = (idx == e_id)
                if sel.any():
                    tokens = flat[sel]
                    y = self.experts[e_id](tokens)
                    g = gates[sel, choice].unsqueeze(-1)
                    out[sel] = out[sel] + y * g

        out = out.reshape(b, s, d)
        return self.dropout(out), assignment, gates

    @staticmethod
    def aux_load_balancer_loss(assignment: torch.Tensor, num_experts: int) -> torch.Tensor:
        return load_balancer_loss(assignment, num_experts)

