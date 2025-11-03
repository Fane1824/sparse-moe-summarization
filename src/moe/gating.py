from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (tokens, d_model)
        logits = self.w_gate(x)  # (tokens, E)
        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)
        gates = F.softmax(topk_vals, dim=-1)  # normalized over top-k
        return topk_idx, gates


class HashRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int, seed: int = 42):
        super().__init__()
        assert k == 1, "Hash router supports k=1 in this simple implementation"
        self.num_experts = num_experts
        self.k = k
        self.register_buffer("proj", torch.randn(d_model))
        torch.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simple LSH-like projection to decide expert
        proj = (x @ self.proj)  # (tokens)
        buckets = torch.remainder((proj * 9973).long().abs(), self.num_experts)  # (tokens)
        idx = buckets.unsqueeze(-1)
        gates = torch.ones_like(idx, dtype=x.dtype)
        return idx, gates


def load_balancer_loss(assignment: torch.Tensor, num_experts: int) -> torch.Tensor:
    # assignment: (tokens, k) expert indices
    # encourage uniform expert usage
    tokens = assignment.numel()
    hist = torch.bincount(assignment.view(-1), minlength=num_experts).float() / tokens
    target = torch.full_like(hist, 1.0 / num_experts)
    return F.mse_loss(hist, target)

