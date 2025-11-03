from typing import Optional

import torch
import torch.nn as nn


class FeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward)
        self.w2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x))))


class LoRALinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float):
        super().__init__(in_features, out_features, bias=True)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        if r > 0:
            self.A = nn.Parameter(torch.zeros(out_features, r))
            self.B = nn.Parameter(torch.zeros(r, in_features))
            nn.init.kaiming_uniform_(self.A, a=5**0.5)
            nn.init.kaiming_uniform_(self.B, a=5**0.5)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        if self.r > 0:
            lora = (x @ self.B.t()) @ self.A.t()
            lora = self.dropout(lora) * self.scaling
            return base + lora
        return base


class LoRAFeedForwardExpert(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, r: int, alpha: int, lora_dropout: float):
        super().__init__()
        self.w1 = LoRALinear(d_model, dim_feedforward, r=r, alpha=alpha, dropout=lora_dropout)
        self.w2 = LoRALinear(dim_feedforward, d_model, r=r, alpha=alpha, dropout=lora_dropout)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x))))

