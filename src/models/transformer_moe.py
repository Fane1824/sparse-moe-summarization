from typing import Tuple, Optional

import torch
import torch.nn as nn

from src.moe.layer import SparseMoE
from src.models.attention import GroupedQueryAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        moe: SparseMoE | None,
        use_gqa: bool = False,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.use_gqa = use_gqa
        
        if use_gqa:
            # Use custom GQA implementation
            self.self_attn = GroupedQueryAttention(d_model, nhead, num_kv_heads=num_kv_heads, dropout=dropout)
            self.cross_attn = GroupedQueryAttention(d_model, nhead, num_kv_heads=num_kv_heads, dropout=dropout)
        else:
            # Use PyTorch's standard MultiheadAttention
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = moe if moe is not None else nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.Tensor, 
        enc_out: torch.Tensor, 
        attn_mask=None,
        self_attn_past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_attn_past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        logs = {}
        present_key_values = {}
        
        if self.use_gqa:
            # Self-attention with KV caching
            h, _, self_past_kv = self.self_attn(
                x, x, x, 
                attn_mask=attn_mask,
                past_key_value=self_attn_past_key_value,
                use_cache=use_cache
            )
            x = self.norm1(x + self.drop(h))
            
            # Cross-attention: cache encoder K/V once, reuse forever
            # For cross-attention, K and V come from enc_out which doesn't change
            if cross_attn_past_key_value is not None:
                # Reuse cached encoder K/V
                h, _, cross_past_kv = self.cross_attn(
                    x, enc_out, enc_out,
                    past_key_value=cross_attn_past_key_value,
                    use_cache=use_cache
                )
            else:
                # First time: compute and cache encoder K/V
                h, _, cross_past_kv = self.cross_attn(
                    x, enc_out, enc_out,
                    use_cache=use_cache
                )
            x = self.norm2(x + self.drop(h))
            
            if use_cache:
                present_key_values['self_attn'] = self_past_kv
                present_key_values['cross_attn'] = cross_past_kv
        else:
            # PyTorch MultiheadAttention doesn't support KV caching
            # For backward compatibility, just run normal attention
            h, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
            x = self.norm1(x + self.drop(h))
            h, _ = self.cross_attn(x, enc_out, enc_out)
            x = self.norm2(x + self.drop(h))
        
        if isinstance(self.mlp, SparseMoE):
            h, assign, gates = self.mlp(x)
            logs["assignment"] = assign
            logs["gates"] = gates
        else:
            h = self.mlp(x)
        x = self.norm3(x + self.drop(h))
        
        if use_cache:
            logs["past_key_values"] = present_key_values
        
        return x, logs


class SimpleTransformerWithMoE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        moe_cfg: dict,
        use_gqa: bool = False,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(2048, d_model)
        self.enc_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, nhead, dim_feedforward, dropout, moe=None,
                    use_gqa=use_gqa, num_kv_heads=num_kv_heads
                )
                for _ in range(num_encoder_layers)
            ]
        )
        moe = SparseMoE(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_experts=int(moe_cfg["num_experts"]),
            k=int(moe_cfg["k"]),
            router=str(moe_cfg["router"]),
            dropout=dropout,
            use_lora_experts=bool(moe_cfg.get("use_lora_experts", False)),
            lora_r=int(moe_cfg.get("lora", {}).get("r", 16)),
            lora_alpha=int(moe_cfg.get("lora", {}).get("alpha", 32)),
            lora_dropout=float(moe_cfg.get("lora", {}).get("dropout", 0.05)),
        )
        self.dec_layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, nhead, dim_feedforward, dropout, moe=moe,
                    use_gqa=use_gqa, num_kv_heads=num_kv_heads
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self, 
        src_ids: torch.Tensor, 
        tgt_ids: torch.Tensor, 
        tgt_mask=None,
        past_key_values: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        device = src_ids.device
        # Embeddings
        src_pos = torch.arange(0, src_ids.size(1), device=device).unsqueeze(0)
        tgt_pos = torch.arange(0, tgt_ids.size(1), device=device).unsqueeze(0)
        src = self.embed(src_ids) + self.pos_embed(src_pos)
        tgt = self.embed(tgt_ids) + self.pos_embed(tgt_pos)

        # Encoder (no caching for encoder, it only runs once)
        for layer in self.enc_layers:
            src, _ = layer(src, src)
        
        # Decoder with KV caching support
        logs_all = {"assignments": []}
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.dec_layers):
            # Get past KV for this layer if available
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            if layer_past is not None:
                self_past_kv = layer_past.get('self_attn', None)
                cross_past_kv = layer_past.get('cross_attn', None)
            else:
                self_past_kv = None
                cross_past_kv = None
            
            tgt, logs = layer(
                tgt, src, 
                attn_mask=tgt_mask,
                self_attn_past_key_value=self_past_kv,
                cross_attn_past_key_value=cross_past_kv,
                use_cache=use_cache
            )
            
            if "assignment" in logs:
                # assignment is (batch, seq, k), flatten to (batch*seq, k)
                logs_all["assignments"].append(logs["assignment"].reshape(-1, logs["assignment"].size(-1)))
            
            if use_cache and "past_key_values" in logs:
                present_key_values.append(logs["past_key_values"])

        logits = self.lm_head(tgt)
        
        if use_cache:
            logs_all["past_key_values"] = present_key_values
        
        return logits, logs_all

