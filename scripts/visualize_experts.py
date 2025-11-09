import os
import sys
import argparse
from collections import Counter

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.models.transformer_moe import SimpleTransformerWithMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/moe_viz")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)
    moe_cfg = cfg["moe"].copy()
    xsum_cfg = cfg["xsum"].copy()

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleTransformerWithMoE(
        vocab_size=int(cfg["moe"]["vocab_size"]),
        d_model=int(moe_cfg["d_model"]),
        nhead=int(moe_cfg["nhead"]),
        num_encoder_layers=int(moe_cfg["num_encoder_layers"]),
        num_decoder_layers=int(moe_cfg["num_decoder_layers"]),
        dim_feedforward=int(moe_cfg["dim_feedforward"]),
        dropout=float(moe_cfg["dropout"]),
        moe_cfg=moe_cfg,
    ).to(device)

    dataset, text_col, summary_col = load_xsum(xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"])
    texts = dataset["validation"][text_col][:64]

    usage = Counter()
    with torch.no_grad():
        for t in texts:
            src = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=xsum_cfg["max_source_length"]).to(device)
            dec = torch.full((1, 4), tokenizer.pad_token_id, dtype=torch.long, device=device)  # few steps
            logits, logs = model(src["input_ids"], dec)
            for assign in logs.get("assignments", []):
                vals = assign.view(-1).tolist()
                usage.update(vals)

    experts = list(range(int(moe_cfg["num_experts"])))
    counts = [usage[e] for e in experts]
    plt.figure(figsize=(8, 4))
    plt.bar(experts, counts)
    plt.xlabel("Expert ID")
    plt.ylabel("Token Assignments (approx)")
    plt.title("MoE Expert Usage (Validation subset)")
    plt.tight_layout()
    out_path = os.path.join(args.output, "expert_usage.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

