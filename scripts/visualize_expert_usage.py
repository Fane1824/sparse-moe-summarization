"""
Visualize Expert Usage Over Time for Trained MoE Models

This script:
1. Loads trained MoE checkpoints from the runs folder
2. Runs inference on test samples
3. Tracks expert assignments over time (token position)
4. Creates visualizations comparing different models
"""

import os
import sys
import argparse
from collections import Counter, defaultdict

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.models.transformer_moe import SimpleTransformerWithMoE


def load_model_from_checkpoint(checkpoint_path, cfg, device):
    """Load a trained MoE model from checkpoint."""
    checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    config_file = os.path.join(checkpoint_path, "moe_config.pt")
    
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    # Load config if available
    if os.path.exists(config_file):
        moe_cfg = torch.load(config_file)
    else:
        moe_cfg = cfg["moe"].copy()
    
    # Load checkpoint to check for GQA
    checkpoint = torch.load(checkpoint_file, map_location=device)
    use_gqa = checkpoint.get('use_gqa', False)
    num_kv_heads = checkpoint.get('num_kv_heads', None)
    
    # Create model
    model = SimpleTransformerWithMoE(
        vocab_size=int(cfg["moe"]["vocab_size"]),
        d_model=int(moe_cfg["d_model"]),
        nhead=int(moe_cfg["nhead"]),
        num_encoder_layers=int(moe_cfg["num_encoder_layers"]),
        num_decoder_layers=int(moe_cfg["num_decoder_layers"]),
        dim_feedforward=int(moe_cfg["dim_feedforward"]),
        dropout=float(moe_cfg["dropout"]),
        moe_cfg=moe_cfg,
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, moe_cfg


def collect_expert_assignments(model, tokenizer, texts, device, max_samples=100, max_gen_length=32):
    """
    Collect expert assignments during generation.
    
    Returns:
        expert_usage_over_time: dict mapping position -> Counter of expert IDs
        total_expert_usage: Counter of total expert usage
        expert_per_layer: dict mapping layer_idx -> Counter of expert usage
    """
    expert_usage_over_time = defaultdict(Counter)
    total_expert_usage = Counter()
    expert_per_layer = defaultdict(Counter)
    
    with torch.no_grad():
        for text in tqdm(texts[:max_samples], desc="Collecting expert assignments"):
            # Tokenize source
            src = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(device)
            
            # Start with pad token
            dec = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            
            # Generate tokens
            for pos in range(max_gen_length):
                logits, logs = model(src["input_ids"], dec)
                
                # Collect expert assignments from all decoder layers
                for layer_idx, assign in enumerate(logs.get("assignments", [])):
                    # assign shape: (batch*seq, k) where k is number of experts per token
                    # For single batch: (seq, k)
                    expert_ids = assign.view(-1).cpu().tolist()
                    
                    # Track overall usage
                    total_expert_usage.update(expert_ids)
                    
                    # Track per-layer usage
                    expert_per_layer[layer_idx].update(expert_ids)
                    
                    # Track usage at this generation position (only last token)
                    # Last seq_length tokens correspond to current decoder state
                    last_token_experts = assign[-1].cpu().tolist()  # Last token's experts
                    expert_usage_over_time[pos].update(last_token_experts)
                
                # Get next token
                next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                dec = torch.cat([dec, next_id], dim=1)
                
                # Early stopping
                if next_id.item() == tokenizer.eos_token_id:
                    break
    
    return expert_usage_over_time, total_expert_usage, expert_per_layer


def plot_expert_usage_distribution(expert_usage, num_experts, title, output_path):
    """Plot bar chart of total expert usage."""
    experts = list(range(num_experts))
    counts = [expert_usage[e] for e in experts]
    
    plt.figure(figsize=(10, 5))
    plt.bar(experts, counts, color='steelblue', alpha=0.8)
    plt.xlabel("Expert ID", fontsize=12)
    plt.ylabel("Number of Token Assignments", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_expert_usage_over_time(usage_over_time, num_experts, title, output_path):
    """Plot heatmap showing expert usage across generation positions."""
    positions = sorted(usage_over_time.keys())
    
    # Create matrix: positions x experts
    matrix = np.zeros((len(positions), num_experts))
    for i, pos in enumerate(positions):
        for expert_id, count in usage_over_time[pos].items():
            matrix[i, expert_id] = count
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        matrix.T, 
        cmap='YlOrRd', 
        xticklabels=positions if len(positions) <= 20 else 5,
        yticklabels=list(range(num_experts)),
        cbar_kws={'label': 'Number of Assignments'}
    )
    plt.xlabel("Generation Position (Token Index)", fontsize=12)
    plt.ylabel("Expert ID", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_expert_usage_per_layer(expert_per_layer, num_experts, title, output_path):
    """Plot expert usage distribution across different decoder layers."""
    num_layers = len(expert_per_layer)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 4), sharey=True)
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx in range(num_layers):
        experts = list(range(num_experts))
        counts = [expert_per_layer[layer_idx][e] for e in experts]
        
        axes[layer_idx].bar(experts, counts, color='teal', alpha=0.7)
        axes[layer_idx].set_xlabel("Expert ID", fontsize=10)
        axes[layer_idx].set_title(f"Layer {layer_idx}", fontsize=11, fontweight='bold')
        axes[layer_idx].grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel("Number of Token Assignments", fontsize=10)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_comparison_across_models(model_results, output_path):
    """Compare expert usage across multiple models."""
    num_models = len(model_results)
    num_experts = None
    
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 4), sharey=True)
    if num_models == 1:
        axes = [axes]
    
    for idx, (model_name, expert_usage) in enumerate(model_results.items()):
        if num_experts is None:
            num_experts = max(expert_usage.keys()) + 1
        
        experts = list(range(num_experts))
        counts = [expert_usage[e] for e in experts]
        
        axes[idx].bar(experts, counts, color=f'C{idx}', alpha=0.7)
        axes[idx].set_xlabel("Expert ID", fontsize=11)
        axes[idx].set_title(model_name, fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel("Number of Token Assignments", fontsize=11)
    fig.suptitle("Expert Usage Comparison Across Models", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize expert usage from trained MoE models")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--checkpoints", nargs='+', required=True, 
                       help="Paths to model checkpoints (e.g., runs/moe_topk/best_model runs/moe_hash/best_model)")
    parser.add_argument("--output", default="visual_eval_runs/visualizations")
    parser.add_argument("--max_samples", type=int, default=100, help="Number of test samples to analyze")
    parser.add_argument("--max_gen_length", type=int, default=32, help="Maximum generation length")
    parser.add_argument("--compare", action="store_true", help="Create comparison plot across all models")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)
    xsum_cfg = cfg["xsum"].copy()

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    print("\n" + "="*80)
    print("Loading dataset...")
    print("="*80)
    dataset, text_col, summary_col = load_xsum(
        xsum_cfg["dataset_name"], 
        xsum_cfg["text_column"], 
        xsum_cfg["summary_column"]
    )
    test_texts = dataset["test"][text_col]
    print(f"✓ Loaded {len(test_texts)} test samples\n")

    # Store results for comparison
    all_model_results = {}

    # Process each checkpoint
    for checkpoint_path in args.checkpoints:
        model_name = os.path.basename(os.path.dirname(checkpoint_path))
        
        print("="*80)
        print(f"Processing: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print("="*80)
        
        # Load model
        try:
            model, moe_cfg = load_model_from_checkpoint(checkpoint_path, cfg, device)
            num_experts = int(moe_cfg["num_experts"])
            print(f"✓ Model loaded (num_experts={num_experts})\n")
        except Exception as e:
            print(f"✗ Failed to load model: {e}\n")
            continue
        
        # Collect expert assignments
        print(f"Collecting expert assignments from {args.max_samples} samples...")
        usage_over_time, total_usage, usage_per_layer = collect_expert_assignments(
            model, tokenizer, test_texts, device, 
            max_samples=args.max_samples,
            max_gen_length=args.max_gen_length
        )
        
        # Store for comparison
        all_model_results[model_name] = total_usage
        
        # Create visualizations for this model
        model_output_dir = os.path.join(args.output, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # 1. Total expert usage distribution
        plot_expert_usage_distribution(
            total_usage, 
            num_experts,
            f"Expert Usage Distribution - {model_name}",
            os.path.join(model_output_dir, "expert_distribution.png")
        )
        
        # 2. Expert usage over time (heatmap)
        plot_expert_usage_over_time(
            usage_over_time,
            num_experts,
            f"Expert Usage Over Generation Time - {model_name}",
            os.path.join(model_output_dir, "expert_usage_over_time.png")
        )
        
        # 3. Expert usage per decoder layer
        plot_expert_usage_per_layer(
            usage_per_layer,
            num_experts,
            f"Expert Usage Per Decoder Layer - {model_name}",
            os.path.join(model_output_dir, "expert_usage_per_layer.png")
        )
        
        # Save statistics
        stats_file = os.path.join(model_output_dir, "expert_stats.txt")
        with open(stats_file, 'w') as f:
            f.write(f"Expert Usage Statistics - {model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total experts: {num_experts}\n")
            f.write(f"Total assignments: {sum(total_usage.values())}\n")
            f.write(f"Active experts: {len([e for e in range(num_experts) if total_usage[e] > 0])}\n\n")
            
            f.write("Expert usage counts:\n")
            for expert_id in range(num_experts):
                count = total_usage[expert_id]
                percentage = 100 * count / sum(total_usage.values()) if sum(total_usage.values()) > 0 else 0
                f.write(f"  Expert {expert_id:2d}: {count:6d} assignments ({percentage:5.2f}%)\n")
        
        print(f"✓ Saved statistics: {stats_file}\n")
    
    # Create comparison plot if requested
    if args.compare and len(all_model_results) > 1:
        print("="*80)
        print("Creating comparison visualization...")
        print("="*80)
        plot_comparison_across_models(
            all_model_results,
            os.path.join(args.output, "model_comparison.png")
        )
        print()
    
    print("="*80)
    print(f"✓ All visualizations saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
