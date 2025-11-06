import os
import sys
import argparse
import json
import csv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.models.transformer_moe import SimpleTransformerWithMoE
from src.metrics.compute import compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE model from local checkpoint or HF Hub")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/moe_eval")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint path (e.g., runs/moe_topk/best_model)")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="HF Hub repo ID (e.g., 'username/moe-topk-xsum')")
    parser.add_argument("--hub_checkpoint", type=str, default="best_model", help="Checkpoint name in hub repo")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate (default: all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--include_summac", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)
    moe_cfg = cfg["moe"].copy()
    xsum_cfg = cfg["xsum"].copy()

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine checkpoint path
    if args.hub_repo_id:
        # Load from HuggingFace Hub
        print(f"\n{'='*80}")
        print(f"📦 Loading MoE model from HuggingFace Hub")
        print(f"{'='*80}")
        print(f"Repo: {args.hub_repo_id}")
        print(f"Checkpoint: {args.hub_checkpoint}")
        print(f"{'='*80}\n")
        
        from huggingface_hub import hf_hub_download
        
        # Download checkpoint files
        model_file = hf_hub_download(
            repo_id=args.hub_repo_id,
            filename=f"{args.hub_checkpoint}/pytorch_model.bin",
            repo_type="model"
        )
        config_file = hf_hub_download(
            repo_id=args.hub_repo_id,
            filename=f"{args.hub_checkpoint}/moe_config.pt",
            repo_type="model"
        )
        
        # Load config
        moe_cfg = torch.load(config_file)
        checkpoint_path = model_file
        
    elif args.checkpoint:
        # Load from local checkpoint
        print(f"\n{'='*80}")
        print(f"📂 Loading MoE model from local checkpoint")
        print(f"{'='*80}")
        print(f"Path: {args.checkpoint}")
        print(f"{'='*80}\n")
        
        checkpoint_path = os.path.join(args.checkpoint, "pytorch_model.bin")
        config_path = os.path.join(args.checkpoint, "moe_config.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if os.path.exists(config_path):
            moe_cfg = torch.load(config_path)
    else:
        raise ValueError("Must specify either --checkpoint (local) or --hub_repo_id (HF Hub)")

    # Create model
    # Load GQA settings from checkpoint if available
    use_gqa = False
    num_kv_heads = None
    
    if args.hub_repo_id:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'use_gqa' in checkpoint:
        use_gqa = checkpoint['use_gqa']
        num_kv_heads = checkpoint.get('num_kv_heads', None)
        if use_gqa:
            print(f"✓ Model uses Grouped Query Attention (num_kv_heads={num_kv_heads})")
        else:
            print("✓ Model uses standard Multi-Head Attention")
    
    if moe_cfg.get('use_lora_experts', False):
        print(f"✓ Model uses LoRA-based experts")
    
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
    print("✓ Model loaded successfully\n")

    dataset, text_col, summary_col = load_xsum(xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"])
    
    # Limit samples if requested
    max_samples = args.max_samples if args.max_samples else len(dataset["test"])
    
    print(f"Generating summaries for {max_samples} test samples with batch_size={args.batch_size}...")

    preds = []
    all_texts = []
    all_refs = []
    
    # Enable mixed precision for faster inference (works for both GQA and MHA)
    use_amp = torch.cuda.is_available()
    
    with torch.no_grad():
        # Use automatic mixed precision if available
        if use_amp:
            with torch.cuda.amp.autocast():
                # Process in batches - load data on-the-fly
                for batch_start in tqdm(range(0, max_samples, args.batch_size), desc="Batches"):
                    batch_end = min(batch_start + args.batch_size, max_samples)
                    
                    # Load only the current batch from dataset
                    batch_texts = dataset["test"][text_col][batch_start:batch_end]
                    batch_refs = dataset["test"][summary_col][batch_start:batch_end]
                    
                    # Store for later metrics computation
                    all_texts.extend(batch_texts)
                    all_refs.extend(batch_refs)
                    
                    # Tokenize batch
                    src = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        truncation=True, 
                        padding=True, 
                        max_length=xsum_cfg["max_source_length"]
                    ).to(device)
                    
                    batch_size = len(batch_texts)
                    dec = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    
                    # Track which sequences are finished
                    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                    
                    # KV cache (only works with GQA models)
                    past_key_values = None
                    
                    # Generate for this batch with early stopping and KV caching
                    max_length = int(cfg["generation"]["max_new_tokens"])
                    for step in range(max_length):
                        # Only process unfinished sequences
                        if finished.all():
                            break
                        
                        if use_gqa:
                            # Use KV caching for GQA models
                            if step == 0:
                                # First step: process full decoder input
                                logits, logs = model(src["input_ids"], dec, use_cache=True)
                            else:
                                # Subsequent steps: only process new token with cache
                                logits, logs = model(
                                    src["input_ids"], 
                                    dec[:, -1:],  # Only the last token
                                    past_key_values=past_key_values,
                                    use_cache=True
                                )
                            past_key_values = logs.get("past_key_values", None)
                        else:
                            # No caching for standard MHA models
                            logits, _ = model(src["input_ids"], dec)
                        
                        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                        dec = torch.cat([dec, next_id], dim=1)
                        
                        # Mark sequences that generated EOS token
                        finished |= (next_id.squeeze(-1) == eos_token_id)
                    
                    # Decode batch predictions
                    for i in range(batch_size):
                        text = tokenizer.decode(dec[i], skip_special_tokens=True)
                        preds.append(text)
        else:
            # CPU inference without AMP
            for batch_start in tqdm(range(0, max_samples, args.batch_size), desc="Batches"):
                batch_end = min(batch_start + args.batch_size, max_samples)
                
                # Load only the current batch from dataset
                batch_texts = dataset["test"][text_col][batch_start:batch_end]
                batch_refs = dataset["test"][summary_col][batch_start:batch_end]
                
                # Store for later metrics computation
                all_texts.extend(batch_texts)
                all_refs.extend(batch_refs)
                
                # Tokenize batch
                src = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=xsum_cfg["max_source_length"]
                ).to(device)
                
                batch_size = len(batch_texts)
                dec = torch.full((batch_size, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
                
                # Track which sequences are finished
                finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
                eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                
                # KV cache (only works with GQA models)
                past_key_values = None
                
                # Generate for this batch with early stopping and KV caching
                max_length = int(cfg["generation"]["max_new_tokens"])
                for step in range(max_length):
                    # Only process unfinished sequences
                    if finished.all():
                        break
                    
                    if use_gqa:
                        # Use KV caching for GQA models
                        if step == 0:
                            # First step: process full decoder input
                            logits, logs = model(src["input_ids"], dec, use_cache=True)
                        else:
                            # Subsequent steps: only process new token with cache
                            logits, logs = model(
                                src["input_ids"], 
                                dec[:, -1:],  # Only the last token
                                past_key_values=past_key_values,
                                use_cache=True
                            )
                        past_key_values = logs.get("past_key_values", None)
                    else:
                        # No caching for standard MHA models
                        logits, _ = model(src["input_ids"], dec)
                    
                    next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                    dec = torch.cat([dec, next_id], dim=1)
                    
                    # Mark sequences that generated EOS token
                    finished |= (next_id.squeeze(-1) == eos_token_id)
                
                # Decode batch predictions
                for i in range(batch_size):
                    text = tokenizer.decode(dec[i], skip_special_tokens=True)
                    preds.append(text)

    print("\nGeneration complete. Saving predictions...")
    
    # Save predictions to JSONL
    jsonl_path = os.path.join(args.output, "moe_predictions.jsonl")
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for text, ref, pred in zip(all_texts, all_refs, preds):
                record = {
                    "source_text": text,
                    "reference_summary": ref,
                    "generated_summary": pred
                }
                f.write(json.dumps(record) + "\n")
        print(f"✓ Saved predictions to {jsonl_path}")
    except Exception as e:
        print(f"✗ Error saving predictions to JSONL: {e}")

    # Save predictions to CSV
    csv_path = os.path.join(args.output, "moe_predictions.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source_text", "reference_summary", "generated_summary"])
            for text, ref, pred in zip(all_texts, all_refs, preds):
                writer.writerow([text, ref, pred])
        print(f"✓ Saved predictions to {csv_path}")
    except Exception as e:
        print(f"✗ Error saving predictions to CSV: {e}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(preds, all_refs, all_texts, include_summac=args.include_summac)
    
    metrics_path = os.path.join(args.output, "metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.4f}")
        else:
            print(f"{k:20s}: {v}")
    print("="*80)
    print(f"\n✓ Metrics saved to {metrics_path}")
    print(f"✓ All results saved to {args.output}/")
    print(f"  - Predictions (JSONL): moe_predictions.jsonl")
    print(f"  - Predictions (CSV): moe_predictions.csv")
    print(f"  - Metrics: metrics.txt")


if __name__ == "__main__":
    main()

