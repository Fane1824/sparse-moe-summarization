import os
import sys
import argparse
import json
import csv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.models.transformer_moe import SimpleTransformerWithMoE
from src.moe.layer import SparseMoE
from src.metrics.compute import compute_all_metrics


def generate_square_subsequent_mask(sz: int, device) -> torch.Tensor:
    """Generate a causal mask for the decoder to prevent attending to future positions."""
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def collate_batch(batch, tokenizer, max_source_len: int, max_target_len: int):
    docs = [b["document"] for b in batch]
    sums = [b["summary"] for b in batch]
    src = tokenizer(docs, truncation=True, padding=True, max_length=max_source_len, return_tensors="pt")
    tgt = tokenizer(sums, truncation=True, padding=True, max_length=max_target_len, return_tensors="pt")
    return src["input_ids"], tgt["input_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/moe_topk")
    parser.add_argument("--router", choices=["topk", "hash"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="HF Hub repo ID (e.g., 'username/moe-topk-xsum')")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--use_gqa", action="store_true", help="Use Grouped Query Attention instead of standard MHA")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of KV heads for GQA (default: same as num_heads for MHA)")
    parser.add_argument("--use_lora_experts", action="store_true", help="Use LoRA-based experts in MoE layers")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank (default: from config)")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha scaling (default: from config)")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout (default: from config)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint in the output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)
    moe_cfg = cfg["moe"].copy()
    if args.router is not None:
        moe_cfg["router"] = args.router

    if moe_cfg["router"] == "hash":
        moe_cfg["k"] = 1

    # Override LoRA experts setting if specified via command line
    if args.use_lora_experts:
        moe_cfg["use_lora_experts"] = True
        # Override LoRA hyperparameters if specified
        if args.lora_r is not None:
            moe_cfg["lora"]["r"] = args.lora_r
        if args.lora_alpha is not None:
            moe_cfg["lora"]["alpha"] = args.lora_alpha
        if args.lora_dropout is not None:
            moe_cfg["lora"]["dropout"] = args.lora_dropout
        print(f"🔧 Using LoRA-based experts (r={moe_cfg['lora']['r']}, alpha={moe_cfg['lora']['alpha']}, dropout={moe_cfg['lora']['dropout']})")
    
    # Set up GQA configuration
    use_gqa = args.use_gqa
    num_kv_heads = args.num_kv_heads
    if use_gqa:
        if num_kv_heads is None:
            # Default: use num_heads // 2 for GQA
            num_kv_heads = int(moe_cfg["nhead"]) // 2
        print(f"🔧 Using Grouped Query Attention (num_heads={moe_cfg['nhead']}, num_kv_heads={num_kv_heads})")
    else:
        print("🔧 Using standard Multi-Head Attention")

    xsum_cfg = cfg["xsum"].copy()
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

    dataset, text_col, summary_col = load_xsum(xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"])
    train_dl = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer, xsum_cfg["max_source_length"], xsum_cfg["max_target_length"]),
    )
    val_dl = DataLoader(
        dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer, xsum_cfg["max_source_length"], xsum_cfg["max_target_length"]),
    )

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
        use_gqa=use_gqa,
        num_kv_heads=num_kv_heads,
    ).to(device)

    # CRITICAL: Lower learning rate for training transformers from scratch
    # 3e-4 is too high and causes mode collapse
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Add learning rate scheduler with warmup
    num_training_steps = len(train_dl) * (args.epochs or 1)
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=num_training_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        div_factor=10.0,  # start_lr = max_lr / div_factor = 1e-5
        final_div_factor=100.0  # end_lr = max_lr / final_div_factor = 1e-6
    )
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    epochs = args.epochs or 1
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if requested
    if args.resume:
        # Find the latest checkpoint
        checkpoint_dirs = []
        if os.path.exists(args.output):
            for item in os.listdir(args.output):
                item_path = os.path.join(args.output, item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-epoch-"):
                    try:
                        epoch_num = int(item.split("-")[-1])
                        checkpoint_dirs.append((epoch_num, item_path))
                    except ValueError:
                        continue
        
        if checkpoint_dirs:
            # Sort by epoch number and get the latest
            checkpoint_dirs.sort(reverse=True)
            latest_epoch, latest_checkpoint = checkpoint_dirs[0]
            checkpoint_file = os.path.join(latest_checkpoint, "pytorch_model.bin")
            
            if os.path.exists(checkpoint_file):
                print(f"\n{'='*80}")
                print(f"📂 Resuming from checkpoint: {latest_checkpoint}")
                print(f"{'='*80}\n")
                
                checkpoint = torch.load(checkpoint_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                
                # Load GQA settings from checkpoint
                if 'use_gqa' in checkpoint:
                    use_gqa = checkpoint['use_gqa']
                    num_kv_heads = checkpoint.get('num_kv_heads', None)
                
                print(f"✓ Loaded checkpoint from epoch {start_epoch}")
                print(f"✓ Best validation loss so far: {best_val_loss:.4f}")
                print(f"✓ Will continue training from epoch {start_epoch + 1} to {epochs}")
                
                # Recreate scheduler with correct step count
                steps_completed = start_epoch * len(train_dl)
                num_training_steps = len(train_dl) * epochs
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=1e-4,
                    total_steps=num_training_steps,
                    pct_start=0.1,
                    anneal_strategy='cos',
                    div_factor=10.0,
                    final_div_factor=100.0,
                    last_epoch=steps_completed - 1
                )
                print(f"✓ Scheduler resumed from step {steps_completed}\n")
            else:
                print(f"⚠ Checkpoint file not found: {checkpoint_file}")
                print("Starting training from scratch...\n")
        else:
            print("⚠ No checkpoints found in output directory")
            print("Starting training from scratch...\n")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total = 0.0
        for src_ids, tgt_ids in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)
            # teacher forcing: input is shifted right, targets are shifted left
            tgt_in = tgt_ids[:, :-1]
            tgt_out = tgt_ids[:, 1:]
            
            # Create causal mask for decoder
            tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), device)
            
            logits, logs = model(src_ids, tgt_in, tgt_mask=tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            # add load balancer loss
            if logs.get("assignments"):
                assign = torch.cat(logs["assignments"], dim=0)
                lb_loss = SparseMoE.aux_load_balancer_loss(assign, int(moe_cfg["num_experts"]))
                loss = loss + float(moe_cfg["load_balancer_weight"]) * lb_loss
            optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Update learning rate after each step
            total += loss.item()
        
        avg_train_loss = total / max(1, len(train_dl))
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} train loss: {avg_train_loss:.4f}, lr: {current_lr:.2e}")

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_ids, tgt_ids in val_dl:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)
                tgt_in = tgt_ids[:, :-1]
                tgt_out = tgt_ids[:, 1:]
                
                tgt_mask = generate_square_subsequent_mask(tgt_in.size(1), device)
                logits, logs = model(src_ids, tgt_in, tgt_mask=tgt_mask)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / max(1, len(val_dl))
        print(f"Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

        # Save checkpoint every N epochs or if best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        if (epoch + 1) % args.save_every == 0 or is_best:
            checkpoint_name = f"checkpoint-epoch-{epoch+1}"
            if is_best:
                checkpoint_name = "best_model"
            
            checkpoint_path = os.path.join(args.output, checkpoint_name)
            os.makedirs(checkpoint_path, exist_ok=True)
            
            # Save model, optimizer, and config
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'use_gqa': use_gqa,
                'num_kv_heads': num_kv_heads,
            }, os.path.join(checkpoint_path, "pytorch_model.bin"))
            
            torch.save(moe_cfg, os.path.join(checkpoint_path, "moe_config.pt"))
            
            # Save training info
            info = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'router_type': moe_cfg['router'],
                'num_experts': moe_cfg['num_experts'],
                'use_lora_experts': moe_cfg.get('use_lora_experts', False),
                'use_gqa': use_gqa,
                'num_kv_heads': num_kv_heads,
            }
            with open(os.path.join(checkpoint_path, "training_info.json"), "w") as f:
                json.dump(info, f, indent=2)
            
            print(f"{'✓ Best model' if is_best else 'Checkpoint'} saved to {checkpoint_path}")
            
            # Push to hub if requested
            if args.push_to_hub and args.hub_repo_id:
                try:
                    from huggingface_hub import HfApi, create_repo
                    
                    api = HfApi()
                    try:
                        create_repo(args.hub_repo_id, repo_type="model", exist_ok=True)
                    except Exception:
                        pass
                    
                    # Upload checkpoint folder
                    api.upload_folder(
                        folder_path=checkpoint_path,
                        repo_id=args.hub_repo_id,
                        path_in_repo=checkpoint_name,
                        repo_type="model",
                        commit_message=f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}",
                    )
                    print(f"  Pushed to hub: {args.hub_repo_id}/{checkpoint_name}")
                except Exception as e:
                    print(f"  Warning: Failed to push checkpoint to hub: {e}")

        # quick val generation for a few samples
        with torch.no_grad():
            samples = [dataset["validation"][i] for i in range(8)]
            texts = [s[text_col] for s in samples]
            refs = [s[summary_col] for s in samples]
            preds = []
            for t in texts:
                src = tokenizer(t, return_tensors="pt", truncation=True, padding=True, max_length=xsum_cfg["max_source_length"]).to(device)
                # greedy decode for simplicity
                dec = torch.full((1, 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
                for _ in range(int(cfg["generation"]["max_new_tokens"])):
                    # No mask needed for inference (only generating one token at a time)
                    logits, _ = model(src["input_ids"], dec)
                    next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                    dec = torch.cat([dec, next_id], dim=1)
                text = tokenizer.decode(dec[0], skip_special_tokens=True)
                preds.append(text)
            metrics = compute_all_metrics(preds, refs, texts, include_summac=False)
            print(f"  Val sample metrics: {{{', '.join([f'{k}: {round(v, 4)}' for k, v in metrics.items() if isinstance(v, (int, float))])}}}")

    # Final save
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == "__main__":
    main()

