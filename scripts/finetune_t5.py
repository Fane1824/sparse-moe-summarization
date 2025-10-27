import os
import sys
import argparse
import json  # Added for jsonl output
import csv   # Added for csv output
from typing import Dict
import warnings

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from huggingface_hub import HfApi

# Assuming utils/config.py, data/xsum.py, and metrics/compute.py exist in src/
# If not, these imports might need adjustment based on your project structure
from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum, build_preprocess_fn
from src.metrics.compute import compute_all_metrics


def get_latest_local_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint in the local output directory.
    
    Returns:
        str: Path to latest checkpoint directory, or None if not found
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint-"):
            try:
                # Extract checkpoint number
                num = int(item.split("-")[1])
                checkpoints.append((num, item_path))
            except (ValueError, IndexError):
                continue
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number and return the latest
    checkpoints.sort(key=lambda x: x[0])
    latest_path = checkpoints[-1][1]
    print(f"Found latest local checkpoint: {latest_path}")
    return latest_path


def get_latest_checkpoint(repo_id: str) -> tuple:
    """Find the latest checkpoint subdirectory in the Hub repo.
    
    Returns:
        tuple: (repo_id, subfolder) where subfolder is None if no checkpoints found
    """
    if not repo_id:
        return None, None
    
    api = HfApi()
    try:
        repo_files = api.list_repo_files(repo_id)
        # Find all checkpoint directories (e.g., checkpoint-1000, checkpoint-2000)
        checkpoints = set()
        for file_path in repo_files:
            if "/" in file_path:
                subdir = file_path.split("/")[0]
                if subdir.startswith("checkpoint-"):
                    checkpoints.add(subdir)
        
        if not checkpoints:
            # No checkpoints found
            print(f"No checkpoints found in {repo_id}")
            return None, None
        
        # Sort by checkpoint number and get the latest
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0
        )
        latest = sorted_checkpoints[-1]
        print(f"Found latest checkpoint: {latest}")
        return repo_id, latest
    except Exception as e:
        print(f"Error finding checkpoints in {repo_id}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/t5_base")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="HF Hub repo ID (e.g., 'username/t5-base-xsum')")
    parser.add_argument("--resume_from_hub", action="store_true", help="Resume training from latest checkpoint in hub_repo_id")
    parser.add_argument("--resume_from_local", action="store_true", help="Resume training from latest local checkpoint in output dir")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ft_cfg = cfg["finetune"]["t5_base"]
    xsum_cfg = cfg["xsum"].copy()

    # Check if we should resume from local or hub
    resume_checkpoint = None
    
    # Priority: local checkpoint > hub checkpoint
    if args.resume_from_local:
        print(f"\n{'='*80}")
        print(f"🔄 RESUME MODE: Checking for local checkpoints in {args.output}")
        print(f"{'='*80}\n")
        local_checkpoint = get_latest_local_checkpoint(args.output)
        if local_checkpoint:
            resume_checkpoint = ("local", local_checkpoint)
            print(f"✓ Will resume from local checkpoint: {local_checkpoint}\n")
        else:
            print(f"⚠️  No local checkpoints found. Starting from base model: {ft_cfg['model_name']}\n")
    
    elif args.resume_from_hub and args.hub_repo_id:
        print(f"\n{'='*80}")
        print(f"🔄 RESUME MODE: Checking for checkpoints in {args.hub_repo_id}")
        print(f"{'='*80}\n")
        repo_id, subfolder = get_latest_checkpoint(args.hub_repo_id)
        if repo_id and subfolder:
            resume_checkpoint = ("hub", (repo_id, subfolder))
            print(f"✓ Will resume from: {repo_id}/{subfolder}\n")
        else:
            print(f"⚠️  No checkpoints found. Starting from base model: {ft_cfg['model_name']}\n")

    # Load model and tokenizer
    if resume_checkpoint:
        checkpoint_type, checkpoint_info = resume_checkpoint
        
        if checkpoint_type == "local":
            checkpoint_path = checkpoint_info
            print(f"Loading tokenizer from local checkpoint: {checkpoint_path}")
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            
            # Check if it's a PEFT model
            try:
                peft_config = PeftConfig.from_pretrained(checkpoint_path)
                base_model_name = peft_config.base_model_name_or_path
                print(f"Detected PEFT checkpoint. Loading base model: {base_model_name}")
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                print(f"Loading PEFT adapters from: {checkpoint_path}")
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                print("✓ Successfully loaded PEFT model from local checkpoint")
            except:
                # Not a PEFT model, load as full model
                print(f"Loading full model from local checkpoint: {checkpoint_path}")
                model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
                
                # If LoRA is enabled in config, we need to apply it
                if bool(ft_cfg.get("lora", {}).get("enabled", False)):
                    print("⚠️  LoRA enabled in config but checkpoint is not a PEFT model.")
                    print("    Applying LoRA to the loaded model (this will reset adapter weights).")
                    lora = ft_cfg["lora"]
                    lconf = LoraConfig(
                        r=int(lora.get("r", 16)),
                        lora_alpha=int(lora.get("alpha", 32)),
                        lora_dropout=float(lora.get("dropout", 0.05)),
                        target_modules=lora.get("target_modules", ["q", "k", "v", "o"]),
                        bias="none",
                        task_type="SEQ_2_SEQ_LM",
                    )
                    model = get_peft_model(model, lconf)
        
        else:  # checkpoint_type == "hub"
            repo_id, subfolder = checkpoint_info
            print(f"Loading tokenizer from checkpoint: {repo_id}/{subfolder}")
            tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
            
            # Check if it's a PEFT model
            try:
                peft_config = PeftConfig.from_pretrained(repo_id, subfolder=subfolder)
                base_model_name = peft_config.base_model_name_or_path
                print(f"Detected PEFT checkpoint. Loading base model: {base_model_name}")
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
                print(f"Loading PEFT adapters from: {repo_id}/{subfolder}")
                model = PeftModel.from_pretrained(base_model, repo_id, subfolder=subfolder)
                print("✓ Successfully loaded PEFT model from checkpoint")
            except:
                # Not a PEFT model, load as full model
                print(f"Loading full model from checkpoint: {repo_id}/{subfolder}")
                model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, subfolder=subfolder)
                
                # If LoRA is enabled in config, we need to apply it to the loaded model
                if bool(ft_cfg.get("lora", {}).get("enabled", False)):
                    print("⚠️  LoRA enabled in config but checkpoint is not a PEFT model.")
                    print("    Applying LoRA to the loaded model (this will reset adapter weights).")
                    lora = ft_cfg["lora"]
                    lconf = LoraConfig(
                        r=int(lora.get("r", 16)),
                        lora_alpha=int(lora.get("alpha", 32)),
                        lora_dropout=float(lora.get("dropout", 0.05)),
                        target_modules=lora.get("target_modules", ["q", "k", "v", "o"]),
                        bias="none",
                        task_type="SEQ_2_SEQ_LM",
                    )
                    model = get_peft_model(model, lconf)
    else:
        # Start from scratch with base model
        print(f"Loading base model: {ft_cfg['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(ft_cfg["model_name"])
        model = AutoModelForSeq2SeqLM.from_pretrained(ft_cfg["model_name"])

        if bool(ft_cfg.get("lora", {}).get("enabled", False)):
            print("Applying LoRA configuration...")
            lora = ft_cfg["lora"]
            lconf = LoraConfig(
                r=int(lora.get("r", 16)),
                lora_alpha=int(lora.get("alpha", 32)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                target_modules=lora.get("target_modules", ["q", "k", "v", "o"]),
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            model = get_peft_model(model, lconf)

    dataset, text_col, summary_col = load_xsum(
        xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"]
    )
    preprocess = build_preprocess_fn(tokenizer, xsum_cfg)
    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_metrics_fn(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # light metric during training (ROUGE only), full metrics after eval
        from evaluate import load as load_eval

        rouge = load_eval("rouge")
        res = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        return {"rouge1": res.get("rouge1", 0.0), "rouge2": res.get("rouge2", 0.0), "rougeL": res.get("rougeL", 0.0)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(ft_cfg["lr"]),
        per_device_train_batch_size=int(ft_cfg["batch_size"]),
        per_device_eval_batch_size=int(ft_cfg["batch_size"]),
        gradient_accumulation_steps=int(ft_cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(ft_cfg["num_train_epochs"]), 
        predict_with_generate=True,
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        max_grad_norm=1.0, 
        report_to=["none"],
        logging_steps=cfg["logging"]["log_steps"],
        hub_model_id=args.hub_repo_id if args.push_to_hub else None,
        push_to_hub=args.push_to_hub,
        push_to_hub_strategy="checkpoint" if args.push_to_hub else None,  # Push each checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=int(ft_cfg.get("early_stopping_patience", 3)),
        early_stopping_threshold=float(ft_cfg.get("early_stopping_threshold", 0.0))
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[early_stopping],
    )

    trainer.train()

    # Final evaluation on test with full metrics
    test_texts = dataset["test"][xsum_cfg["text_column"]]
    test_refs = dataset["test"][xsum_cfg["summary_column"]]

    preds = []
    device = model.device
    bs = int(ft_cfg["batch_size"]) if ft_cfg.get("batch_size") else 8
    print(f"Generating predictions on test set (batch size: {bs})...")
    for i in range(0, len(test_texts), bs):
        batch = test_texts[i : i + bs]
        inputs = tokenizer(
            batch,
            max_length=int(xsum_cfg["max_source_length"]),
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                num_beams=int(cfg["generation"]["num_beams"]),
                max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
                length_penalty=float(cfg["generation"]["length_penalty"]),
                no_repeat_ngram_size=int(cfg["generation"]["no_repeat_ngram_size"]),
            )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    
    print("Generation complete. Saving predictions...")

    # --- NEW: Save predictions to JSONL and CSV ---
    num_preds = len(preds)
    # Slice texts and refs to match the number of predictions, just in case
    sliced_texts = test_texts[:num_preds]
    sliced_refs = test_refs[:num_preds]

    # Save to JSONL
    jsonl_path = os.path.join(args.output, "predictions.jsonl")
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for text, ref, pred in zip(sliced_texts, sliced_refs, preds):
                record = {
                    "source_text": text,
                    "reference_summary": ref,
                    "generated_summary": pred
                }
                f.write(json.dumps(record) + "\n")
        print(f"Successfully saved predictions to {jsonl_path}")
    except Exception as e:
        print(f"Error saving predictions to JSONL: {e}")

    # Save to CSV
    csv_path = os.path.join(args.output, "predictions.csv")
    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["source_text", "reference_summary", "generated_summary"])
            # Write data
            for text, ref, pred in zip(sliced_texts, sliced_refs, preds):
                writer.writerow([text, ref, pred])
        print(f"Successfully saved predictions to {csv_path}")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
    # --- End of new code ---


    # Compute final metrics
    metrics = compute_all_metrics(preds, sliced_refs, sliced_texts, include_summac=False)
    
    # Save metrics to a text file
    with open(os.path.join(args.output, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("Final metrics:")
    print(metrics)

    # Push to hub
    if args.push_to_hub and args.hub_repo_id:
        try:
            print(f"Pushing model to hub: {args.hub_repo_id}")
            trainer.push_to_hub(commit_message="Trained on XSum dataset")
            tokenizer.push_to_hub(args.hub_repo_id, commit_message="Add tokenizer")
            print(f"Successfully pushed to https://huggingface.co/{args.hub_repo_id}")
        except Exception as e:
            print(f"Failed to push to hub: {e}")
            print("Make sure HUGGINGFACE_HUB_TOKEN is set and you have write access to the repo")


if __name__ == "__main__":
    main()