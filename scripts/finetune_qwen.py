import os
import sys
import argparse
from typing import Dict
import warnings
import json
import csv

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,        
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from huggingface_hub import HfApi
from tqdm import tqdm

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
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


def format_instruction(doc: str, summary: str | None = None) -> str:
    prompt = (
        "Summarize the following news article in one sentence.\n\n"
        f"Article:\n{doc}\n\nSummary:"
    )
    if summary is not None:
        # Note: Added a space before {summary}
        return prompt + f" {summary}"
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/qwen_3b")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="HF Hub repo ID (e.g., 'username/qwen-3b-xsum')")
    parser.add_argument("--resume_from_hub", action="store_true", help="Resume training from latest checkpoint in hub_repo_id")
    parser.add_argument("--resume_from_local", action="store_true", help="Resume training from latest local checkpoint in output dir")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ft_cfg = cfg["finetune"]["qwen_3b"]
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
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                print(f"Loading PEFT adapters from: {checkpoint_path}")
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
                print("✓ Successfully loaded PEFT model from local checkpoint")
            except:
                # Not a PEFT model, load as full model
                print(f"Loading full model from local checkpoint: {checkpoint_path}")
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
                
                # If LoRA is enabled in config, we need to apply it
                if bool(ft_cfg.get("lora", {}).get("enabled", True)):
                    print("⚠️  LoRA enabled in config but checkpoint is not a PEFT model.")
                    print("    Applying LoRA to the loaded model (this will reset adapter weights).")
                    lora = ft_cfg["lora"]
                    lconf = LoraConfig(
                        r=int(lora.get("r", 16)),
                        lora_alpha=int(lora.get("alpha", 32)),
                        lora_dropout=float(lora.get("dropout", 0.05)),
                        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                        bias="none",
                        task_type="CAUSAL_LM",
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
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                print(f"Loading PEFT adapters from: {repo_id}/{subfolder}")
                model = PeftModel.from_pretrained(base_model, repo_id, subfolder=subfolder)
                print("✓ Successfully loaded PEFT model from checkpoint")
            except:
                # Not a PEFT model, load as full model
                print(f"Loading full model from checkpoint: {repo_id}/{subfolder}")
                model = AutoModelForCausalLM.from_pretrained(repo_id, subfolder=subfolder)
                
                # If LoRA is enabled in config, we need to apply it to the loaded model
                if bool(ft_cfg.get("lora", {}).get("enabled", True)):
                    print("⚠️  LoRA enabled in config but checkpoint is not a PEFT model.")
                    print("    Applying LoRA to the loaded model (this will reset adapter weights).")
                    lora = ft_cfg["lora"]
                    lconf = LoraConfig(
                        r=int(lora.get("r", 16)),
                        lora_alpha=int(lora.get("alpha", 32)),
                        lora_dropout=float(lora.get("dropout", 0.05)),
                        target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                        bias="none",
                        task_type="CAUSAL_LM",
                    )
                    model = get_peft_model(model, lconf)
    else:
        # Start from scratch with base model
        print(f"Loading base model: {ft_cfg['model_name']}")
        tokenizer = AutoTokenizer.from_pretrained(ft_cfg["model_name"])
        model = AutoModelForCausalLM.from_pretrained(ft_cfg["model_name"])

        if bool(ft_cfg.get("lora", {}).get("enabled", True)):
            print("Applying LoRA configuration...")
            lora = ft_cfg["lora"]
            lconf = LoraConfig(
                r=int(lora.get("r", 16)),
                lora_alpha=int(lora.get("alpha", 32)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                target_modules=lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lconf)
    
    # Set pad token for tokenizer and model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set pad token for tokenizer and model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- FIX 1: Set default padding side to RIGHT for training ---
    # The tokenize_fn will use this default
    tokenizer.padding_side = "right"

    # Note: This is important for generation, let the model know
    # which token is for padding
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset, text_col, summary_col = load_xsum(
        xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"]
    )
    
    # Get max length from config, provide a default
    max_length_cfg = xsum_cfg.get("max_source_length", 512)
    # Ensure it's an integer
    try:
        max_length = int(max_length_cfg)
    except ValueError:
        print(f"Warning: Invalid max_source_length '{max_length_cfg}', using 512.")
        max_length = 512

    def tokenize_fn(example):
        # This formats the prompt *without* the summary
        prompt_text = format_instruction(example[text_col])
        # This formats the full text *with* the summary
        # We add the EOS token here to signal the end of the summary
        full_text = format_instruction(example[text_col], example[summary_col]) + tokenizer.eos_token

        # Tokenize full text with truncation AND PADDING (uses tokenizer.padding_side)
        full_tok = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        # Tokenize prompt *without* truncation to get its original length
        # We don't pad this one, we just need the length
        prompt_tok = tokenizer(prompt_text, truncation=False, padding=False)

        # Get the length of the *full tokenized sequence* (now always max_length)
        full_len = len(full_tok["input_ids"])
        # Get the length of the *prompt*
        prompt_len = len(prompt_tok["input_ids"])

        labels = full_tok["input_ids"].copy()
        
        # Determine the actual length of the prompt *as it appears* in the
        # truncated full_tok. It's the smaller of the two lengths.
        mask_len = min(prompt_len, full_len)
        
        # Mask the prompt part by setting to -100
        labels[:mask_len] = [-100] * mask_len
        
        # Also mask all padding tokens in labels
        for i in range(len(labels)):
            if full_tok["attention_mask"][i] == 0:
                labels[i] = -100

        full_tok["labels"] = labels
        return full_tok

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(ft_cfg["lr"]),
        per_device_train_batch_size=int(ft_cfg["batch_size"]),
        per_device_eval_batch_size=int(ft_cfg["batch_size"]),
        gradient_accumulation_steps=int(ft_cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(ft_cfg["num_train_epochs"]),
        fp16=bool(ft_cfg.get("fp16", True)) and torch.cuda.is_available(),
        report_to=["none"],
        logging_steps=cfg["logging"]["log_steps"],
        hub_model_id=args.hub_repo_id if args.push_to_hub else None,
        push_to_hub=args.push_to_hub,
        push_to_hub_strategy="checkpoint" if args.push_to_hub else None,  # Push each checkpoint
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=int(ft_cfg.get("early_stopping_patience", 3)),
        early_stopping_threshold=float(ft_cfg.get("early_stopping_threshold", 0.0))
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    trainer.train()

    # Final evaluation on test set with generation and full metrics
    print("\n--- Starting Final Evaluation on Test Set ---")
    test_texts = dataset["test"][text_col]
    test_refs = dataset["test"][summary_col]

    preds = []
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(ft_cfg["batch_size"]) if ft_cfg.get("batch_size") else 1
    
    # --- FIX 2: Set padding side to LEFT for batched generation ---
    tokenizer.padding_side = "left"

    for i in tqdm(range(0, len(test_texts), bs), desc="Generating Predictions"):
        batch = test_texts[i : i + bs]
        prompts = [format_instruction(x) for x in batch]
        
        # This tokenizer call will now use "left" padding
        inputs = tokenizer(
            prompts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get the length of the input prompt
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                num_beams=int(cfg["generation"]["num_beams"]),
                max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
                no_repeat_ngram_size=int(cfg["generation"]["no_repeat_ngram_size"]),
            )
        
        # Get only the tokens *after* the input prompt
        output_tokens = gen[:, input_length:]

        # Decode just the new tokens
        outs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        
        preds.extend([out.strip() for out in outs])

    # --- FIX 4: Reset tokenizer padding side to default (right) ---
    tokenizer.padding_side = "right"


    # Save metrics
    print("\n--- Calculating Metrics ---")
    metrics = compute_all_metrics(preds, test_refs[: len(preds)], test_texts[: len(preds)], include_summac=False)
    metrics_path = os.path.join(args.output, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Metrics saved to {metrics_path}")
    print(metrics)

    # Save predictions
    jsonl_path = os.path.join(args.output, "test_predictions.jsonl")
    csv_path = os.path.join(args.output, "test_predictions.csv")
    
    to_save = []
    for src, ref, pred in zip(test_texts, test_refs, preds):
        to_save.append({
            "source_text": src,
            "reference_summary": ref,
            "generated_prediction": pred
        })

    # Write JSONL
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in to_save:
            f.write(json.dumps(item) + "\n")
    print(f"Predictions saved to {jsonl_path}")

    # Write CSV
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["source_text", "reference_summary", "generated_prediction"])
        writer.writeheader()
        writer.writerows(to_save)
    print(f"Predictions saved to {csv_path}")


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