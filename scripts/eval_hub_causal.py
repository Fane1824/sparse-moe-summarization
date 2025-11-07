import os
import sys
import argparse
import json
import csv
from typing import List
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.metrics.compute import compute_all_metrics


def format_instruction(doc: str) -> str:
    return (
        "Summarize the following news article in one sentence.\n\n"
        f"Article:\n{doc}\n\nSummary:"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a causal LM from HF Hub on XSum (local or HF)")
    parser.add_argument("--hub_repo_id", required=True, help="HF Hub repo id, e.g. username/qwen-3b-xsum")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/eval_hub_causal")
    parser.add_argument("--include_summac", action="store_true")
    
    # OPTIMIZATION: Added --batch_size argument
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)

    xsum_cfg = cfg["xsum"].copy()
    gen_cfg = cfg["generation"].copy()

    print(f"Loading model from Hub: {args.hub_repo_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PEFT-aware loading: Check if the repo contains PEFT adapters
    try:
        # Try to load as a PEFT model (LoRA adapters)
        peft_config = PeftConfig.from_pretrained(args.hub_repo_id)
        base_model_name = peft_config.base_model_name_or_path
        
        print("\n" + "="*80)
        print("🔧 PEFT MODE DETECTED")
        print("="*80)
        print(f"✓ Found LoRA adapters in: {args.hub_repo_id}")
        print(f"✓ Base model: {base_model_name}")
        print(f"✓ Loading tokenizer from: {args.hub_repo_id}")
        print("="*80 + "\n")
        
        tokenizer = AutoTokenizer.from_pretrained(args.hub_repo_id, padding_side='left')
        
        print(f"Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
        )
        
        print(f"Applying PEFT adapters from: {args.hub_repo_id}...")
        model = PeftModel.from_pretrained(base_model, args.hub_repo_id)
        
        print("\n" + "="*80)
        print("✅ Successfully loaded FINETUNED model with PEFT adapters")
        print("="*80 + "\n")
        
    except Exception as e:
        # If PEFT loading fails, fall back to loading as a full model
        print("\n" + "="*80)
        print("⚠️  FULL MODEL MODE (No PEFT adapters detected)")
        print("="*80)
        print(f"Reason: {str(e)[:100]}")
        print(f"Loading as standard model from: {args.hub_repo_id}")
        print("="*80 + "\n")
        
        print(f"Loading tokenizer: {args.hub_repo_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.hub_repo_id, padding_side='left')
        
        print(f"Loading model: {args.hub_repo_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.hub_repo_id,
            torch_dtype=torch.bfloat16,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"Loading dataset: {xsum_cfg['dataset_name']}")
    dataset, text_col, summary_col = load_xsum(xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"])
    
    # This is correct, it's only using the test set
    test_ds = dataset["test"]

    texts = test_ds[text_col]
    refs = test_ds[summary_col]
    preds: List[str] = []

    # OPTIMIZATION: Use the batch size from args
    bs = args.batch_size
    print(f"Generating summaries for {len(texts)} samples with batch size {bs}...")

    # OPTIMIZATION: Use tqdm for a progress bar
    for i in tqdm(range(0, len(texts), bs), desc="Generating Summaries"):
        batch = texts[i : i + bs]
        prompts = [format_instruction(x) for x in batch]
        
        inputs = tokenizer(
            prompts,
            max_length=int(xsum_cfg["max_source_length"]),
            truncation=True,
            padding=True,  # This will now use left-padding
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # FIX: Get input length for robust prompt stripping
            input_token_len = inputs["input_ids"].shape[1]

            gen = model.generate(
                **inputs,
                num_beams=int(gen_cfg["num_beams"]),
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                no_repeat_ngram_size=int(gen_cfg["no_repeat_ngram_size"]),
            )
            
            # FIX: Slice the generation to *only* include new (non-prompt) tokens
            gen_tokens = gen[:, input_token_len:]

        # Decode only the new tokens
        outs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        preds.extend([out.strip() for out in outs])

    print("Generation complete. Saving predictions...")
    
    # Clean repo_id for filenames
    safe_repo_id = args.hub_repo_id.replace('/', '_')
    
    # Slice texts and refs to match the number of predictions
    num_preds = len(preds)
    sliced_texts = texts[:num_preds]
    sliced_refs = refs[:num_preds]
    
    # Save to JSONL
    jsonl_path = os.path.join(args.output, f"{safe_repo_id}_predictions.jsonl")
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
    csv_path = os.path.join(args.output, f"{safe_repo_id}_predictions.csv")
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
    
    # Compute and save metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(preds, sliced_refs, sliced_texts, include_summac=args.include_summac)
    
    out_path = os.path.join(args.output, f"{safe_repo_id}_metrics.txt")
    print(f"Saving metrics to {out_path}")
    
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print("\n--- Metrics ---")
    print(metrics)
    print("---------------")
    print(f"\nAll results saved to {args.output}/")
    print(f"  - Predictions (JSONL): {safe_repo_id}_predictions.jsonl")
    print(f"  - Predictions (CSV): {safe_repo_id}_predictions.csv")
    print(f"  - Metrics: {safe_repo_id}_metrics.txt")



if __name__ == "__main__":
    main()


