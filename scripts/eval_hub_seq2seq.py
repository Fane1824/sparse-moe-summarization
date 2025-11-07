import os
import sys
import argparse
import json  # Added for jsonl output
import csv   # Added for csv output
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi
from tqdm import tqdm

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.metrics.compute import compute_all_metrics


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def get_latest_checkpoint(repo_id: str) -> tuple:
    """Find the latest checkpoint subdirectory in the Hub repo.
    
    Returns:
        tuple: (repo_id, subfolder) where subfolder is None if no checkpoints found
    """
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
            # No checkpoints found, return base repo
            print("No checkpoints found. Using base repo.")
            return repo_id, None
        
        # Sort by checkpoint number and get the latest
        sorted_checkpoints = sorted(
            checkpoints,
            key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0
        )
        latest = sorted_checkpoints[-1]
        print(f"Found latest checkpoint: {latest}")
        return repo_id, latest
    except Exception as e:
        print(f"Error finding checkpoints: {e}. Using base repo.")
        return repo_id, None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a seq2seq model from HF Hub on XSum (local or HF)")
    parser.add_argument("--hub_repo_id", required=True, help="HF Hub repo id, e.g. username/t5-base-xsum")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/eval_hub_seq2seq")
    parser.add_argument("--include_summac", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)

    xsum_cfg = cfg["xsum"].copy()
    gen_cfg = cfg["generation"]
    
    # Clean repo_id for filenames
    safe_repo_id = args.hub_repo_id.replace('/', '_')

    # Find and load the latest checkpoint
    repo_id, subfolder = get_latest_checkpoint(args.hub_repo_id)
    
    # Load model and tokenizer
    if subfolder:
        print(f"\n{'='*80}")
        print(f"� Loading model from: {repo_id}/{subfolder}")
        print(f"{'='*80}\n")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder=subfolder)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, subfolder=subfolder)
    else:
        print(f"\n{'='*80}")
        print(f"📦 Loading model from: {repo_id}")
        print(f"{'='*80}\n")
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset, text_col, summary_col = load_xsum(xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"])
    test_ds = dataset["test"]

    texts = test_ds[text_col]
    refs = test_ds[summary_col]
    preds: List[str] = []

    bs = 128
    batches = chunk_list(texts, bs)
    for batch in tqdm(batches, desc="Generating summaries"):
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
                num_beams=int(gen_cfg["num_beams"]),
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                length_penalty=float(gen_cfg["length_penalty"]),
                no_repeat_ngram_size=int(gen_cfg["no_repeat_ngram_size"]),
            )
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))

    print("Generation complete. Saving predictions...")
    
    # --- NEW: Save predictions to JSONL and CSV ---
    num_preds = len(preds)
    # Slice texts and refs to match the number of predictions
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
    # --- End of new code ---

    # Compute and save metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(preds, sliced_refs, sliced_texts, include_summac=args.include_summac)
    
    out_path = os.path.join(args.output, f"{safe_repo_id}_metrics.txt")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    
    print("Final metrics:")
    print(metrics)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()