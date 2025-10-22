import os
import sys
import argparse
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.config import load_config, load_env
from src.data.xsum import load_xsum
from src.metrics.compute import compute_all_metrics


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--env", default=None)
    parser.add_argument("--output", default="runs/baseline_bart")
    parser.add_argument("--include_summac", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    load_env(args.env)
    cfg = load_config(args.config)

    bart_cfg = cfg["bart"]
    xsum_cfg = cfg["xsum"].copy()

    dataset, text_col, summary_col = load_xsum(
        xsum_cfg["dataset_name"], xsum_cfg["text_column"], xsum_cfg["summary_column"]
    )
    test_ds = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(bart_cfg["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(bart_cfg["model_name"]).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    texts = test_ds[text_col]
    refs = test_ds[summary_col]
    preds: List[str] = []

    device = model.device
    bs = int(bart_cfg.get("batch_size", 8))
    for batch in chunk_list(texts, bs):
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

    # Save predictions (JSONL and CSV)
    jsonl_path = os.path.join(args.output, "predictions.jsonl")
    csv_path = os.path.join(args.output, "predictions.csv")
    try:
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for i, (src, ref, pred) in enumerate(zip(texts, refs, preds)):
                obj = {"id": i, "source": src, "reference": ref, "prediction": pred}
                jf.write(__import__("json").dumps(obj, ensure_ascii=False) + "\n")
        import csv
        with open(csv_path, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["id", "source", "reference", "prediction"])
            for i, (src, ref, pred) in enumerate(zip(texts, refs, preds)):
                writer.writerow([i, src, ref, pred])
    except Exception as e:
        print(f"Warning: failed to write prediction files: {e}")

    # Compute and save metrics
    metrics = compute_all_metrics(preds, refs[: len(preds)], texts[: len(preds)], include_summac=args.include_summac)

    out_path = os.path.join(args.output, "metrics.txt")
    with open(out_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(metrics)


if __name__ == "__main__":
    main()

