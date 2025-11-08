import os
import sys
import argparse
import json
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.metrics.compute import compute_all_metrics


def read_jsonl(path: str, source_key: str, reference_key: str, prediction_key: str):
    sources: List[str] = []
    references: List[str] = []
    predictions: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sources.append(obj[source_key])
            references.append(obj[reference_key])
            predictions.append(obj[prediction_key])
    return sources, references, predictions


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from a JSONL of predictions")
    parser.add_argument("--jsonl", required=True, help="Path to predictions.jsonl with fields: source, reference, prediction")
    parser.add_argument("--output", default=None, help="Optional output directory to save metrics.txt")
    parser.add_argument("--include_summac", action="store_true", help="Compute SummaC factuality (optional, may require separate install)")
    parser.add_argument("--source_key", default="source")
    parser.add_argument("--reference_key", default="reference")
    parser.add_argument("--prediction_key", default="prediction")
    args = parser.parse_args()

    sources, references, predictions = read_jsonl(
        args.jsonl, args.source_key, args.reference_key, args.prediction_key
    )

    metrics = compute_all_metrics(predictions, references, sources, include_summac=args.include_summac)

    # Print
    print({k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})

    # Optionally save
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_path = os.path.join(args.output, "metrics.txt")
        with open(out_path, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()


