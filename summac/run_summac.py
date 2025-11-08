#!/usr/bin/env python3
"""
Script to evaluate summaries using SummaC (Summary Consistency) metric.
SummaC measures the factual consistency between source documents and summaries.
"""

import argparse
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
from summac.model_summac import SummaCZS, SummaCConv
from tqdm import tqdm
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK data (punkt_tab)...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading required NLTK data (punkt)...")
    nltk.download('punkt', quiet=True)


def load_data(input_file):
    """
    Load data from CSV or JSONL file.
    
    Args:
        input_file: Path to the input file (CSV or JSONL)
        
    Returns:
        List of dictionaries with 'source' and 'prediction' keys
    """
    input_path = Path(input_file)
    
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_file)
        data = df.to_dict('records')
    elif input_path.suffix == '.jsonl':
        # Try pandas first, fall back to manual parsing if it fails
        try:
            df = pd.read_json(input_file, lines=True, encoding='utf-8')
            data = df.to_dict('records')
        except (ValueError, Exception) as e:
            print(f"Warning: pandas failed to load JSONL ({e}), using manual parsing...")
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as json_err:
                        print(f"Warning: Skipping line {line_num} due to JSON error: {json_err}")
                        continue
            print(f"Successfully loaded {len(data)} records manually")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .csv or .jsonl")
    
    return data


def evaluate_summac(data, model_type='conv', batch_size=8, device='cuda'):
    """
    Evaluate summaries using SummaC.
    
    Args:
        data: List of dictionaries with source and prediction keys
        model_type: 'zs' for zero-shot or 'conv' for convolutional model
        batch_size: Batch size for processing
        device: 'cuda' or 'cpu'
        
    Returns:
        List of SummaC scores
    """
    print(f"Initializing SummaC model (type: {model_type}, device: {device})...")
    
    if model_type == 'zs':
        model = SummaCZS(granularity="sentence", model_name="vitc", device=device)
    elif model_type == 'conv':
        model = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", 
                          nli_labels="e", device=device, start_file="default", 
                          agg="mean")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'zs' or 'conv'")
    
    # Handle different column name formats
    # Format 1: source, prediction, reference (baseline_bart)
    # Format 2: source_text, generated_summary, reference_summary (eval_hub)
    if 'source' in data[0]:
        sources = [item['source'] for item in data]
        predictions = [item['prediction'] for item in data]
    elif 'source_text' in data[0]:
        sources = [item['source_text'] for item in data]
        predictions = [item['generated_summary'] for item in data]
    else:
        raise ValueError(f"Unknown data format. Expected 'source'/'prediction' or 'source_text'/'generated_summary' keys")
    
    print(f"Evaluating {len(data)} samples...")
    
    # Process in batches to show progress
    all_scores = []
    for i in tqdm(range(0, len(sources), batch_size), desc="Processing batches"):
        batch_sources = sources[i:i+batch_size]
        batch_predictions = predictions[i:i+batch_size]
        
        scores = model.score(batch_sources, batch_predictions)
        all_scores.extend(scores['scores'])
    
    return all_scores


def append_to_metrics(metrics_file, summac_score, model_type):
    """
    Append SummaC score to the metrics file.
    
    Args:
        metrics_file: Path to the metrics.txt file
        summac_score: The computed SummaC score
        model_type: Type of SummaC model used
    """
    metrics_path = Path(metrics_file)
    
    # Create the file if it doesn't exist
    if not metrics_path.exists():
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.touch()
    
    with open(metrics_file, 'a', encoding='utf-8') as f:
        f.write(f"summac_{model_type}: {summac_score}\n")
    
    print(f"\nSummaC score appended to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate summaries using SummaC metric"
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Path to input file (CSV or JSONL with source and prediction columns)'
    )
    parser.add_argument(
        '--metrics-file',
        type=str,
        help='Path to metrics.txt file to append results (default: same directory as input)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['zs', 'conv'],
        default='conv',
        help='SummaC model type: "zs" (zero-shot) or "conv" (convolutional, default)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use: "cuda" or "cpu" (default: cuda)'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        help='GPU device number to use (e.g., 0, 1, 2). Sets CUDA_VISIBLE_DEVICES. Only applicable when --device is cuda.'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (default: all samples)'
    )
    parser.add_argument(
        '--save-scores',
        action='store_true',
        help='Save individual scores to a file'
    )
    
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if GPU number is specified
    if args.gpu is not None and args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")
    
    # Determine metrics file path
    if args.metrics_file:
        metrics_file = args.metrics_file
    else:
        input_path = Path(args.input_file)
        metrics_file = input_path.parent / 'metrics.txt'
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)
    print(f"Loaded {len(data)} samples")
    
    # Limit samples if max_samples is specified
    if args.max_samples is not None and args.max_samples < len(data):
        data = data[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Evaluate with SummaC
    scores = evaluate_summac(
        data,
        model_type=args.model_type,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Calculate average score
    avg_score = np.mean(scores)
    print(f"\n{'='*50}")
    print(f"SummaC ({args.model_type}) Average Score: {avg_score:.6f}")
    print(f"{'='*50}")
    
    # Append to metrics file
    append_to_metrics(metrics_file, avg_score, args.model_type)
    
    # Optionally save individual scores
    if args.save_scores:
        input_path = Path(args.input_file)
        scores_file = input_path.parent / f'summac_{args.model_type}_scores.json'
        
        scores_data = [
            {
                'id': data[i].get('id', i),
                'summac_score': scores[i]
            }
            for i in range(len(scores))
        ]
        
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, indent=2)
        
        print(f"Individual scores saved to {scores_file}")


if __name__ == '__main__':
    main()
