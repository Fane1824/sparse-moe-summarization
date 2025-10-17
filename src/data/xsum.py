import os
import re
from typing import Dict, Any, Tuple
from pathlib import Path

from datasets import Dataset, DatasetDict


def parse_summary_file(file_path: str) -> Dict[str, str]:
    """
    Parse a .summary file from the local XSum dataset.
    Returns dict with 'document' and 'summary' keys.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract sections using regex
    url_match = re.search(r'\[SN\]URL\[SN\]\s*(.+?)(?=\[SN\]|$)', content, re.DOTALL)
    title_match = re.search(r'\[SN\]TITLE\[SN\]\s*(.+?)(?=\[SN\]|$)', content, re.DOTALL)
    first_sentence_match = re.search(r'\[SN\]FIRST-SENTENCE\[SN\]\s*(.+?)(?=\[SN\]|$)', content, re.DOTALL)
    rest_body_match = re.search(r'\[SN\]RESTBODY\[SN\]\s*(.+?)(?=\[SN\]|$)', content, re.DOTALL)
    
    # Extract text (strip whitespace)
    title = title_match.group(1).strip() if title_match else ""
    first_sentence = first_sentence_match.group(1).strip() if first_sentence_match else ""
    rest_body = rest_body_match.group(1).strip() if rest_body_match else ""
    
    # Summary is the first sentence
    summary = first_sentence
    
    # --- THIS IS THE FIX ---
    # Document is title + rest body (NO first_sentence)
    document_parts = []
    if title:
        document_parts.append(title)
    # We remove this line:
    # if first_sentence:
    #     document_parts.append(first_sentence)
    if rest_body:
        document_parts.append(rest_body)
    
    document = " ".join(document_parts).strip()
    # --- END OF FIX ---

    # Handle cases where document might be empty after fix (if no title/rest_body)
    if not document:
        # Fallback to just rest_body if title is also missing
        # If rest_body is also empty, this will be an empty string, which is fine
        # We must avoid using first_sentence
        document = rest_body
        
    return {
        'document': document,
        'summary': summary,
        'id': os.path.basename(file_path).replace('.summary', '')
    }


def load_xsum_from_local(data_dir: str = "bbc-summary-data") -> DatasetDict:
    """
    Load XSum dataset from local bbc-summary-data directory.
    Returns a DatasetDict with train/validation/test splits.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    # Find all .summary files
    summary_files = list(data_path.glob("*.summary"))
    print(f"Found {len(summary_files)} summary files")
    
    if len(summary_files) == 0:
        raise ValueError(f"No .summary files found in {data_dir}")
    
    # Parse all files
    documents = []
    summaries = []
    ids = []
    
    empty_docs = 0
    for file_path in summary_files:
        try:
            parsed = parse_summary_file(str(file_path))
            # Skip examples where the document is empty after parsing
            if not parsed['document'] or not parsed['summary']:
                empty_docs += 1
                continue

            documents.append(parsed['document'])
            summaries.append(parsed['summary'])
            ids.append(parsed['id'])
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            continue
    
    if empty_docs > 0:
        print(f"Warning: Skipped {empty_docs} files with empty document or summary.")
    print(f"Successfully parsed {len(documents)} files")
    
    # Create dataset
    dataset = Dataset.from_dict({
        'id': ids,
        'document': documents,
        'summary': summaries,
    })
    
    # Create splits (80% train, 10% validation, 10% test)
    # Note: This is a simple split. For exact XSum splits, you'd need split files
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = dataset['test'].train_test_split(test_size=0.5, seed=42)
    
    dataset_dict = DatasetDict({
        'train': dataset['train'],
        'validation': test_val['train'],
        'test': test_val['test'],
    })
    
    print(f"Dataset splits - Train: {len(dataset_dict['train'])}, "
          f"Validation: {len(dataset_dict['validation'])}, "
          f"Test: {len(dataset_dict['test'])}")
    
    return dataset_dict


def load_xsum(dataset_name: str, text_column: str, summary_column: str):
    """
    Load XSum dataset. Tries local directory first, then falls back to HuggingFace.
    """
    # Check if local directory exists (try multiple possible locations)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    local_data_dir = project_root / "bbc-summary-data"
    
    # Also try current working directory
    if not local_data_dir.exists():
        cwd_data_dir = Path("bbc-summary-data").resolve()
        if cwd_data_dir.exists():
            local_data_dir = cwd_data_dir
    
    if local_data_dir.exists():
        print(f"Loading dataset from local directory: {local_data_dir}")
        dataset = load_xsum_from_local(str(local_data_dir))
        return dataset, text_column, summary_column
    else:
        # Fallback to HuggingFace loading (original code)
        print(f"Local dataset not found, loading from HuggingFace: {dataset_name}")
        from datasets import load_dataset as hf_load_dataset
        
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        cache_dir = os.path.join(hf_home, "datasets")
        
        try:
            dataset = hf_load_dataset(
                dataset_name,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            raise RuntimeError(
                f"Could not load dataset. Local directory not found at {local_data_dir} "
                f"and HuggingFace loading failed."
            ) from e
        
        return dataset, text_column, summary_column


def build_preprocess_fn(tokenizer, cfg: Dict[str, Any]):
    max_source_length = int(cfg["max_source_length"]) if cfg.get("max_source_length") else 768
    max_target_length = int(cfg["max_target_length"]) if cfg.get("max_target_length") else 64
    pad_to_max = bool(cfg.get("pad_to_max_length", False))

    padding = "max_length" if pad_to_max else False

    def preprocess(examples: Dict[str, Any]) -> Dict[str, Any]:
        inputs = examples[cfg["text_column"]]
        targets = examples[cfg["summary_column"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def get_splits(dataset, seed: int = 42):
    # XSum already provides train/validation/test
    return dataset["train"], dataset["validation"], dataset["test"]