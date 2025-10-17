import os
import sys
import argparse

# Add project root to path
# Assumes this script is saved in the same directory as train.py/evaluate.py
# If you save it elsewhere, you might need to adjust this path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.utils.config import load_config, load_env
    from src.data.xsum import load_xsum
except ImportError:
    print("Error: Could not import required functions.")
    print("Make sure 'src/utils/config.py' and 'src/data/xsum.py' exist")
    print("And that this script is run from a location that can see the 'src' directory.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Check XSum data loading for leaks")
    parser.add_argument("--config", default="../config/config.yaml", help="Path to config file")
    parser.add_argument("--env", default=None, help="Path to .env file")
    args = parser.parse_args()

    try:
        load_env(args.env)
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file '{args.config}': {e}")
        print("Please ensure the path is correct.")
        return

    if "xsum" not in cfg:
        print(f"Error: 'xsum' section not found in '{args.config}'")
        return
        
    xsum_cfg = cfg["xsum"].copy()
    text_col_name = xsum_cfg.get("text_column")
    summary_col_name = xsum_cfg.get("summary_column")

    if not text_col_name or not summary_col_name:
        print("Error: 'text_column' or 'summary_column' not set in config's 'xsum' section.")
        return

    print(f"--- Loading Data from Config ---")
    print(f"Config file: {args.config}")
    print(f"Text column:     '{text_col_name}'")
    print(f"Summary column:  '{summary_col_name}'")
    print("---------------------------------\n")

    if text_col_name == summary_col_name:
        print("!!! CRITICAL ERROR: DATA LEAKAGE !!!")
        print(f"Your 'text_column' and 'summary_column' are both set to '{text_col_name}'.")
        print("This is the cause of your perfect scores.")
        print("Please edit your config file to point 'text_column' to the source document (e.g., 'document').")
        print("---------------------------------")
        return

    print("Data columns seem correct. Loading dataset sample...")
    
    try:
        dataset, text_col, summary_col = load_xsum(
            xsum_cfg["dataset_name"], text_col_name, summary_col_name
        )
    except Exception as e:
        print(f"Error loading dataset '{xsum_cfg['dataset_name']}': {e}")
        return

    print("\n--- First Example from 'test' split ---")
    
    try:
        sample = dataset["test"][0]
        
        print("\n=== SOURCE TEXT (from column '{}') ===\n".format(text_col))
        print(sample[text_col])
        
        print("\n=== REFERENCE SUMMARY (from column '{}') ===\n".format(summary_col))
        print(sample[summary_col])
        
        print("\n---------------------------------")
        print("Check passed: Source text and reference summary are different.")
        print("Your data loading appears to be correct.")
        
    except Exception as e:
        print(f"\nError accessing data columns: {e}")
        print(f"Make sure columns '{text_col}' and '{summary_col}' exist in the dataset.")


if __name__ == "__main__":
    main()