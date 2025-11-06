# Sparse mixture-of-experts for text summarization

An Advanced NLP project studying extreme summarization on XSum. The goal is to compare pretrained summarizers with a sparse mixture-of-experts Transformer trained from scratch.

## Setup

Use Python 3.10 or newer and run commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pip install datasets PyYAML seaborn tqdm
```

Experiment settings live in `config/config.yaml`.

## BART baseline

```bash
python scripts/baseline_bart.py --output runs/baseline_bart
```

Evaluation reports ROUGE, BLEU, BERTScore, compression, and source overlap.

## Fine-tuning

```bash
python scripts/finetune_t5.py --output runs/t5_base
python scripts/finetune_qwen.py --output runs/llama_1b
```

The causal-model script uses Llama-3.2-1B-Instruct through the `finetune.qwen_3b` configuration.

## Sparse MoE

```bash
python scripts/train_moe.py --router topk --epochs 10 --output runs/moe_topk
python scripts/infer_moe.py --checkpoint runs/moe_topk/best_model --output runs/moe_eval
```

Use `--router hash` for single-expert hash routing. GQA and low-rank expert variants are available through training flags.
