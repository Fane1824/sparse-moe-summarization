from typing import List, Dict, Any

import evaluate
import numpy as np
from bert_score import score as bertscore
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

FUNCTION_WORDS = set(stopwords.words('english'))


def compute_rouge_n(preds: List[str], refs: List[str], n_values: List[int] = [1, 2, 3, 4]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    
    rouge_types = [f'rouge{n}' for n in n_values]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores_dict = {rouge_type: [] for rouge_type in rouge_types}
    
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for rouge_type in rouge_types:
            scores_dict[rouge_type].append(scores[rouge_type].fmeasure)
            
    return {rouge_type: float(np.mean(scores_dict[rouge_type])) for rouge_type in rouge_types}


def compute_rouge(preds: List[str], refs: List[str]) -> Dict[str, float]:
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    rougeL_scores = []
    
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rougeL": float(np.mean(rougeL_scores)),
    }


def compute_bleu(preds: List[str], refs: List[str]) -> Dict[str, float]:
    bleu = evaluate.load("sacrebleu")
    result = bleu.compute(predictions=preds, references=[[r] for r in refs])
    return {"bleu": float(result.get("score", 0.0))}


def compute_bertscore(preds: List[str], refs: List[str]) -> Dict[str, float]:
    P, R, F1 = bertscore(preds, refs, lang="en", rescale_with_baseline=True)
    return {
        "bertscore_p": float(P.mean().item()),
        "bertscore_r": float(R.mean().item()),
        "bertscore_f1": float(F1.mean().item()),
    }


def compute_compression_ratio(preds: List[str], srcs: List[str]) -> Dict[str, float]:
    lens = []
    for p, s in zip(preds, srcs):
        lens.append(len(p.split()) / max(1, len(s.split())))
    return {"compression_ratio": float(np.mean(lens))}


def compute_extractiveness(preds: List[str], srcs: List[str]) -> Dict[str, float]:
    overlap_percentages = []
    for p, s in zip(preds, srcs):
        p_words = [w.lower() for w in p.split() if w.lower() not in FUNCTION_WORDS]
        s_words = set([w.lower() for w in s.split() if w.lower() not in FUNCTION_WORDS])
        if len(p_words) == 0:
            overlap_percentages.append(0.0)
        else:
            overlap = sum(1 for w in p_words if w in s_words) / len(p_words)
            overlap_percentages.append(overlap)
    return {"extractiveness": float(np.mean(overlap_percentages))}


def compute_summac(preds: List[str], srcs: List[str]) -> Dict[str, float]:
    # Optional (downloads weights on first use)
    try:
        from summaC import SummaCZS

        model = SummaCZS(granularity="sentence", model_name="mnli")
        scores = []
        for p, s in zip(preds, srcs):
            res = model.score([{"candidate": p, "document": s}])
            scores.append(res[0]["score"])
        return {"summac": float(np.mean(scores))}
    except Exception:
        return {"summac": float("nan")}


def compute_all_metrics(preds: List[str], refs: List[str], srcs: List[str], include_summac: bool = False) -> Dict[str, float]:
    results: Dict[str, float] = {}
    results.update(compute_rouge_n(preds, refs))
    results.update(compute_rouge(preds, refs))
    results.update(compute_bleu(preds, refs))
    results.update(compute_bertscore(preds, refs))
    results.update(compute_compression_ratio(preds, srcs))
    results.update(compute_extractiveness(preds, srcs))
    if include_summac:
        results.update(compute_summac(preds, srcs))
    return results

