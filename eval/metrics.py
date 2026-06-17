"""
NLP metrics for ViPET-VLM evaluation (report generation + VQA tasks).

Computes the 4 metrics reported in the paper's results table:
    BLEU-4, ROUGE-1 (R-1), ROUGE-L (R-L), BERTScore (BERT)

Usage:
    from eval.metrics import compute_nlp_metrics

    predictions = ["Hinh anh PET/CT cho thay ...", ...]
    references  = ["Hinh anh PET/CT toan than cho thay ...", ...]

    scores = compute_nlp_metrics(predictions, references)
    # {"bleu4": 23.22, "rouge1": 57.61, "rougeL": 43.80, "bert": 77.06}

Notes:
    - Vietnamese tokenization: BLEU/ROUGE here use simple whitespace
      tokenization (each space-separated syllable as one token). This is
      the most common default when no Vietnamese word-segmenter (e.g.
      underthesea/pyvi) is specified. If the original paper used a
      word-segmenter, scores will not be directly comparable on the nose
      -- but should still be in the same ballpark and consistent for
      relative (ablation) comparisons within this project.
    - BERTScore uses lang="vi", which resolves to a multilingual BERT
      model under the hood (bert-score does not ship a Vietnamese-specific
      default). This is the standard fallback used by the bert-score
      library itself for languages without a dedicated default model.
    - All scores are returned as percentages (0-100), matching the paper's
      table format, not raw 0-1 fractions.
"""

from typing import Dict, List

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer. See module docstring for rationale."""
    return text.strip().split()


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    """
    Corpus-level BLEU-4 (4-gram, equal weights) with smoothing
    (method4, handles short/zero n-gram-match sentences gracefully --
    important here since some VQA answers are very short).
    Returns BLEU-4 score as a percentage (0-100).
    """
    hyps = [_tokenize(p) for p in predictions]
    refs = [[_tokenize(r)] for r in references]

    smoothing = SmoothingFunction().method4
    score = corpus_bleu(
        refs, hyps,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )
    return score * 100


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Average ROUGE-1 and ROUGE-L F-measure across all pairs.
    Returns {"rouge1": float, "rougeL": float} as percentages (0-100).
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    r1_scores, rL_scores = [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1_scores.append(scores["rouge1"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": (sum(r1_scores) / len(r1_scores)) * 100 if r1_scores else 0.0,
        "rougeL": (sum(rL_scores) / len(rL_scores)) * 100 if rL_scores else 0.0,
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "vi",
    batch_size: int = 32,
) -> float:
    """
    Average BERTScore F1 across all pairs.
    Returns BERTScore F1 as a percentage (0-100).
    """
    _, _, f1 = bert_score.score(
        predictions, references,
        lang=lang,
        batch_size=batch_size,
        verbose=False,
    )
    return f1.mean().item() * 100


def compute_nlp_metrics(
    predictions: List[str],
    references: List[str],
    bertscore_lang: str = "vi",
) -> Dict[str, float]:
    """
    Compute all 4 NLP metrics at once (BLEU-4, ROUGE-1, ROUGE-L, BERTScore).
    Returns dict with keys: "bleu4", "rouge1", "rougeL", "bert" -- all 0-100 scale.
    """
    assert len(predictions) == len(references), (
        f"predictions ({len(predictions)}) and references ({len(references)}) "
        f"must have the same length"
    )
    assert len(predictions) > 0, "predictions/references must not be empty"

    bleu4  = compute_bleu4(predictions, references)
    rouge  = compute_rouge(predictions, references)
    bert   = compute_bertscore(predictions, references, lang=bertscore_lang)

    return {
        "bleu4":  round(bleu4, 2),
        "rouge1": round(rouge["rouge1"], 2),
        "rougeL": round(rouge["rougeL"], 2),
        "bert":   round(bert, 2),
    }


def evaluate_predictions_file(
    predictions_path: str,
    pred_key: str = "generated",
    ref_key: str = "ground_truth",
) -> Dict[str, float]:
    """
    Load a JSON file of [{"prediction": ..., "reference": ...}, ...]
    and compute NLP metrics over it. Use pred_key/ref_key to match your
    actual predictions.json field names if they differ.
    """
    import json
    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predictions = [d[pred_key] for d in data]
    references  = [d[ref_key] for d in data]

    return compute_nlp_metrics(predictions, references)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute NLP metrics for a predictions JSON file")
    parser.add_argument("predictions_path", help="Path to predictions JSON file")
    parser.add_argument("--pred_key", default="generated", help="JSON key for generated text")
    parser.add_argument("--ref_key", default="ground_truth", help="JSON key for ground-truth text")
    args = parser.parse_args()

    scores = evaluate_predictions_file(args.predictions_path, args.pred_key, args.ref_key)
    print(json.dumps(scores, indent=2, ensure_ascii=False))
