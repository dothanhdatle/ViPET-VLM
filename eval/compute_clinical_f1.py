"""
Compute aggregate Clinical F1 scores (F1-T, F1-TP, F1-TF, F1-TPF) across
a whole test set, from the output of extract_value_gpt.py.

Reuses PredictionEvaluator (already-tested matching/mapping logic from
the original paper's repo) by writing each patient's lesion lists to
small temp files and calling it per-patient, then micro-averaging
(summing correct/total_gt/total_pred across ALL patients before
computing precision/recall/F1 -- not averaging per-patient F1s).

Usage:
    python eval/compute_clinical_f1.py --input_file extracted_values.json
    python eval/compute_clinical_f1.py --input_file extracted_values.json \
        --output_file clinical_f1_27_official.json \
        --label "27-patient official medical_test_set"
"""

import os
import json
import tempfile
import argparse

from evaluate_predictions import PredictionEvaluator


# Field combinations matching the paper's 4 Clinical F1 metrics.
# "hinh_dang" = lesion Type (category names like "hạch"/"nốt mờ"/"khối mờ"
#               match the paper's Type categories exactly, despite the
#               literal Vietnamese meaning "shape" -- verified against
#               categories_ver1.py's hinh_dang_ton_thuong dict).
# "vi_tri"    = Position, "tang_chuyen_hoa" = FDG uptake.
METRIC_FIELDS = {
    "F1-T":   ["hinh_dang"],
    "F1-TP":  ["hinh_dang", "vi_tri"],
    "F1-TF":  ["hinh_dang", "tang_chuyen_hoa"],
    "F1-TPF": ["hinh_dang", "vi_tri", "tang_chuyen_hoa"],
}


def evaluate_one_patient(generated_lesions: list, ground_truth_lesions: list, fields: list) -> dict:
    """
    Run PredictionEvaluator for ONE patient's lesion lists, for a given
    field combination. Writes to temp files since PredictionEvaluator
    only reads from disk paths.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as gt_f:
        json.dump(ground_truth_lesions, gt_f, ensure_ascii=False)
        gt_path = gt_f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as pred_f:
        json.dump(generated_lesions, pred_f, ensure_ascii=False)
        pred_path = pred_f.name

    try:
        evaluator = PredictionEvaluator(gt_file=gt_path, pred_file=pred_path, fields=fields)
        result = evaluator.evaluate()
    finally:
        os.remove(gt_path)
        os.remove(pred_path)

    return result


def compute_clinical_f1(extracted_data: list) -> dict:
    """
    Loop over all patients, accumulate correct/total_gt/total_pred per
    metric (micro-average), then compute final precision/recall/F1.

    Args:
        extracted_data: list of {"patient_id", "extracted_generated",
                                  "extracted_ground_truth"} dicts
                         (output of extract_value_gpt.py)

    Returns:
        dict: {metric_name: {"precision":..., "recall":..., "f1":...,
                              "correct":..., "total_gt":..., "total_pred":...}}
    """
    totals = {name: {"correct": 0, "total_gt": 0, "total_pred": 0} for name in METRIC_FIELDS}

    for item in extracted_data:
        generated    = item.get("extracted_generated") or []
        ground_truth = item.get("extracted_ground_truth") or []

        for metric_name, fields in METRIC_FIELDS.items():
            result = evaluate_one_patient(generated, ground_truth, fields)
            totals[metric_name]["correct"]    += result["correct_predictions"]
            totals[metric_name]["total_gt"]   += result["total_gt"]
            totals[metric_name]["total_pred"] += result["total_pred"]

    final_scores = {}
    for metric_name, t in totals.items():
        precision = t["correct"] / t["total_pred"] if t["total_pred"] > 0 else 0.0
        recall    = t["correct"] / t["total_gt"]   if t["total_gt"]   > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        final_scores[metric_name] = {
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
            "f1":        round(f1 * 100, 2),
            "correct":    t["correct"],
            "total_gt":   t["total_gt"],
            "total_pred": t["total_pred"],
        }

    return final_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute aggregate Clinical F1 (T/TP/TF/TPF) over a test set")
    parser.add_argument("--input_file", required=True,
                        help="Path to extracted_values.json (output of extract_value_gpt.py)")
    parser.add_argument("--output_file", default=None,
                        help="Optional path to save the computed scores as JSON")
    parser.add_argument("--label", default=None,
                        help="Optional label to print with results (e.g. '202-patient test set')")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    print(f"Computing Clinical F1 over {len(extracted_data)} patients"
          + (f" [{args.label}]" if args.label else "") + "...\n")

    scores = compute_clinical_f1(extracted_data)

    print(f"{'Metric':<10} {'Precision':>10} {'Recall':>10} {'F1':>8}   (correct/gt/pred)")
    for name, s in scores.items():
        print(f"{name:<10} {s['precision']:>9.2f}% {s['recall']:>9.2f}% {s['f1']:>7.2f}%   "
              f"({s['correct']}/{s['total_gt']}/{s['total_pred']})")

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"\nSaved -> {args.output_file}")
