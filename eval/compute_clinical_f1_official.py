"""
Compute Clinical F1 against the OFFICIAL ViMed-PET medical_test_set
(paper's hand-verified ~80-patient ground truth), restricted to whichever
of those patients overlap with OUR OWN test split -- for a true
apples-to-apples comparison with the paper's published Table 4 numbers.

Ground truth for these patients comes DIRECTLY from the official JSON
files (already physician-verified, already in the exact structured format
extract_value_gpt.py would otherwise produce via GPT-4o) -- no API call
needed for the ground-truth side, only for our model's generated text.

Matches patients by NUMERIC PATIENT ID extracted via regex from both
filenames/patient_id strings (robust to the different naming conventions
between our report_path format and the official filename format).

Usage:
    python eval/compute_clinical_f1_official.py \
        --predictions_file predictions_full_v2.json \
        --official_gt_dir ./medical_test_set_official \
        --output_file clinical_f1_official_overlap.json
"""
import os
import re
import json
import argparse

from extract_value_gpt import extract_value
from compute_clinical_f1 import compute_clinical_f1


# Paper's Table 4 GPT-4o baseline (few-shot), computed on the FULL official
# 80-patient set -- printed alongside our result for reference. Not a
# perfect comparison (our overlap subset is smaller, and is our MODEL not
# GPT-4o), but same patients/GT/formula, so the closest apples-to-apples
# context available.
PAPER_GPT4O_TABLE4 = {"F1-T": 24.21, "F1-TP": 13.62, "F1-TF": 20.57, "F1-TPF": 7.87}


def extract_patient_num(name: str):
    """Pull the trailing numeric patient id out of any naming convention
    used in this project, e.g. 'patient_83' -> '83', or a filename like
    'PETCT_2017_THANG 12_29_patient_83_REPORT_patient_83.json' -> '83'."""
    match = re.search(r"patient_(\d+)", name)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Clinical F1 on the overlap between our test split and "
                    "the paper's official hand-verified medical_test_set."
    )
    parser.add_argument("--predictions_file", required=True,
                        help="predictions_full_v2.json (output of inference/generate.py)")
    parser.add_argument("--official_gt_dir", required=True,
                        help="Local dir containing the downloaded medical_test_set/*.json files")
    parser.add_argument("--output_file", default="clinical_f1_official_overlap.json")
    args = parser.parse_args()

    with open(args.predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    pred_by_num = {}
    for item in predictions:
        num = extract_patient_num(item["patient_id"])
        if num:
            pred_by_num[num] = item

    official_files = [f for f in os.listdir(args.official_gt_dir) if f.endswith(".json")]
    official_by_num = {}
    for fname in official_files:
        num = extract_patient_num(fname)
        if num:
            official_by_num[num] = fname

    overlap_nums = sorted(set(pred_by_num) & set(official_by_num), key=int)
    print(f"Official medical_test_set: {len(official_by_num)} patients found in --official_gt_dir")
    print(f"Our predictions file:      {len(pred_by_num)} patients")
    print(f"Overlap (apples-to-apples): {len(overlap_nums)} patients\n")

    if not overlap_nums:
        print("No overlap found -- check patient_id naming or --official_gt_dir path.")
        return

    extracted_data = []
    for i, num in enumerate(overlap_nums):
        pred_item = pred_by_num[num]
        gt_path = os.path.join(args.official_gt_dir, official_by_num[num])
        with open(gt_path, "r", encoding="utf-8") as f:
            official_gt = json.load(f)

        print(f"[{i+1}/{len(overlap_nums)}] Extracting generated report for "
              f"patient_{num}...")
        extracted_generated = extract_value(pred_item["generated"])

        extracted_data.append({
            "patient_id":             pred_item["patient_id"],
            "extracted_generated":    extracted_generated,
            "extracted_ground_truth": official_gt,
        })

    scores = compute_clinical_f1(extracted_data)

    print(f"\nKết quả trên {len(overlap_nums)}-patient overlap (model của bạn):")
    print(f"{'Metric':<10} {'Precision':>10} {'Recall':>10} {'F1':>8}   (correct/gt/pred)")
    for name, s in scores.items():
        print(f"{name:<10} {s['precision']:>9.2f}% {s['recall']:>9.2f}% {s['f1']:>7.2f}%   "
              f"({s['correct']}/{s['total_gt']}/{s['total_pred']})")

    print(f"\nThe context: paper Table 4 GPT-4o baseline (few-shot, FULL 80-patient set):")
    print(f"{'Metric':<10} {'F1':>8}")
    for name, val in PAPER_GPT4O_TABLE4.items():
        print(f"{name:<10} {val:>7.2f}%")
    print(f"\n(Lưu ý: N khác nhau -- {len(overlap_nums)} patient overlap vs 80 patient gốc "
          f"-- và đây là model của bạn so với GPT-4o, không phải model chính của paper. "
          f"Vẫn là so sánh công bằng nhất có thể: cùng patient, cùng ground-truth verify tay, "
          f"cùng công thức F1.)")

    output = {
        "overlap_count": len(overlap_nums),
        "overlap_patient_ids": [pred_by_num[n]["patient_id"] for n in overlap_nums],
        "scores": scores,
        "paper_gpt4o_table4_for_reference": PAPER_GPT4O_TABLE4,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.output_file}")


if __name__ == "__main__":
    main()
