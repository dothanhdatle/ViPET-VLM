import json
from collections import defaultdict

from rouge_score import rouge_scorer


REPORT_PATH = (
    "/workspace/evaluation/predictions/"
    "predictions_report_structured_test.json"
)
VQA_PATH = (
    "/workspace/evaluation/predictions/"
    "predictions_vqa_test.json"
)
OUTPUT_PATH = "/workspace/evaluation/demo_candidates.json"

TOP_K = 20


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    reports = load_json(REPORT_PATH)
    vqa_items = load_json(VQA_PATH)

    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=False,
    )

    # Tránh chọn patient có nhiều study vì prediction chưa lưu report_path.
    reports_by_patient = defaultdict(list)
    for item in reports:
        reports_by_patient[item["patient_id"]].append(item)

    vqa_by_patient = defaultdict(list)
    for item in vqa_items:
        vqa_by_patient[item["patient_id"]].append(item)

    candidates = []

    for patient_id, patient_reports in reports_by_patient.items():
        if len(patient_reports) != 1:
            continue

        if patient_id not in vqa_by_patient:
            continue

        report = patient_reports[0]
        report_score = scorer.score(
            report["ground_truth"],
            report["generated"],
        )["rougeL"].fmeasure

        qa_results = []
        for qa in vqa_by_patient[patient_id]:
            score = scorer.score(
                qa["ground_truth"],
                qa["generated"],
            )["rougeL"].fmeasure

            qa_results.append({
                "question": qa["question"],
                "generated": qa["generated"],
                "ground_truth": qa["ground_truth"],
                "rougeL": round(score, 4),
            })

        qa_scores = [x["rougeL"] for x in qa_results]
        mean_vqa = sum(qa_scores) / len(qa_scores)
        worst_vqa = min(qa_scores)

        # Ưu tiên cả chất lượng trung bình lẫn câu trả lời yếu nhất.
        combined_score = (
            0.40 * report_score
            + 0.40 * mean_vqa
            + 0.20 * worst_vqa
        )

        qa_results.sort(key=lambda x: x["rougeL"], reverse=True)

        candidates.append({
            "patient_id": patient_id,
            "combined_score": round(combined_score, 4),
            "report_rougeL": round(report_score, 4),
            "mean_vqa_rougeL": round(mean_vqa, 4),
            "worst_vqa_rougeL": round(worst_vqa, 4),
            "num_questions": len(qa_results),
            "report_generated": report["generated"],
            "report_ground_truth": report["ground_truth"],
            "vqa": qa_results,
        })

    candidates.sort(
        key=lambda x: x["combined_score"],
        reverse=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            candidates[:TOP_K],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "rank patient combined report  mean_vqa worst_vqa questions"
    )
    for rank, item in enumerate(candidates[:TOP_K], start=1):
        print(
            f"{rank:>4} {item['patient_id']:<16} "
            f"{item['combined_score']:.4f}   "
            f"{item['report_rougeL']:.4f}   "
            f"{item['mean_vqa_rougeL']:.4f}   "
            f"{item['worst_vqa_rougeL']:.4f}   "
            f"{item['num_questions']}"
        )

    print(f"\nSaved detailed candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()