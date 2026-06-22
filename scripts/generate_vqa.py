"""
Generate multi-turn VQA conversations from ViMed-PET reports using GPT.

Following ViMed-PET paper methodology (Appendix B.1):
    - System prompt instructs GPT to act as medical assistant
    - Few-shot examples guide format and tone
    - Output: multi-turn Q&A grounded in the clinical report

Usage:
    python scripts/generate_vqa.py \\
        --metadata /content/ViPET-data/metadata_subset.csv \\
        --local_data_dir /content/ViPET-data \\
        --output_path /content/ViPET-data/vqa_conversations.json \\
        --api_key sk-... \\
        --model gpt-5.4-mini \\
        --max_samples 12
"""

import os
import sys
import time
import json
import argparse
import pandas as pd
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import BaseViPETDataset, parse_report


# ── System prompt (translated from Figure 3, in Vietnamese) ──
SYSTEM_PROMPT = """Bạn là trợ lý y tế và được cung cấp thông tin liên quan đến một hình ảnh y khoa. \
Thông tin này dưới dạng một báo cáo lâm sàng ngắn, bao gồm vị trí của hình ảnh và một số phát hiện chẩn đoán ban đầu. \
Dựa trên đó, bạn cần trả lời các câu hỏi như thể bạn đang trực tiếp xem hình ảnh. \
Hãy tạo một đoạn hội thoại giữa bạn (đóng vai trợ lý y tế) và bệnh nhân, tập trung vào nội dung của hình ảnh. \
Cả câu hỏi và câu trả lời trong hội thoại phải phản ánh việc bạn đang quan sát trực tiếp hình ảnh. \
Các câu hỏi phải đa dạng, và câu trả lời của bạn phải dựa hoàn toàn vào thông tin có sẵn. \
Câu hỏi nên bao gồm nhiều khía cạnh của nội dung hình ảnh, bao gồm vị trí giải phẫu, kích thước hoặc đặc điểm của tổn thương, \
và các đặc điểm lâm sàng quan sát được khác. \
Chỉ đặt câu hỏi có thể trả lời chắc chắn, dựa trên thông tin trực tiếp có trong hình ảnh hoặc thông tin có thể suy luận rõ ràng. \
Không đặt câu hỏi không thể trả lời chắc chắn. \
Khi trả lời các câu hỏi phức tạp, hãy đưa ra câu trả lời chi tiết, có lý giải rõ ràng. \
Tránh đặt câu hỏi hoặc trả lời dựa trên thông tin mơ hồ, giả định hoặc không thể xác minh. \
Dưới đây là một ví dụ để bạn tham khảo."""


# ── Few-shot example (Vietnamese, based on Figure 4 structure) ──
FEWSHOT_INPUT = """Hình ảnh này chụp vùng lồng ngực của bệnh nhân. \
Tăng chuyển hóa FDG sinh lý được quan sát ở tim, phù hợp với hoạt động chuyển hóa bình thường. \
Không phát hiện tràn dịch màng phổi hai bên và không có tràn dịch màng tim. \
Vài hạch trung thất kích thước khoảng 10mm được ghi nhận ở vùng cạnh khí quản, dưới cung động mạch chủ và carina. \
Các hạch này không tăng chuyển hóa FDG. Có hình ảnh đông đặc dạng dải ở thùy giữa phải và kính mờ ở thùy dưới phải, \
cả hai đều không tăng chuyển hóa FDG, gợi ý lành tính. Vài hạch nách hai bên kích thước 10mm cũng không tăng chuyển hóa FDG."""

FEWSHOT_OUTPUT = json.dumps({
    "conversation": [
        {
            "question": "Hình ảnh này chụp vùng nào của cơ thể?",
            "answer": "Hình ảnh này chụp vùng lồng ngực của bệnh nhân."
        },
        {
            "question": "Có phát hiện hạch bất thường ở lồng ngực không?",
            "answer": "Có, có vài hạch trung thất kích thước khoảng 10mm ở vùng cạnh khí quản, dưới cung động mạch chủ và carina. Tuy nhiên, các hạch này không tăng chuyển hóa FDG, gợi ý lành tính."
        },
        {
            "question": "Có bất thường ở màng phổi hoặc màng tim không?",
            "answer": "Không, không phát hiện tràn dịch màng phổi hai bên và không có tràn dịch màng tim."
        },
        {
            "question": "Có tổn thương nhu mô phổi không?",
            "answer": "Có, có hình ảnh đông đặc dạng dải ở thùy giữa phải và kính mờ ở thùy dưới phải. Cả hai tổn thương đều không tăng chuyển hóa FDG, gợi ý đây là tổn thương lành tính."
        },
        {
            "question": "Có hạch bất thường ở vùng nách không?",
            "answer": "Có, vài hạch nách hai bên kích thước khoảng 10mm được ghi nhận, nhưng không tăng chuyển hóa FDG, không có dấu hiệu ác tính."
        },
    ]
}, ensure_ascii=False, indent=2)

def build_messages(report_text: str) -> list:
    """Build chat messages for GPT API call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEWSHOT_INPUT},
        {"role": "assistant", "content": FEWSHOT_OUTPUT},
        {
            "role": "user",
            "content": (
                "Dựa trên báo cáo PET/CT dưới đây, hãy tạo 8-14 cặp hỏi đáp bằng tiếng Việt. "
                "Câu hỏi và câu trả lời phải bám sát báo cáo, không thêm thông tin ngoài báo cáo. "
                "Chỉ trả về JSON object đúng định dạng sau:\n"
                '{"conversation": [{"question": "...", "answer": "..."}]}\n\n'
                f"Báo cáo:\n{report_text}"
            ),
        },
    ]


def generate_conversation(client, report_text: str, model: str, max_retries: int = 3) -> list:
    """
    Call GPT API to generate VQA conversation from a report.

    Returns:
        list of {"question": ..., "answer": ...} dicts
    """
    messages = build_messages(report_text)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            obj = json.loads(content)

            if isinstance(obj, list):
                conversation = obj
            elif isinstance(obj, dict):
                conversation = None
                for key in ["conversation", "conversations", "qa", "questions"]:
                    if key in obj:
                        conversation = obj[key]
                        break
                if conversation is None:
                    print(f"  WARNING: JSON object has no conversation key: {list(obj.keys())}")
                    return []
            else:
                print(f"  WARNING: Unexpected JSON type: {type(obj)}")
                return []

            cleaned = []
            for qa in conversation:
                if not isinstance(qa, dict):
                    continue

                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()

                if not question or not answer:
                    continue

                cleaned.append({
                    "question": question,
                    "answer": answer,
                })

            return cleaned

        except Exception as e:
            print(f"  WARNING: API/parse failed attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)

    return []

def to_llava_conversations(qa_pairs: list) -> list:
    conversations = []

    for i, qa in enumerate(qa_pairs):
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()

        if not question or not answer:
            continue

        if i == 0:
            question = "<image>\n" + question

        conversations.append({"from": "human", "value": question})
        conversations.append({"from": "gpt", "value": answer})

    return conversations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",       required=True)
    parser.add_argument("--local_data_dir", required=True)
    parser.add_argument("--output_path",    default="vqa_conversations.json")
    parser.add_argument("--api_key",        required=True)
    parser.add_argument("--model",          default="gpt-5.4-mini")
    parser.add_argument("--use_english",    action="store_true")
    parser.add_argument("--max_samples",    type=int, default=None)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # Load metadata + reports
    dataset = BaseViPETDataset(
        metadata_path=args.metadata,
        use_english=args.use_english,
        local_data_dir=args.local_data_dir,
    )

    df = dataset.df
    if args.max_samples:
        df = df.head(args.max_samples)

    results = []
    for i, row in df.iterrows():
        report = dataset._load_report(row)
        report_text = report["full_text"]

        if not report_text.strip():
            print(f"[{i+1}/{len(df)}] {row['name']} — empty report, skip")
            continue

        print(f"[{i+1}/{len(df)}] {row['name']} — generating...")
        conversation = generate_conversation(client, report_text, args.model)
            
        if conversation:
            results.append({
                "patient_id": row["name"],
                "report_path": row["report_path"],
                "pet_path": row["pet_path"],
                "report": report_text,
                "conversations": to_llava_conversations(conversation),
            })
            print(f"  -> {len(conversation)} QA pairs")
        else:
            print(f"  -> failed")

    # Save
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_qa = sum(len(r["conversation"]) for r in results)
    print(f"\nDone! {len(results)} conversations, {total_qa} QA pairs total.")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
