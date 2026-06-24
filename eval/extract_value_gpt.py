import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=120.0,
    max_retries=5,
)
EXTRACTION_MODEL = "gpt-4o-mini"

def extract_value(prompt):
    system_prompt = f"""
Hãy trả lời những thông tin chính xác và đầy đủ, chi tiết từ report nhưng không được tự nghĩ ra câu trả lời.
Chỉ trả về một list chứa những json object như yêu cầu, không được phép có bất kể kí tự gì không liên quan.
Giả sử bạn là một AI chuyên trích xuất thông tin từ những report y tế. Hãy làm theo các bước sau đây:
1. Tìm và đọc những đoạn thông tin liên quan đến những câu hỏi:
    - Có khối u, hạch, tổn thương hay bất thường (kén khí, nốt mờ nhỏ, ...) ở phổi hoặc di căn đến phổi không?
    - Kích thước của tổn thương, khối u, hạch, bất thường?
    - Hình dạng của tổn thương, khối u, hạch, bất thường?
    - Mức độ FDG?
    - SUVmax?
    - Tổn thương (khối u), bất thường có tăng chuyển hóa FDG hay không? (Có tăng, không tăng, tăng cao hoặc tăng ít, nếu không có thông tin thì điền là Không có)
    - Vị trí của tổn thương (khối u), bất thường?
    - Có xâm lấn hoặc dính vào hay không? Xâm lấn đi đâu? (xung quanh, thành ngực, mạch máu, ...)?
    - Giai đoạn của tổn thương (khối u), bất thường?
        + U nguyên phát
        + Di căn hạch
        + Di căn xa
2. Từ những đoạn thông tin tìm thấy, trích xuất ra những thông tin quan trọng (Thông tin nào không có thì hãy ghi là 'Không có'). Hãy trả ra một list, mỗi phần tử là 1 json object theo format sau:
    {{
    'Kích thước khối u, tổn thương, bất thường': ...,
    'Hình dạng của khối u, tổn thương, bất thường': ...,
    'Vị trí của khối u, tổn thương, bất thường': ...,
    'Mức độ FDG': {{
        'SUVmax': ...,
        'Tăng chuyển hoá FDG': ...
    }},
    'Xâm lấn': ... (Nếu không có thì để là Không có)
    'Giai đoạn chuyển hoá': ...
    }}
Dưới đây là ví dụ về cách thực hiện:
Ví dụ 1:
Input:
Tổn thương:
- Hình ảnh khối mờ bờ tua gai ở hạ phân thùy I thùy trên phổi phải kích thước 74 x 56 mm, tăng chuyển hóa FDG (SUVmax: 14,9).

Output:
[
    {{
        "Kích thước khối u, tổn thương, bất thường": "74 x 56 mm",
        "Hình dạng của khối u, tổn thương, bất thường": "Hình ảnh khối mờ bờ tua gai",
        "Vị trí của khối u, tổn thương, bất thường": "hạ phân thùy I thùy trên phổi phải",
        "Mức độ FDG": {{
            "SUVmax": "14.9",
            "Tăng chuyển hoá FDG": "Có tăng"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }}
]

Ví dụ 2:
Input:
Các tổn thương:
Gan:
- Tại phân thùy IV có hình ảnh nốt giảm tỷ trọng đường kính 10mm, không tăng chuyển hóa FDG, theo dõi nang gan.
Phổi:
- Có dãn phế nang 2 bên đỉnh phổi kèm vôi hóa dạng nốt nhỏ rải rác. SUVmax: 2.5
- Có hạch trước khí quản đoạn cao, đoạn thấp, kích thước 12 x 10 mm (SUVmax: 3.3).

Output:
[
    {{
        "Kích thước khối u, tổn thương, bất thường": "Không có",
        "Hình dạng của khối u, tổn thương, bất thường": "Nốt vôi hóa",
        "Vị trí của khối u, tổn thương, bất thường": "đỉnh phổi hai bên",
        "Mức độ FDG": {{
            "SUVmax": "2.5",
            "Tăng chuyển hoá FDG": "Không có"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }},
    {{
        "Kích thước khối u, tổn thương, bất thường": "12 x 10 mm",
        "Hình dạng của khối u, tổn thương, bất thường": "Hình ảnh hạch",
        "Vị trí của khối u, tổn thương, bất thường": "Trước khí quản đoạn cao và đoạn thấp",
        "Mức độ FDG": {{
            "SUVmax": "3.3",
            "Tăng chuyển hoá FDG": "Không có"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }}
]
"""

    PROMPT_MESSAGES = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    completion = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        messages=PROMPT_MESSAGES,
        max_tokens=1000,
        temperature=0,
    )
    response_text = completion.choices[0].message.content

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Could not parse GPT response as JSON, raw text:\n{response_text}")
        return []

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract structured clinical attributes from a predictions.json "
                     "file (output of inference/generate.py) using GPT-4o-mini, for "
                     "both the generated and ground_truth report text of each patient."
    )
    parser.add_argument("--input_file", type=str, default="predictions.json",
                        help="Path to predictions.json (list of {patient_id, generated, ground_truth})")
    parser.add_argument("--output_file", type=str, default="extracted_values.json",
                        help="Path to write the combined extracted-values JSON")
    parser.add_argument("--patient_ids", type=str, default=None,
                        help="Optional path to a .txt file with one patient_id per line "
                             "to filter which records get processed (e.g. for the "
                             "27-patient official medical_test_set subset)")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.patient_ids:
        with open(args.patient_ids) as f:
            keep_ids = set(line.strip() for line in f if line.strip())
        data = [d for d in data if d["patient_id"] in keep_ids]
        print(f"Filtered to {len(data)} records matching {args.patient_ids}")

    results = []
    for i, item in enumerate(data):
        print(f"[{i+1}/{len(data)}] Extracting patient_id={item['patient_id']}...")
        extracted_generated    = extract_value(item["generated"])
        extracted_ground_truth = extract_value(item["ground_truth"])
        results.append({
            "patient_id":             item["patient_id"],
            "extracted_generated":    extracted_generated,
            "extracted_ground_truth": extracted_ground_truth,
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} extracted records -> {args.output_file}")