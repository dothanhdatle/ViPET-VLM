import os
import sys
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import get_transform
from inference.generate import load_checkpoint
from models.vlms.vipet_vlm import build_model


VQA_CONFIG = os.getenv("VIPET_VQA_CONFIG", "configs/experiments/stage3_vqa_multi_lora.yaml")
VQA_CHECKPOINT = os.getenv("VIPET_VQA_CHECKPOINT", "/workspace/checkpoints/stage3_vqa_multiturn/stage3_best.pt")

REPORT_CONFIG = os.getenv("VIPET_REPORT_CONFIG","configs/experiments/stage3_report_lora.yaml")
REPORT_CHECKPOINT = os.getenv("VIPET_REPORT_CHECKPOINT", "/workspace/checkpoints/stage3_report_structured/stage3_best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_REPORT = (
    "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "Hãy viết báo cáo PET/CT bằng tiếng Việt theo đúng cấu trúc sau: "
    "Đầu - cổ, Lồng ngực, Ổ bụng - khung chậu, "
    "Hệ cơ - xương, Kết luận.\n"
    "Báo cáo:\n"
)

_model_cache = {}


def load_npz_volume(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    key = "data" if "data" in data.files else data.files[0]
    return data[key].astype(np.float32)


def get_model(config_path: str, checkpoint_path: str):
    cache_key = (config_path, checkpoint_path)
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model"]["use_lora"] = True

    print(f"Loading model from {config_path} on {DEVICE}...")
    model = build_model(config, DEVICE)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    _model_cache[cache_key] = model
    return model


def preview_pet_slice(pet_path):
    if pet_path is None:
        return None

    volume = load_npz_volume(pet_path)

    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]

    if volume.ndim != 3:
        raise ValueError(f"Expected PET volume 3D, got {volume.shape}")

    # Coronal maximum-intensity projection
    img = np.max(volume, axis=1)

    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)

    return (img * 255).astype(np.uint8)


def transform_pet(pet_path: str) -> torch.Tensor:
    pet_transform = get_transform("ctvit", modality="pet")
    pet_raw = load_npz_volume(pet_path)
    return pet_transform(pet_raw).unsqueeze(0).to(DEVICE)


def clean_generated_text(text: str) -> str:
    text = text.strip()

    if text.startswith("Trợ lý:"):
        text = text[len("Trợ lý:"):].strip()

    for marker in ["\nNgười dùng:", "\nUser:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    return text


@torch.inference_mode()
def generate_with_prompt(
    model,
    pet_path: str,
    prompt: str,
    max_new_tokens: int,
    repetition_penalty: float = 1.15,
) -> str:
    pet = transform_pet(pet_path)

    encoded = model.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(DEVICE)

    output_ids = model.generate(
        pet=pet,
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
    )

    return clean_generated_text(model.decode(output_ids)[0])


def build_multiturn_prompt(
    question: str,
    history: List[Tuple[str, str]],
    max_history_turns: int = 5,
) -> str:
    parts = []

    for user_msg, assistant_msg in (history or [])[-max_history_turns:]:
        parts.append(f"Người dùng: {user_msg.strip()}\n")
        parts.append(f"Trợ lý: {assistant_msg.strip()}\n")

    parts.append(f"Người dùng: {question.strip()}\n")
    parts.append("Trợ lý: ")

    return "".join(parts)


def on_pet_change(pet_path):
    if pet_path is None:
        return None, [], ""

    try:
        return preview_pet_slice(pet_path), [], ""
    except Exception as e:
        gr.Warning(f"Không thể đọc PET: {e}")
        return None, [], ""


def run_report(pet_path):
    if pet_path is None:
        return "Vui lòng tải lên file PET (.npz)."

    try:
        model = get_model(REPORT_CONFIG, REPORT_CHECKPOINT)
        return generate_with_prompt(model, pet_path, PROMPT_REPORT, max_new_tokens=1024)
    except Exception as e:
        return f"Lỗi: {e}"


def run_vqa_chat(pet_path, question, history):
    history = list(history or [])

    if pet_path is None:
        gr.Warning("Vui lòng tải lên file PET (.npz).")
        return history, question

    if not question or not question.strip():
        return history, ""

    question = question.strip()

    try:
        model = get_model(VQA_CONFIG, VQA_CHECKPOINT)
        prompt = build_multiturn_prompt(question, history)
        answer = generate_with_prompt(
            model,
            pet_path,
            prompt,
            max_new_tokens=128,
            repetition_penalty=1.15,
        )
    except Exception as e:
        answer = f"Lỗi: {e}"

    history.append((question, answer))
    return history, ""


with gr.Blocks(title="ViPET-VLM Demo") as demo:
    gr.Markdown("# ViPET-VLM PET-only Demo")
    gr.Markdown("Research demo for PET/CT report generation and multi-turn VQA.")

    with gr.Row():
        pet_file = gr.File(label="PET volume (.npz)", type="filepath")
        pet_preview = gr.Image(label="PET coronal MIP preview", type="numpy", height=360)

    with gr.Tabs():
        with gr.Tab("Sinh báo cáo"):
            report_btn = gr.Button("Sinh báo cáo", variant="primary")
            report_output = gr.Textbox(label="Báo cáo sinh ra", lines=14)

        with gr.Tab("Hỏi đáp PET"):
            chatbot = gr.Chatbot(label="Hội thoại VQA", height=420)
            vqa_question = gr.Textbox(
                label="Câu hỏi",
                placeholder="Ví dụ: Có phát hiện tăng chuyển hóa FDG bất thường không?",
            )
            with gr.Row():
                vqa_btn = gr.Button("Trả lời", variant="primary")
                clear_btn = gr.Button("Xóa hội thoại")

    pet_file.change(on_pet_change, inputs=pet_file, outputs=[pet_preview, chatbot, report_output])
    report_btn.click(run_report, inputs=pet_file, outputs=report_output)

    vqa_btn.click(
        run_vqa_chat,
        inputs=[pet_file, vqa_question, chatbot],
        outputs=[chatbot, vqa_question],
    )
    vqa_question.submit(
        run_vqa_chat,
        inputs=[pet_file, vqa_question, chatbot],
        outputs=[chatbot, vqa_question],
    )
    clear_btn.click(lambda: [], outputs=chatbot)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", share=True)