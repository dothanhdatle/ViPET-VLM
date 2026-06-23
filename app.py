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

REPORT_CONFIG = os.getenv("VIPET_REPORT_CONFIG", VQA_CONFIG)
REPORT_CHECKPOINT = os.getenv("VIPET_REPORT_CHECKPOINT", VQA_CHECKPOINT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_REPORT = (
    "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
    "Hãy viết báo cáo y tế chi tiết cho ảnh này.\n"
    "Báo cáo: "
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

    volume = np.squeeze(load_npz_volume(pet_path))

    if volume.ndim == 2:
        img = volume
    elif volume.ndim == 3:
        shape = volume.shape
        if shape[1] == shape[2]:
            axis = 0
        elif shape[0] == shape[1]:
            axis = 2
        elif shape[0] == shape[2]:
            axis = 1
        else:
            axis = int(np.argmax(shape))
        img = np.take(volume, shape[axis] // 2, axis=axis)
    else:
        raise ValueError(f"Unsupported PET shape: {volume.shape}")

    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img.astype(np.float32)


def transform_pet(pet_path: str) -> torch.Tensor:
    pet_transform = get_transform("ctvit", modality="pet")
    pet_raw = load_npz_volume(pet_path)
    return pet_transform(pet_raw).unsqueeze(0).to(DEVICE)


def clean_generated_text(text: str) -> str:
    text = text.strip()
    for marker in ["\nNgười dùng:", "\nTrợ lý:", "Người dùng:", "Trợ lý:"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    return text


@torch.no_grad()
def generate_with_prompt(model, pet_path: str, prompt: str, max_new_tokens: int) -> str:
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
        repetition_penalty=1.15,
    )

    return clean_generated_text(model.decode(output_ids)[0])


def build_multiturn_prompt(question: str, history: List[Tuple[str, str]]) -> str:
    parts = []

    for user_msg, assistant_msg in history or []:
        parts.append(f"Người dùng: {user_msg}\n")
        parts.append(f"Trợ lý: {assistant_msg}\n")

    parts.append(
        "Người dùng: Đây là ảnh PET/CT toàn thân của bệnh nhân. "
        f"{question}\n"
        "Trợ lý: "
    )
    return "".join(parts)


def on_pet_change(pet_path):
    if pet_path is None:
        return None, []
    try:
        return preview_pet_slice(pet_path), []
    except Exception as e:
        return None, [("System", f"Lỗi đọc PET: {e}")]


def run_report(pet_path):
    if pet_path is None:
        return "Vui lòng tải lên file PET (.npz)."

    try:
        model = get_model(REPORT_CONFIG, REPORT_CHECKPOINT)
        return generate_with_prompt(model, pet_path, PROMPT_REPORT, max_new_tokens=1024)
    except Exception as e:
        return f"Lỗi: {e}"


def run_vqa_chat(pet_path, question, history):
    history = history or []

    if pet_path is None:
        history.append(("System", "Vui lòng tải lên file PET (.npz)."))
        return history, ""

    if not question or not question.strip():
        return history, ""

    question = question.strip()

    try:
        model = get_model(VQA_CONFIG, VQA_CHECKPOINT)
        prompt = build_multiturn_prompt(question, history)
        answer = generate_with_prompt(model, pet_path, prompt, max_new_tokens=128)
    except Exception as e:
        answer = f"Lỗi: {e}"

    history.append((question, answer))
    return history, ""


with gr.Blocks(title="ViPET-VLM Demo") as demo:
    gr.Markdown("# ViPET-VLM PET-only Demo")
    gr.Markdown("Research demo for PET/CT report generation and multi-turn VQA.")

    with gr.Row():
        pet_file = gr.File(label="PET volume (.npz)", type="filepath")
        pet_preview = gr.Image(label="Middle PET slice preview", type="numpy", height=360)

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

    pet_file.change(on_pet_change, inputs=pet_file, outputs=[pet_preview, chatbot])
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
    demo.launch(share=True)