"""
Gradio demo app for ViPET-VLM — supports both report generation and VQA
via tabs. Upload PET/CT .npz files (or point to existing test-set files
for a safe, known-good demo).

Usage:
    python app.py

Then open the printed local/public URL in a browser.
"""

import os
import sys
import yaml
import torch
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.vlms.vipet_vlm import build_model
from inference.generate import load_checkpoint, predict_single


# !!! VERIFY these 4 paths actually exist on the new Vast.ai instance
# before relying on this for the demo -- a wrong path crashes on first click.
REPORT_CONFIG     = "configs/experiments/stage3_lora.yaml"
REPORT_CHECKPOINT = "/workspace/checkpoints/stage3/stage3_best.pt"

VQA_CONFIG     = "configs/experiments/stage3_vqa_lora.yaml"
VQA_CHECKPOINT = "/workspace/checkpoints/stage3_vqa/stage3_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_cache = {"report": None, "vqa": None}


def get_model_for_task(task: str):
    if _model_cache[task] is not None:
        return _model_cache[task]

    config_path     = REPORT_CONFIG if task == "report" else VQA_CONFIG
    checkpoint_path = REPORT_CHECKPOINT if task == "report" else VQA_CHECKPOINT

    with open(config_path) as f:
        config = yaml.safe_load(f)
    config["model"]["use_lora"] = True

    print(f"[{task}] Building model from {config_path}...")
    model = build_model(config, DEVICE)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    _model_cache[task] = model
    print(f"[{task}] Model ready.")
    return model


def run_report(pet_path, ct_path):
    if pet_path is None or ct_path is None:
        return "Vui lòng tải lên cả file PET (.npz) và CT (.npz)."
    model = get_model_for_task("report")
    try:
        result = predict_single(
            model, DEVICE,
            pet_path=pet_path, ct_path=ct_path,
            task="report",
        )
        return result
    except Exception as e:
        return f"Lỗi: {e}"


def run_vqa(pet_path, ct_path, question):
    if pet_path is None or ct_path is None:
        return "Vui lòng tải lên cả file PET (.npz) và CT (.npz)."
    if not question or not question.strip():
        return "Vui lòng nhập câu hỏi."
    model = get_model_for_task("vqa")
    try:
        result = predict_single(
            model, DEVICE,
            pet_path=pet_path, ct_path=ct_path,
            task="vqa", question=question.strip(),
        )
        return result
    except Exception as e:
        return f"Lỗi: {e}"


with gr.Blocks(title="ViPET-VLM Demo") as demo:
    gr.Markdown("# ViPET-VLM — Demo sinh báo cáo & hỏi-đáp PET/CT")
    gr.Markdown(
        "Tải lên file PET và CT (.npz) của một bệnh nhân. "
    )

    with gr.Tab("Sinh báo cáo"):
        with gr.Row():
            report_pet = gr.File(label="File PET (.npz)", type="filepath")
            report_ct  = gr.File(label="File CT (.npz)",  type="filepath")
        report_btn    = gr.Button("Sinh báo cáo", variant="primary")
        report_output = gr.Textbox(label="Báo cáo sinh ra", lines=12)
        report_btn.click(run_report, inputs=[report_pet, report_ct], outputs=report_output)

    with gr.Tab("Hỏi - Đáp (VQA)"):
        with gr.Row():
            vqa_pet = gr.File(label="File PET (.npz)", type="filepath")
            vqa_ct  = gr.File(label="File CT (.npz)",  type="filepath")
        vqa_question = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Có phát hiện khối u không?")
        vqa_btn       = gr.Button("Trả lời", variant="primary")
        vqa_output    = gr.Textbox(label="Trả lời", lines=6)
        vqa_btn.click(run_vqa, inputs=[vqa_pet, vqa_ct, vqa_question], outputs=vqa_output)


if __name__ == "__main__":
    demo.launch(share=True)