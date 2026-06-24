import os
import sys
from typing import List, Tuple

import gradio as gr
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

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

    pet_coronal = volume[:, volume.shape[1] // 2, :]

    fig, ax = plt.subplots(figsize=(3, 5))
    ax.imshow(
        pet_coronal,
        cmap="gray_r",
        aspect="auto",
    )
    ax.axis("off")
    fig.tight_layout()

    return fig


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


APP_CSS = """
:root {
    --app-bg: #f4f7f8;
    --panel-bg: #ffffff;
    --text-main: #14252b;
    --text-muted: #52666d;
    --border: #d7e0e3;
    --primary: #087f75;
    --primary-hover: #06685f;
}

.gradio-container {
    width: 100% !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    box-sizing: border-box !important;
    background: var(--app-bg) !important;
    color: var(--text-main) !important;
}

.main {
    background: var(--app-bg) !important;
}

#app-header {
    margin-bottom: 12px !important;
}

#app-header h1 {
    margin-bottom: 4px !important;
    color: var(--text-main) !important;
}

#app-header p {
    color: var(--text-muted) !important;
}

#main-workspace {
    display: grid !important;
    grid-template-columns: minmax(0, 3fr) minmax(320px, 2fr) !important;
    width: 100% !important;
    gap: 16px !important;
    align-items: stretch !important;
}

#function-panel,
#image-panel {
    width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    background: var(--panel-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 14px !important;
}

#pet-upload,
#pet-preview,
#report-output,
#vqa-chat {
    background: var(--panel-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

#pet-upload {
    min-height: 72px !important;
}

#pet-preview {
    height: 520px !important;
    overflow: hidden !important;
}

#pet-preview .plot-container {
    height: 475px !important;
}

#report-output textarea {
    min-height: 470px !important;
    color: var(--text-main) !important;
    line-height: 1.55 !important;
}

#vqa-chat {
    height: 430px !important;
}

button.primary {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #ffffff !important;
}

button.primary:hover {
    background: var(--primary-hover) !important;
    border-color: var(--primary-hover) !important;
}

.tabs button.selected {
    color: var(--primary) !important;
    border-color: var(--primary) !important;
}

footer {
    display: none !important;
}

@media (max-width: 720px) {
    .gradio-container {
        padding: 12px !important;
    }

    #main-workspace {
        grid-template-columns: 1fr !important;
    }

    #pet-preview {
        height: 440px !important;
    }

    #pet-preview .plot-container {
        height: 395px !important;
    }

    #report-output textarea,
    #vqa-chat {
        min-height: 380px !important;
        height: 380px !important;
    }
}
"""


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Inter"),
        "Arial",
        "sans-serif",
    ],
)


with gr.Blocks(
    title="ViPET-VLM Demo",
    theme=theme,
    css=APP_CSS,
) as demo:
    with gr.Column(elem_id="app-header"):
        gr.Markdown("# ViPET-VLM PET-only Demo")
        gr.Markdown(
            "Sinh báo cáo và hỏi đáp từ ảnh PET toàn thân 3D."
        )

    with gr.Row(
        equal_height=True,
        elem_id="main-workspace",
    ):
        # Cột trái: sinh báo cáo và VQA.
        with gr.Column(elem_id="function-panel"):
            with gr.Tabs():
                with gr.Tab("Sinh báo cáo"):
                    report_output = gr.Textbox(
                        label="Báo cáo sinh ra",
                        lines=18,
                        elem_id="report-output",
                    )

                    report_btn = gr.Button(
                        "Sinh báo cáo",
                        variant="primary",
                    )

                with gr.Tab("Hỏi đáp PET"):
                    chatbot = gr.Chatbot(
                        label="Hội thoại VQA",
                        height=430,
                        elem_id="vqa-chat",
                    )

                    vqa_question = gr.Textbox(
                        label="Câu hỏi",
                        placeholder=(
                            "Ví dụ: Có phát hiện tăng chuyển hóa "
                            "FDG bất thường không?"
                        ),
                    )

                    with gr.Row():
                        vqa_btn = gr.Button(
                            "Trả lời",
                            variant="primary",
                        )
                        clear_btn = gr.Button("Xóa hội thoại")

        # Cột phải: upload và preview PET.
        with gr.Column(elem_id="image-panel"):
            pet_file = gr.File(
                label="PET volume (.npz)",
                type="filepath",
                elem_id="pet-upload",
            )

            pet_preview = gr.Plot(
                label="PET coronal slice preview",
                elem_id="pet-preview",
            )

    pet_file.change(
        on_pet_change,
        inputs=pet_file,
        outputs=[pet_preview, chatbot, report_output],
    )

    report_btn.click(
        run_report,
        inputs=pet_file,
        outputs=report_output,
    )

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

    clear_btn.click(
        lambda: [],
        outputs=chatbot,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        share=True,
    )