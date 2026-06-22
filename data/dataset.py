"""
ViMed-PET-CT Dataset
HuggingFace: thainamhoang/ViMed-PET-CT

Whole-body PET/CT volumes (.npz, key='data', shape=(D,H,W), dtype=int16)
Report JSON structure:
{
    "Nhận định kết quả": ...,   ← clinical impression
    "Mô tả hình ảnh": {
        "Đầu - cổ": ...,
        "Lồng ngực": ...,
        "Ổ bụng - khung chậu": ...,
        "Hệ cơ - xương": ...
    }
}

Split strategy (temporal, following paper):
    2017 (THANG 8-12):     train=8-10,  val=11,    test=12
    2018 (THANG 1-4,7-12): train=1-4,7-9, val=10,  test=11-12
    2019 (THANG 5,6,10-12):train=5,6,10, val=11,   test=12
    2023 (THANG 1-12):     train=1-9,   val=10,    test=11-12
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download


HF_REPO = "thainamhoang/ViMed-PET-CT"

# Report keys
IMPRESSION_KEY = "Nhận định kết quả"
FINDINGS_KEY   = "Mô tả hình ảnh"

# Temporal split config — theo paper ViMed-PET
# large data: 2018, 2023 | small data: 2017, 2019
SPLIT_CONFIG = {
    2017: {  # small, THANG 8-12
        "train": {"exclude": ["THANG 11", "THANG 12"]},
        "val":   {"include": ["THANG 11"]},
        "test":  {"include": ["THANG 12"]},
    },
    2018: {  # large, THANG 1-4, 7-12
        "train": {"exclude": ["THANG 10", "THANG 11", "THANG 12"]},
        "val":   {"include": ["THANG 10"]},
        "test":  {"include": ["THANG 11", "THANG 12"]},
    },
    2019: {  # small, THANG 5,6,10-12
        "train": {"exclude": ["THANG 11", "THANG 12"]},
        "val":   {"include": ["THANG 11"]},
        "test":  {"include": ["THANG 12"]},
    },
    2023: {  # large, whole year
        "train": {"exclude": ["THANG 10", "THANG 11", "THANG 12"]},
        "val":   {"include": ["THANG 10"]},
        "test":  {"include": ["THANG 11", "THANG 12"]},
    },
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract_month(pet_path: str) -> str:
    """Extract 'THANG X' từ path. Ví dụ: 'PETCT2023/THANG 10/PET/...' → 'THANG 10'"""
    match = re.search(r'(THANG \d+)', pet_path)
    return match.group(1) if match else ""


def parse_report(data: Dict) -> Dict[str, str]:
    """
    Parse report JSON thành structured dict.

    Returns dict với keys:
        full_text, impression, findings,
        head_neck, chest, abdomen, skeleton
    """
    impression    = data.get(IMPRESSION_KEY, "")
    findings_dict = data.get(FINDINGS_KEY, {})

    head_neck = findings_dict.get("Đầu - cổ", "")
    chest     = findings_dict.get("Lồng ngực", "")
    abdomen   = findings_dict.get("Ổ bụng - khung chậu", "")
    skeleton  = findings_dict.get("Hệ cơ - xương", "")

    findings_text = " ".join(filter(None, [head_neck, chest, abdomen, skeleton]))
    full_text     = " ".join(filter(None, [findings_text, impression]))

    return {
        "full_text":  full_text,
        "impression": impression,
        "findings":   findings_text,
        "head_neck":  head_neck,
        "chest":      chest,
        "abdomen":    abdomen,
        "skeleton":   skeleton,
    }


# ─────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────

class BaseViPETDataset(Dataset):
    """
    Base class: load metadata và handle HuggingFace download.

    Args:
        metadata_path: path tới metadata.csv local.
                       Nếu không tồn tại → tự download từ HuggingFace.
        repo_id:       HuggingFace dataset repo ID.
        use_english:   True → dùng reports_en/ thay vì reports/.
        cache_dir:     HuggingFace local cache directory.
    """

    def __init__(
        self,
        metadata_path: str,
        repo_id: str = HF_REPO,
        use_english: bool = False,
        cache_dir: Optional[str] = None,
        local_data_dir: Optional[str] = None,
    ):
        self.repo_id        = repo_id
        self.use_english    = use_english
        self.cache_dir      = cache_dir
        self.local_data_dir = local_data_dir

        if os.path.exists(metadata_path):
            self.df = pd.read_csv(metadata_path)
        else:
            print("Downloading metadata.csv from HuggingFace...")
            csv_path = hf_hub_download(
                repo_id=repo_id,
                filename="metadata.csv",
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            self.df = pd.read_csv(csv_path)

        print(f"Loaded {len(self.df)} patients")

    def __len__(self) -> int:
        return len(self.df)

    def _download(self, relative_path: str) -> str:
        """
        Load file từ local nếu có, không thì download từ HuggingFace.
        
        Args:
            relative_path: path tương đối trong dataset (e.g. "PETCT2017/THANG 8/PET/...")
        """
        # Check local trước
        if self.local_data_dir is not None:
            local_path = os.path.join(self.local_data_dir, relative_path)
            if os.path.exists(local_path):
                return local_path

        # Fallback về HuggingFace
        return hf_hub_download(
            repo_id=self.repo_id,
            filename=relative_path,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )

    def _load_npz(self, npz_path: str) -> Optional[np.ndarray]:
        """
        Load CT hoặc PET volume.
        key='data', shape=(D,H,W), dtype int16 → convert float32.
        """
        try:
            local_path = self._download(npz_path)
            return np.load(local_path)["data"].astype(np.float32)
        except Exception as e:
            print(f"Warning: Cannot load {npz_path}: {e}")
            return None

    def _load_report(self, row: pd.Series) -> Dict[str, str]:
        """
        Load và parse report JSON.
        Returns dict: full_text, impression, findings,
                      head_neck, chest, abdomen, skeleton
        """
        path_col = "report_en_path" if self.use_english else "report_path"
        try:
            local_path = self._download(row[path_col])
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return parse_report(data)
        except Exception as e:
            print(f"Warning: Cannot load report {row[path_col]}: {e}")
            return parse_report({})


# ─────────────────────────────────────────────
# Main dataset — 3D (CT-ViT, Cosmos)
# ─────────────────────────────────────────────

class ViPET3DDataset(BaseViPETDataset):
    """
    Main dataset cho ViMed-PET pipeline.
    Load whole-body CT + PET volume (3D) và structured report.

    Args:
        load_ct:   load CT volume (default True)
        load_pet:  load PET volume (default True)
        transform: optional transform áp lên numpy array
    """

    def __init__(
        self,
        metadata_path: str,
        repo_id: str = HF_REPO,
        use_english: bool = False,
        load_ct: bool = True,
        load_pet: bool = True,
        pet_transform=None,
        ct_transform=None,
        cache_dir: Optional[str] = None,
        local_data_dir: Optional[str] = None,
    ):
        super().__init__(metadata_path, repo_id, use_english, cache_dir, local_data_dir)
        self.load_ct       = load_ct
        self.load_pet      = load_pet
        self.pet_transform = pet_transform
        self.ct_transform  = ct_transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row    = self.df.iloc[idx]
        report = self._load_report(row)

        item = {
            "patient_id": row["name"],
            "report":     report,
        }

        if self.load_ct:
            ct = self._load_npz(row["ct_path"])
            item["ct"] = self.ct_transform(ct) if (self.ct_transform and ct is not None) else ct

        if self.load_pet:
            pet = self._load_npz(row["pet_path"])
            item["pet"] = self.pet_transform(pet) if (self.pet_transform and pet is not None) else pet

        return item


# ─────────────────────────────────────────────
# Experimental — 2D baseline via MIP
# ─────────────────────────────────────────────

class ViPET2DDataset(BaseViPETDataset):
    """
    Experimental dataset cho 2D vision encoder baseline (CLIP, ViT).
    Convert 3D PET → 2D bằng Maximum Intensity Projection (MIP).

    NOTE: Không dùng trong main pipeline. Chỉ để so sánh baseline.

    Args:
        mip_axis: 0=Sagittal, 1=Coronal, 2=Axial (default)
    """

    def __init__(
        self,
        metadata_path: str,
        repo_id: str = HF_REPO,
        use_english: bool = False,
        mip_axis: int = 2,
        transform=None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(metadata_path, repo_id, use_english, cache_dir)
        self.mip_axis  = mip_axis
        self.transform = transform

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row    = self.df.iloc[idx]
        report = self._load_report(row)
        pet_3d = self._load_npz(row["pet_path"])

        pet_2d = np.max(pet_3d, axis=self.mip_axis) if pet_3d is not None else None
        if self.transform and pet_2d is not None:
            pet_2d = self.transform(pet_2d)

        return {
            "image":      pet_2d,
            "report":     report,
            "patient_id": row["name"],
        }


# ─────────────────────────────────────────────
# Split — temporal strategy theo paper
# ─────────────────────────────────────────────

def split_metadata(
    df: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """
    Split metadata theo temporal strategy của paper ViMed-PET.

    Strategy per year:
        2017 (THANG 8-12):       train=8-10,    val=11,  test=12
        2018 (THANG 1-4, 7-12):  train=1-4,7-9, val=10,  test=11-12
        2019 (THANG 5,6,10-12):  train=5,6,10,  val=11,  test=12
        2023 (THANG 1-12):       train=1-9,     val=10,  test=11-12

    Args:
        df:    full metadata DataFrame (2757 rows)
        split: "train", "val", hoặc "test"

    Returns:
        Filtered DataFrame
    """
    assert split in ["train", "val", "test"], \
        f"split phải là train/val/test, got '{split}'"

    df = df.copy()
    df["_month"] = df["pet_path"].apply(extract_month)

    result_rows = []
    for _, row in df.iterrows():
        year  = int(row["year"]) if pd.notna(row.get("year")) else None
        month = row["_month"]

        if year is None or month == "" or year not in SPLIT_CONFIG:
            continue

        config = SPLIT_CONFIG[year][split]

        if "include" in config:
            if month in config["include"]:
                result_rows.append(row)
        elif "exclude" in config:
            if month not in config["exclude"]:
                result_rows.append(row)

    result_df = (pd.DataFrame(result_rows)
                   .drop(columns=["_month"])
                   .reset_index(drop=True))

    # Summary
    year_dist = result_df["year"].value_counts().sort_index().to_dict()
    print(f"Split '{split}': {len(result_df)} samples | years: {year_dist}")
    return result_df


# ─────────────────────────────────────────────
# VQA dataset — multi-turn conversations from GPT
# ─────────────────────────────────────────────

def _extract_qa_pairs_from_conversations(conversations: list) -> list:
    qa_pairs = []

    i = 0
    while i < len(conversations) - 1:
        human = conversations[i]
        gpt = conversations[i + 1]

        if human.get("from") == "human" and gpt.get("from") == "gpt":
            question = human.get("value", "").replace("<image>", "").strip()
            answer = gpt.get("value", "").strip()

            if question and answer:
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                })

            i += 2
        else:
            i += 1

    return qa_pairs

class ViPETVQADataset(BaseViPETDataset):
    """
    VQA dataset built from GPT-generated multi-turn conversations
    (output of scripts/generate_vqa.py).

    Each conversation is exploded into individual (image, question, answer)
    training samples — one PET/CT volume paired with each QA pair.

    Expected JSON format:
    [
        {
            "patient_id": "patient_0",
            "report_path": "...",
            "pet_path": "...",
            "report": "...",
            "conversations": [
                {"from": "human", "value": "<image>\n..."},
                {"from": "gpt", "value": "..."},
                ...
            ]
        }
    ]

    Args:
        vqa_path: path to VQA conversations JSON
        load_ct:  load CT volume (default True)
        load_pet: load PET volume (default True)
        transform: optional transform applied to numpy arrays
    """

    def __init__(
        self,
        metadata_path:  str,
        vqa_path:       str,
        repo_id:        str = HF_REPO,
        use_english:    bool = False,
        load_ct:        bool = True,
        load_pet:       bool = True,
        pet_transform=None,
        ct_transform=None,
        cache_dir:      Optional[str] = None,
        local_data_dir: Optional[str] = None,
        allowed_report_paths: Optional[set] = None,
    ):
        super().__init__(metadata_path, repo_id, use_english, cache_dir, local_data_dir)
        self.load_ct       = load_ct
        self.load_pet      = load_pet
        self.pet_transform = pet_transform
        self.ct_transform  = ct_transform

        # Load VQA conversations and explode into flat QA pairs
        with open(vqa_path, "r", encoding="utf-8") as f:
            vqa_records = json.load(f)
            
        self.qa_pairs = []
        self.conversation_items = []

        for conv in vqa_records:
            patient_id = conv["patient_id"]
            report_path = conv.get("report_path")

            if report_path:
                matches = self.df[self.df["report_path"] == report_path]
            else:
                matches = self.df[self.df["name"] == patient_id]

            if len(matches) == 0:
                print(
                    f"Warning: no match for patient_id='{patient_id}' "
                    f"report_path='{report_path}', skipping"
                )
                continue

            row = matches.iloc[0]
            multiturn = conv.get("conversations", [])
            matched_report_path = row["report_path"]
            if allowed_report_paths is not None and matched_report_path not in allowed_report_paths:
                continue

            if multiturn:
                self.conversation_items.append({
                    "patient_id": patient_id,
                    "report_path": row["report_path"],
                    "ct_path": row["ct_path"],
                    "pet_path": row["pet_path"],
                    "conversations": multiturn,
                })

            qa_list = _extract_qa_pairs_from_conversations(multiturn)

            for qa in qa_list:
                self.qa_pairs.append({
                    "patient_id": patient_id,
                    "report_path": row["report_path"],
                    "ct_path": row["ct_path"],
                    "pet_path": row["pet_path"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                })

        print(
            f"Loaded {len(self.qa_pairs)} QA pairs "
            f"from {len(self.conversation_items)} multi-turn conversations"
        )

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.qa_pairs[idx]

        result = {
            "patient_id": item["patient_id"],
            "question":   item["question"],
            "answer":     item["answer"],
        }

        if self.load_ct:
            ct = self._load_npz(item["ct_path"])
            result["ct"] = self.ct_transform(ct) if (self.ct_transform and ct is not None) else ct

        if self.load_pet:
            pet = self._load_npz(item["pet_path"])
            result["pet"] = self.pet_transform(pet) if (self.pet_transform and pet is not None) else pet

        return result
    
class ViPETMultiTurnVQADataset(ViPETVQADataset):
    """
    Multi-turn VQA dataset for Stage 3 instruction tuning.
    Uses the same VQA JSON as ViPETVQADataset, but returns full conversations.
    """

    def __len__(self) -> int:
        return len(self.conversation_items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.conversation_items[idx]

        result = {
            "patient_id": item["patient_id"],
            "conversations": item["conversations"],
        }

        if self.load_ct:
            ct = self._load_npz(item["ct_path"])
            result["ct"] = self.ct_transform(ct) if (self.ct_transform and ct is not None) else ct

        if self.load_pet:
            pet = self._load_npz(item["pet_path"])
            result["pet"] = self.pet_transform(pet) if (self.pet_transform and pet is not None) else pet

        return result

class MixedStage2Dataset(Dataset):
    """
    Combines full-report samples (a ViPET3DDataset, restricted to the
    train split) with a sample of single-turn QA pairs
    into ONE dataset for Stage 2 training.
    """

    QA_PROMPT = (
        "Đây là ảnh PET/CT toàn thân của bệnh nhân. "
        "{question}\n"
        "Trả lời: "
    )

    def __init__(self, report_dataset, qa_dataset, report_prompt: str,
                 qa_per_study: int = 2, seed: int = 42):
        """
        Args:
            report_dataset: a ViPET3DDataset instance, already restricted
                             to the train split (dataset.df already set)
            qa_dataset:      a ViPETVQADataset instance built from a
                             TRAIN-ONLY vqa json
            report_prompt:   the fixed prompt used for report samples
                             (pass Stage2Trainer.PROMPT)
            qa_per_study:  how many QA pairs to sample per study/report.
        """
        self.report_dataset = report_dataset
        self.qa_dataset      = qa_dataset
        self.report_prompt   = report_prompt

        rng = random.Random(seed)

        by_study = {}
        for qa in qa_dataset.qa_pairs:
            by_study.setdefault(qa["report_path"], []).append(qa)

        self.qa_items = []
        for _, items in by_study.items():
            self.qa_items.extend(rng.sample(items, min(qa_per_study, len(items))))

        print(
            f"MixedStage2Dataset: {len(report_dataset)} report samples + "
            f"{len(self.qa_items)} QA samples "
            f"(from {len(by_study)} studies, {qa_per_study}/study max)"
        )

    def __len__(self):
        return len(self.report_dataset) + len(self.qa_items)

    def __getitem__(self, idx):
        if idx < len(self.report_dataset):
            item = self.report_dataset[idx]
            return {
                "pet": item["pet"],
                "patient_id": item["patient_id"],
                "prompt": self.report_prompt,
                "target": item["report"]["full_text"],
            }

        qa = self.qa_items[idx - len(self.report_dataset)]
        pet = self.qa_dataset._load_npz(qa["pet_path"])
        #ct  = self.qa_dataset._load_npz(qa["ct_path"])
        if self.qa_dataset.pet_transform and pet is not None:
            pet = self.qa_dataset.pet_transform(pet)
        #if self.qa_dataset.ct_transform and ct is not None:
            #ct = self.qa_dataset.ct_transform(ct)

        return {
            "pet": pet,
            "patient_id": qa["patient_id"],
            "prompt": self.QA_PROMPT.format(question=qa["question"]),
            "target": qa["answer"],
        }
    
