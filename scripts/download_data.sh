#!/bin/bash
# Download PET-only ViMed-PET-CT data to /workspace/data
# Run after setup_vast.sh

set -e

echo "=== Downloading ViMed-PET-CT PET-only dataset ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
python - <<'EOF'
import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

SAVE_DIR = "/workspace/data"
REPO_ID  = "thainamhoang/ViMed-PET-CT"

os.makedirs(SAVE_DIR, exist_ok=True)

# Use full metadata committed in repo
df = pd.read_csv("/workspace/ViPET-VLM/data/metadata.csv")
print(f"Total patients: {len(df)}")

# Copy metadata to training data dir
df.to_csv(f"{SAVE_DIR}/metadata.csv", index=False)

# PET-only reproduction: no CT needed
cols_to_download = ["pet_path", "report_path"]

failed = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Patients"):
    for col in cols_to_download:
        rel_path = row[col]
        local_path = os.path.join(SAVE_DIR, rel_path)

        if os.path.exists(local_path):
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=rel_path,
                repo_type="dataset",
                local_dir=SAVE_DIR,
            )
        except Exception as e:
            failed.append({
                "path": rel_path,
                "error": str(e),
            })

print(f"\nDone! Failed: {len(failed)}")

if failed:
    failed_path = f"{SAVE_DIR}/download_failed.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, indent=2, ensure_ascii=False)
    print(f"Failed files saved to {failed_path}")
else:
    print("All PET/report files downloaded successfully!")
EOF

echo ""
echo "=== Download complete ==="
du -sh /workspace/data