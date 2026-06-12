#!/bin/bash
# Download full ViMed-PET-CT dataset to /workspace/data
# Run after setup_vast.sh
# Estimated size: ~205GB, ~3-4 hours

echo "=== Downloading ViMed-PET-CT dataset ==="
echo "Estimated size: ~205GB"
echo ""

python - <<'EOF'
import os
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

SAVE_DIR = "/workspace/data"
REPO_ID  = "thainamhoang/ViMed-PET-CT"

# Load metadata
df = pd.read_csv(f"{SAVE_DIR}/metadata.csv")
print(f"Total patients: {len(df)}")

failed = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
    for col in ["ct_path", "pet_path", "report_path"]:
        local_path = os.path.join(SAVE_DIR, row[col])
        if os.path.exists(local_path):
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=row[col],
                repo_type="dataset",
                local_dir=SAVE_DIR,
            )
        except Exception as e:
            failed.append({"path": row[col], "error": str(e)})

print(f"\nDone! Failed: {len(failed)}")
if failed:
    import json
    with open(f"{SAVE_DIR}/download_failed.json", "w") as f:
        json.dump(failed, f, indent=2)
    print(f"Failed files saved to {SAVE_DIR}/download_failed.json")
EOF
