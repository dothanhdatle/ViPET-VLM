import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

class BaseViPETDataset(Dataset):
    """Base class to read metadata and download from HuggingFace"""
    def __init__(self, metadata_path, repo_id="thainamhoang/ViMed-PET-CT", split="train"):
        self.repo_id = repo_id
        self.split = split
        
        if os.path.exists(metadata_path):
            self.df = pd.read_csv(metadata_path)
        else:
            csv_path = hf_hub_download(repo_id=repo_id, filename="metadata.csv", repo_type="dataset")
            self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def _load_report(self, report_path):
        local_path = hf_hub_download(repo_id=self.repo_id, filename=report_path, repo_type="dataset")
        with open(local_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_npz(self, npz_path):
        local_path = hf_hub_download(repo_id=self.repo_id, filename=npz_path, repo_type="dataset")
        return np.load(local_path)['data']


class ViPET2DDataset(BaseViPETDataset):
    """Dataset return 2D images for 2D vision encoder (using MIP to convert 3D to 2D)"""
    def __init__(self, metadata_path, repo_id="thainamhoang/ViMed-PET-CT", split="train", transform=None, mip_axis=1):
        super().__init__(metadata_path, repo_id, split)
        self.transform = transform
        self.mip_axis = mip_axis # 0: Sagittal, 1: Coronal, 2: Axial

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        pet_3d = self._load_npz(row['pet_path'])
        report_data = self._load_report(row['report_path'])
        
        # Convert 3D to 2D (MIP) based on config axis
        pet_2d = np.max(pet_3d, axis=self.mip_axis)
        
        if self.transform:
            pet_2d = self.transform(pet_2d)
        
        return {
            'image': pet_2d, 
            'report_dict': report_data,
            'patient_id': row['name']
        }


class ViPET3DDataset(BaseViPETDataset):
    """Dataset return 3D PET/CT for 3D vision encoder"""
    def __init__(self, metadata_path, repo_id="thainamhoang/ViMed-PET-CT", split="train", transform_3d=None):
        super().__init__(metadata_path, repo_id, split)
        self.transform_3d = transform_3d

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        pet_3d = self._load_npz(row['pet_path'])
        report_data = self._load_report(row['report_path'])
        
        if self.transform_3d:
            pet_3d = self.transform_3d(pet_3d)
            
        return {
            'volume': pet_3d, 
            'report_dict': report_data,
            'patient_id': row['name']
        }
