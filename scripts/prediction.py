#!/usr/bin/env python3

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import load

class CNN3D(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((2,2,2))
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.avgpool(x)
        x = self.flatten(x)
        return x

def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    OUTPUTS = ROOT / "outputs-test"
    MODELS = ROOT / "models"

    rf_model = load(MODELS / "rf_model.joblib")
    print(f"Loaded RandomForest model from {MODELS / 'rf_model.joblib'}")

    seq_feats = {}
    for pt_file in (OUTPUTS / "1D_embeddings").glob("*.pt"):
        obj = torch.load(pt_file)
        seq_feats[pt_file.stem] = obj["embedding"].numpy()

    voxel_feats = {}
    for npy_file in (OUTPUTS / "3D_embeddings").glob("*.npy"):
        voxel = np.load(npy_file)
        if voxel.ndim == 4:  
            voxel = voxel[np.newaxis, ...]  
        voxel_feats[npy_file.stem] = voxel.astype(np.float32)

    # Load test mutations
    df_test = pd.read_csv(DATA / "mutations-test.csv")

    X = []
    mutations_used = []
    missing = []
    cnn3d_model_cache = {}

    for mut in df_test['Mutations']:
        seq_key = f"mutant_{mut}" if f"mutant_{mut}" in seq_feats else mut
        voxel_key = f"mutant_{mut}_activesite_voxel" if f"mutant_{mut}_activesite_voxel" in voxel_feats else None

        if seq_key not in seq_feats or voxel_key not in voxel_feats:
            missing.append(mut)
            continue

        seq_vec = seq_feats[seq_key]  
        voxel_vec = torch.tensor(voxel_feats[voxel_key])  

        in_channels = voxel_vec.shape[1]
        if in_channels not in cnn3d_model_cache:
            cnn3d_model_cache[in_channels] = CNN3D(in_channels=in_channels)
        cnn3d = cnn3d_model_cache[in_channels]

        with torch.no_grad():
            voxel_vec_out = cnn3d(voxel_vec).numpy().reshape(-1)

        X.append(np.concatenate([seq_vec, voxel_vec_out], axis=0))
        mutations_used.append(mut)

    if missing:
        print(f"Warning: missing embeddings for {len(missing)} mutations, they will be skipped")

    X = np.stack(X)
    print(f"Feature matrix shape: {X.shape}")

    # Predict
    y_pred = rf_model.predict(X)
    df_out = pd.DataFrame({
        'Mutations': mutations_used,
        'Predicted': y_pred
    })

    df_out.to_csv(OUTPUTS / "predictions.csv", index=False)
    print(f"Predictions saved to {OUTPUTS / 'predictions.csv'}")

if __name__ == "__main__":
    main()
