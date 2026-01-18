#!/usr/bin/env python3

import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from joblib import dump

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Column name in data/mutations.csv for regression target")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    OUTPUTS = ROOT / "outputs"
    MODELS = ROOT / "models"
    MODELS.mkdir(exist_ok=True)

    df_labels = pd.read_csv(DATA / "mutations.csv")
    y = df_labels[args.label].values

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

    X = []
    missing = []
    cnn3d_model_cache = {}

    for mut in df_labels['Mutations']:
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

    if missing:
        print(f"Warning: missing embeddings for {len(missing)} mutations, they will be skipped")

    X = np.stack(X)
    y_used = y[:X.shape[0]]

    print(f"Final feature matrix shape: {X.shape}, labels shape: {y_used.shape}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_used[train_idx], y_used[test_idx]
        model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)

    print(f"5-fold RMSE: {np.mean(rmses):.4f} +- {np.std(rmses):.4f}")

    final_model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    final_model.fit(X, y_used)
    dump(final_model, MODELS / "rf_model.joblib")
    print(f"Trained model saved to {MODELS / 'rf_model.joblib'}")

if __name__ == "__main__":
    main()
