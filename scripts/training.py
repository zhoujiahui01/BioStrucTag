#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from joblib import dump

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN3D(nn.Module):

    def __init__(self, in_channels=4):

        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, 32, 3)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(32, 64, 3)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(64, 128, 3)

        self.avgpool = nn.AdaptiveAvgPool3d((2,2,2))

        self.flatten = nn.Flatten()

        self.fc_feat = nn.Linear(128*2*2*2, 512)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))

        x = self.avgpool(x)

        x = self.flatten(x)

        feat = self.fc_feat(x)

        out = self.mlp(feat)

        return out, feat

def load_data(DATA, OUTPUTS, label):

    df = pd.read_csv(DATA/"mutations.csv")

    seq_feats = {}

    for pt in (OUTPUTS/"1D_embeddings").glob("*.pt"):

        obj = torch.load(pt)

        seq_feats[pt.stem] = obj["embedding"].numpy()

    voxel_feats = {}

    for npy in (OUTPUTS/"3D_embeddings").glob("*.npy"):

        voxel = np.load(npy)
        
        voxel = np.squeeze(voxel)
        
        if voxel.ndim != 4:
            print(f"Warning: Skip {npy.stem}, invalid voxel dim: {voxel.ndim} (expected 4)")
            continue
        
        voxel_feats[npy.stem] = voxel.astype(np.float32)

    X_seq = []
    X_voxel = []
    y = []

    missing = []

    for _,row in df.iterrows():

        mut = row["Mutations"].replace("/","_")

        seq_key = mut

        voxel_key = f"mutant_{mut}_activesite_voxel"

        if seq_key not in seq_feats or voxel_key not in voxel_feats:

            missing.append(mut)

            continue

        X_seq.append(seq_feats[seq_key])

        X_voxel.append(voxel_feats[voxel_key][np.newaxis, ...])

        y.append(row[label])

    if missing:

        print(f"Warning: missing embeddings for {len(missing)} mutants")

    X_voxel = np.stack(X_voxel)
    X_voxel = np.squeeze(X_voxel, axis=1)  
    
    print(f"After squeeze - X_voxel shape: {X_voxel.shape}")  
    
    return np.stack(X_seq), X_voxel, np.array(y)

def train_cnn(X_voxel,y):


    X_train,X_val,y_train,y_val = train_test_split(
        X_voxel,y,test_size=0.2,random_state=42
    )

    X_train = torch.tensor(X_train).to(device)
    X_val = torch.tensor(X_val).to(device)

    y_train = torch.tensor(y_train).float().unsqueeze(1).to(device)
    y_val = torch.tensor(y_val).float().unsqueeze(1).to(device)

    in_channels = X_train.shape[1]  
    model = CNN3D(in_channels=in_channels).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    loss_fn = nn.MSELoss()

    epochs = 100

    for epoch in range(epochs):

        model.train()

        pred,_ = model(X_train)

        loss = loss_fn(pred,y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_pred,_ = model(X_val)

            val_loss = loss_fn(val_pred,y_val)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train={loss.item():.4f} "
            f"val={val_loss.item():.4f}"
        )

    return model


def extract_struct_features(model,X_voxel):

    model.eval()

    feats = []

    with torch.no_grad():

        for voxel in X_voxel:
            voxel = torch.tensor(voxel).unsqueeze(0).to(device)
            _,feat = model(voxel)
            feats.append(feat.cpu().numpy().flatten())

    return np.array(feats)



def train_rf(X,y):

    kf = KFold(n_splits=5,shuffle=True,random_state=42)

    rmses = []

    for train_idx,test_idx in kf.split(X):

        X_train,X_test = X[train_idx],X[test_idx]

        y_train,y_test = y[train_idx],y[test_idx]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test,pred))

        rmses.append(rmse)

    print(
        f"5-fold RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}"  
    )

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )

    final_model.fit(X,y)

    return final_model


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--label",required=True)

    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]

    DATA = ROOT/"data"
    OUTPUTS = ROOT/"outputs"
    MODELS = ROOT/"models"

    MODELS.mkdir(exist_ok=True)

    print("Loading data...")

    X_seq,X_voxel,y = load_data(DATA,OUTPUTS,args.label)

    print("Sequence feature:",X_seq.shape)  
    print("Voxel feature:",X_voxel.shape)   

    print("\nTraining 3D CNN...")

    cnn_model = train_cnn(X_voxel,y)

    torch.save(cnn_model.state_dict(),MODELS/"cnn3d_weights.pt")

    print("\nExtracting structure embeddings...")

    X_struct = extract_struct_features(cnn_model,X_voxel)

    print("Structure feature:",X_struct.shape)  

    X = np.concatenate([X_seq,X_struct],axis=1)

    print("Final feature matrix:",X.shape)  

    print("\nTraining Random Forest...")

    rf_model = train_rf(X,y)

    dump(rf_model,MODELS/"rf_model.joblib")

    print("\nModels saved:")
    print("models/cnn3d_weights.pt")
    print("models/rf_model.joblib")


if __name__ == "__main__":
    main()
