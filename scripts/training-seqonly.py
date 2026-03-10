#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from joblib import dump


def load_sequence_only_data(DATA, OUTPUTS, label):

    df = pd.read_csv(DATA/"mutations.csv")

    seq_feats = {}
    for pt in (OUTPUTS/"1D_embeddings").glob("*.pt"):
        import torch
        obj = torch.load(pt)
        seq_feats[pt.stem] = obj["embedding"].numpy()

    X_seq = []
    y = []
    missing = []

    for _,row in df.iterrows():

        mut = row["Mutations"].replace("/","_")
        seq_key = mut

        if seq_key not in seq_feats:
            missing.append(mut)
            continue

        X_seq.append(seq_feats[seq_key])
        y.append(row[label])

    if missing:
        print(f"Warning: missing sequence embeddings for {len(missing)} mutants")

    X_seq = np.stack(X_seq)
    print(f"Final sequence feature shape: {X_seq.shape}")
    return X_seq, np.array(y)


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
        f"Sequence Only - 5-fold RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}"
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
    parser.add_argument("--label",required=True, help="Target column name in mutations.csv")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT/"data"
    OUTPUTS = ROOT/"outputs"
    MODELS = ROOT/"models"
    MODELS.mkdir(exist_ok=True)

    print("===== Loading Sequence-Only Data =====")
    X_seq, y = load_sequence_only_data(DATA, OUTPUTS, args.label)

    print("\n===== Training Random Forest =====")
    rf_model = train_rf(X_seq, y)
    dump(rf_model, MODELS/"rf_sequence_only.joblib")

    print("\n===== Model Saving Complete =====")
    print("RF model saved: models/rf_sequence_only.joblib")


if __name__ == "__main__":
    main()