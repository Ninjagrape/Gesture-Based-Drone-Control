#pkl_inspect_export.py
# export training data and metadata from your saved knn pkl
# works with payloads: {"scaler": StandardScaler, "clf": KNeighborsClassifier, "feature_dim": int}

#USAGE:
#python3 pkl_inspect_export.py

import json
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd

PKL_PATH = "gesture_dynamic.pkl"  
OUT_DIR  = Path("pkl_export")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    #loading
    with open(PKL_PATH, "rb") as f:
        payload = pkl.load(f)
        print(payload["clf"].classes_)

    # pull components
    scaler = payload.get("scaler", None)
    clf    = payload.get("clf", None)
    feat_d = payload.get("feature_dim", None)

    if scaler is None or clf is None:
        raise RuntimeError("pkl missing 'scaler' or 'clf' keys")

    # training data stored inside knn
    Xs = getattr(clf, "_fit_X", None)     # scaled features used during fit
    y  = getattr(clf, "_y", None)         # labels
    if Xs is None or y is None:
        raise RuntimeError("knn does not expose _fit_X/_y; cannot export training set")

    Xs = np.asarray(Xs, dtype=np.float64)
    y  = np.asarray(y)

    # invert scaling to get original features 
    # standardscaler stores per-feature mean and scale
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise RuntimeError("scaler missing mean_/scale_, cannot invert scaling")
    mean  = scaler.mean_.astype(np.float64)
    scale = scaler.scale_.astype(np.float64)
    X_unscaled = Xs * scale + mean

    # build dataframes with consistent headers f0..fN
    nfeat = X_unscaled.shape[1]
    cols  = [f"f{i}" for i in range(nfeat)]
    
    #unscaled df
    df_unscaled = pd.DataFrame(X_unscaled, columns=cols)
    df_unscaled.insert(0, "label", y)
    
    #unscaled df
    df_scaled = pd.DataFrame(Xs, columns=cols)
    df_scaled.insert(0, "label", y)

    # save csvs
    df_unscaled.to_csv(OUT_DIR / "training_dynamic_unscaled.csv", index=False)
    df_scaled.to_csv(OUT_DIR / "training_dynamic_scaled.csv", index=False)

    # save metadata
    meta = {
        "feature_dim_reported": int(feat_d) if feat_d is not None else None,
        "feature_dim_detected": int(nfeat),
        "classes": list(getattr(clf, "classes_", [])),
        "n_samples": int(len(y)),
        "n_neighbors": int(getattr(clf, "n_neighbors", -1)),
        "metric": getattr(clf, "metric", "minkowski"),
        "scaler_with_mean": bool(getattr(scaler, "with_mean", True)),
        "scaler_with_std": bool(getattr(scaler, "with_std", True)),
    }
    
    print(meta)
    
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # quick prints
    print(f"[ok] exported -> {OUT_DIR}/training_unscaled.csv and training_scaled.csv")
    print(f"[ok] meta     -> {OUT_DIR}/meta.json")
    print(f"[info] samples={meta['n_samples']} feat_dim={meta['feature_dim_detected']} classes={meta['classes']}")

if __name__ == "__main__":
    main()
