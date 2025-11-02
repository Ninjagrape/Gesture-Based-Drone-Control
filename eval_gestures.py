#!/usr/bin/env python3
# evaluates the knn gesture classifier on saved features by sweeping over k
# expects a csv with columns: label, gesture_type, subject_id, session_id, f0..fN[, v0,v1,v2] 
import nbimporter
import os
import argparse
import time
import json
from pathlib import Path
import sys
import numpy as np
import pickle
import pandas as pd
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, accuracy_score
)
# from mainQ1 import check_pkl_contents

# ----- json helper (convert numpy to python) ------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    return str(o)

#CHECK CONTENTS OF PKL FILES
def summarise_pkl_content(name, val):
    t = type(val).__name__
    shape = getattr(val, "shape", None)
    if shape is not None: 
        print(f"  {name}: {t} shape={shape}")
    else:                
        print(f"  {name}: {t}")
    
def check_pkl_contents(pkl_filename):
    pkl_path = os.path.join( pkl_filename)
    print(pkl_filename, ":")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("TYPE:", type(data).__name__)
    if isinstance(data, dict):
        print("Keys:", list(data.keys()))
        print("Summary:")
        for k, v in data.items():
            summarise_pkl_content(k, v)
        np.set_printoptions(suppress=True, precision=6)
        print("Data:")
        for k,v in data.items():
            if k in data:
                print(f"{k} =\n{data[k]}")
        print("\n")

# ----- io helpers ---------------------------------------------------------------
def load_dataset(csv_path, gesture_type=None):
    # load csv and optionally filter by gesture_type ('static' | 'dynamic')
    df = pd.read_csv(csv_path)
    if gesture_type:
        df = df[df["gesture_type"] == gesture_type]
        
    
    # inside load_dataset(...)
    if "subject_id" in df.columns:
        # factorize to integer codes: same subject -> same int
        groups = pd.factorize(df["subject_id"])[0].astype(np.int32)
    else:
        groups = np.zeros(len(df), dtype=np.int32)


    # features are all f* columns (ignore any v* here; velocity etc may have been concatenated already)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError("no feature columns found (expected columns starting with 'f').")
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    groups = df["subject_id"].to_numpy() if "subject_id" in df.columns else np.array(["na"] * len(df))
    return df, X, y, groups

def dataset_stats(df):
    # minimal stats for sanity checking
    info = {
        "num_samples": int(len(df)),
        "classes": df["label"].value_counts().sort_index().to_dict(),
        "subjects": df["subject_id"].value_counts().sort_index().to_dict() if "subject_id" in df.columns else {},
        "gesture_types": df["gesture_type"].value_counts().sort_index().to_dict() if "gesture_type" in df.columns else {},
    }
    return info

# ----- splitting helpers --------------------------------------------------------
def stratified_split(X, y, test_size=0.15, random_state=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(sss.split(X, y))
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

def subject_holdout(X, y, groups):
        # leave-one-subject-out; if only one subject, fall back to a stratified split
    uniq = np.unique(groups)
    if len(uniq) < 2:
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.2)
        yield X_tr, X_te, y_tr, y_te, -1  # use -1 to indicate random split
        return
    gkf = GroupKFold(n_splits=10)
    for tr_idx, te_idx in gkf.split(X, y, groups):
        held = int(groups[te_idx][0])  # ensure integer
        yield X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], held


# ----- model/train/eval ---------------------------------------------------------
def train_eval_knn(X_tr, y_tr, X_te, y_te, k=7, metric="euclidean"):
    # standardise features (fit on train, apply to test)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric=metric)

    t0 = time.time()
    clf.fit(X_trs, y_tr)
    fit_ms = (time.time() - t0) * 1000.0

    t0 = time.time()
    y_pred = clf.predict(X_tes)
    avg_inf_ms = (time.time() - t0) / max(len(X_tes), 1) * 1000.0

    # labels order fixed to training classes to keep cm shape stable
    train_labels = np.unique(y_tr)

    cm = confusion_matrix(y_te, y_pred, labels=train_labels)
    rep_dict = classification_report(y_te, y_pred, labels=train_labels, output_dict=True, zero_division=0)
    rep_text = classification_report(y_te, y_pred, labels=train_labels, zero_division=0)

    return {
        "k": int(k),
        "metric": metric,
        "acc": accuracy_score(y_te, y_pred),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        "cm": cm,                              # keep as ndarray for pretty print
        "labels_order": train_labels.tolist(), # json safe copy will be made later
        "report_dict": rep_dict,
        "report_text": rep_text,
        "fit_ms": fit_ms,
        "avg_inf_ms": avg_inf_ms,
    }

# ----- pretty prints ------------------------------------------------------------
def print_block(res, header=None, stream=sys.stdout):
    # header
    if header:
        print(header, file=stream)

    # accuracy
    print(f"\ntest accuracy: {res['acc']:.4f}\n", file=stream)

    # confusion matrix (as a tidy numpy-style block)
    print("confusion matrix:\n", file=stream, end="")
    # ensure integer-looking output where applicable
    cm = res["cm"]
    
    if cm.dtype.kind != "f":
        cm_to_show = cm
    else:
        cm_to_show = np.rint(cm).astype(int)
    print(cm_to_show, file=stream)

    # classification report
    print("\n\nclassification report:", file=stream)
    print(res["report_text"], file=stream)
    
    
    # plot_cm(cm, labels=["CHANGE SPEED", "GO", "HOLD", "ROTATE", "STOP", "TRANSLATE"], cmap="Purples", normalise=False)
       # plot

def plot_cm(cm, labels=None, title='Confusion Matrix for \n Drone Hand Gesture Recognition kNN Model \n (k=5)', cmap="coolwarm", normalise=False):
    # cm: 2d array
    if normalise:
        # row-normalise so each row sums to 1
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap )  # use an image, not plot
    plt.title(title, color = "white")
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(colors='white')          # tick colour
    # cbar.set_label('count', color='white')       # label colour (optional)
    # cbar.outline.set_edgecolor('white')          # box outline
    tick_marks = np.arange(cm.shape[0])
    if labels is None:
        labels = [str(i) for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=90, color = "white")
    plt.yticks(tick_marks, labels, color = "white")
    plt.xlabel('Predicted', color = "white")
    plt.ylabel('True', color = "white")
    
    # make background black
    plt.gca().set_facecolor('black')
    plt.gcf().patch.set_facecolor('black')

    # add counts/text
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.show()


def append_summary_row(rows, k, res):
    rows.append({
        "k": int(k),
        "acc": float(res["acc"]),
        "bal_acc": float(res["bal_acc"]),
        "macro_f1": float(res["macro_f1"]),
        "avg_inf_ms": float(res["avg_inf_ms"]),
    })

# ----- main --------------------------------------------------------------------
def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to dataset.csv")
    ap.add_argument("--gesture-type", choices=["static", "dynamic", "all"], default="all")
    ap.add_argument("--protocol", choices=["random", "subject"], default="random")
    ap.add_argument("--metric", default="euclidean")  # manhattan = 'manhattan', default euclidean
    ap.add_argument("--outdir", default="eval_out")
    # comma-separated list of k to sweep, e.g. "1,3,5,7,9"
    ap.add_argument("--ks", default="1,3,5,7,9")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # parse ks
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    # load and filter
    df, X, y, groups = load_dataset(args.csv, None if args.gesture_type == "all" else args.gesture_type)

    # quick stats
    stats = dataset_stats(df)
    pd.Series(stats["classes"]).to_csv(outdir / "class_counts.csv")
    if stats["subjects"]:
        pd.Series(stats["subjects"]).to_csv(outdir / "subject_counts.csv")
    if stats["gesture_types"]:
        pd.Series(stats["gesture_types"]).to_csv(outdir / "gesture_type_counts.csv")

    results = {"dataset_stats": stats, "runs": {}}

    if args.protocol == "random":
        # single stratified split; for each k print the detailed block and record a summary row
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.15)
        summary_rows = []

        for k in ks:
            res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=k, metric=args.metric)
            print_block(res, header=f"\n[random protocol] k = {k}")
            append_summary_row(summary_rows, k, res)
        pd.DataFrame(summary_rows).to_csv(outdir / "k_sweep.csv", index=False)
        results["runs"]["random"] = {"sweep": summary_rows}

    else:
        # subject-wise holdout; falls back to stratified if only one subject
        all_rows = []
        for X_tr, X_te, y_tr, y_te, held in subject_holdout(X, y, groups):
            per_subject_rows = []
            for k in ks:
                res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=k, metric=args.metric)
                print_block(res, header=f"\n[subject holdout] held_subject = {held} | k = {k}")
                append_summary_row(per_subject_rows, k, res)
                r = {"held_subject": str(held), **per_subject_rows[-1]}
                all_rows.append(r)
            # write per-subject summary csv
            pd.DataFrame(per_subject_rows).to_csv(outdir / f"subject_holdout_{held}.csv", index=False)

        if all_rows:
            df_subj = pd.DataFrame(all_rows)
            df_subj.to_csv(outdir / "subject_holdout_sweep.csv", index=False)
            results["runs"]["subject"] = {"sweep": all_rows}
            
    with open(outdir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
        
 

if __name__ == "__main__":
    main()
