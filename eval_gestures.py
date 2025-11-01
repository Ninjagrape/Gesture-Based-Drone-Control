#!/usr/bin/env python3
# evaluates the knn gesture classifier on saved features
# expects a csv with columns: label, gesture_type, subject_id, session_id, f0..fN[, v0,v1,v2]

import argparse
import time
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, accuracy_score
)

# ----- json helper (convert numpy to python) ------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    return str(o)

# ----- io helpers ---------------------------------------------------------------
def load_dataset(csv_path, gesture_type=None):
    # load csv and optionally filter by gesture_type ('static' | 'dynamic')
    df = pd.read_csv(csv_path)
    if gesture_type:
        df = df[df["gesture_type"] == gesture_type]

    # features are all f* columns (and we ignore v* here; they were concatenated at collection if needed)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    groups = df.get("subject_id", pd.Series(["na"] * len(df))).to_numpy()
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
        yield X_tr, X_te, y_tr, y_te, "na"
        return
    gkf = GroupKFold(n_splits=len(uniq))
    for tr_idx, te_idx in gkf.split(X, y, groups):
        held = groups[te_idx][0]
        yield X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], held

# ----- model/train/eval ---------------------------------------------------------
def train_eval_knn(X_tr, y_tr, X_te, y_te, k=7, metric="minkowski"):
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
    rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)

    return {
        "k": int(k),
        "metric": metric,
        "acc": accuracy_score(y_te, y_pred),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        "cm": cm.tolist(),                    # json safe
        "labels_order": train_labels.tolist(),# json safe
        "report": rep,                        # dict of floats
        "fit_ms": fit_ms,
        "avg_inf_ms": avg_inf_ms,
    }

def k_sweep(X_tr, y_tr, X_te, y_te, ks=(1, 3, 5, 7, 9), metric="minkowski"):
    rows = []
    for k in ks:
        res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=k, metric=metric)
        rows.append({
            "k": res["k"],
            "acc": res["acc"],
            "bal_acc": res["bal_acc"],
            "macro_f1": res["macro_f1"],
            "avg_inf_ms": res["avg_inf_ms"],
        })
    return pd.DataFrame(rows)

#print confusition matrix
def print_confusion_matrix(labels, cm, *, normalise=False, stream=sys.stdout):
    import numpy as np
    import pandas as pd
    lab = list(labels)
    m = np.array(cm, dtype=float)
    if normalise and m.sum(axis=1, keepdims=True).all():
        row_sums = m.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        m = m / row_sums
    df = pd.DataFrame(m, index=lab, columns=lab)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nconfusion matrix" + (" (normalised)" if normalise else ""), file=stream)
        print(df.round(3).to_string(), file=stream)




# ----- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to dataset.csv")
    ap.add_argument("--gesture-type", choices=["static", "dynamic", "all"], default="all")
    ap.add_argument("--protocol", choices=["random", "subject"], default="random")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--metric", default="minkowski")  # manhattan = 'manhattan'
    ap.add_argument("--outdir", default="eval_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

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
        # single stratified split, sweep k, then keep best
        X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.15)

        sweep = k_sweep(X_tr, y_tr, X_te, y_te, ks=(1, 3, 5, 7, 9), metric=args.metric)
        sweep.to_csv(outdir / "k_sweep.csv", index=False)

        best_k = int(sweep.sort_values("macro_f1", ascending=False).iloc[0]["k"])
        res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=best_k, metric=args.metric)

        results["runs"]["random"] = {"best_k": best_k, **res}

        # headline print
        print(f"[random] k={res['k']} acc={res['acc']:.3f} bal_acc={res['bal_acc']:.3f} "
              f"macro_f1={res['macro_f1']:.3f} avg_inf_ms={res['avg_inf_ms']:.3f}")
        
        print_confusion_matrix(res["labels_order"], res["cm"], normalise=False)
        print_confusion_matrix(res["labels_order"], res["cm"], normalise=True)


    else:
        # subject-wise holdout; falls back to stratified if only one subject
        rows = []
        cms = {}
        for X_tr, X_te, y_tr, y_te, held in subject_holdout(X, y, groups):
            sweep = k_sweep(X_tr, y_tr, X_te, y_te, ks=(1, 3, 5, 7, 9), metric=args.metric)
            best_k = int(sweep.sort_values("macro_f1", ascending=False).iloc[0]["k"])
            res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=best_k, metric=args.metric)

            rows.append({
                "held_subject": str(held),
                "k": res["k"],
                "acc": res["acc"],
                "bal_acc": res["bal_acc"],
                "macro_f1": res["macro_f1"],
                "avg_inf_ms": res["avg_inf_ms"],
            })
            cms[str(held)] = {"labels_order": res["labels_order"], "cm": res["cm"]}

        df_subj = pd.DataFrame(rows).sort_values(["held_subject"])
        df_subj.to_csv(outdir / "subject_holdout.csv", index=False)

        results["runs"]["subject"] = {"summary": rows, "cms": cms}

        # headline print
        if len(df_subj):
            d = df_subj[["acc", "bal_acc", "macro_f1", "avg_inf_ms"]].describe().loc[["mean", "min", "max"]]
            print(d)

    # save json (numpy-safe)
    with open(outdir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

if __name__ == "__main__":
    main()
