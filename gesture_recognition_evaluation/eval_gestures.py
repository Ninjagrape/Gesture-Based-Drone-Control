#eval_gestures.py
# evaluates the knn gesture classifier on saved features by sweeping over k, random protocol, 5 stratified folds, 85/15 train-test ratio 

# EXPECTS: 
# expects a csv with columns: label, gesture_type, subject_id, session_id, f0..fN[, v0,v1,v2]

#USAGE:
# for default sweep:
# python3 eval_gestures.py --csv dataset.csv 
# for custom k value/s:
# python3 eval_gestures.py --csv dataset.csv --ks 5
# python3 eval_gestures.py --csv dataset.csv --ks 5,7

#LIBRARIES----------------------------------------
import os
import argparse
import time
from pathlib import Path
import sys
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score, accuracy_score
)

#CHECK CONTENTS OF PKL------------------------------
#not used for general execution 
def summarise_pkl_content(name, val):
    """
    print a summary of an object in pkl
    """
    t = type(val).__name__
    shape = getattr(val, "shape", None)
    if shape is not None:
        print(f"  {name}: {t} shape={shape}")
    else:
        print(f"  {name}: {t}")

def check_pkl_contents(pkl_filename):
    """
    print keys plus brief summaries of values of loaded pkl
    """
    pkl_path = os.path.join(pkl_filename)
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
        for k, v in data.items():
            if k in data:
                print(f"{k} =\n{data[k]}")
        print("\n")

#IO HELPERS-------------------------------------------
def load_dataset(csv_path):
    """
    - reads dataset csv and return (df, X, y)
    - X: float32 ndarray built from columns starting with 'f'
    - y: label column as a 1d array
    """
    
    df = pd.read_csv(csv_path)
    
    #collect feature columns 
    feat_cols = [c for c in df.columns if c.startswith("f")]
    
    if not feat_cols:
        raise ValueError("no feature columns found (expected columns starting with 'f').")
    
    #build matrices
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return df, X, y


#STRATIFIED SPLITING----------------------------------
def stratified_split(X, y, test_size=0.15, random_state=67):
    """
    perform a single stratified train/test split with given test_size.
    returns X_tr, X_te, y_tr, y_te.
    """
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(sss.split(X, y))
    return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

#MODEL TRAINING---------------------------------------
def train_eval_knn(X_tr, y_tr, X_te, y_te, k=7):
    """
    - fit a kNN classifier on standardised features and evaluate
    - returns a dict with accuracy, balanced accuracy, macro f1, confusion matrix, report text, and timing
    """
    # standardise features (fit on train, apply to test)
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)

    #employ classifier
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    
    #fit timing (ms)
    t0 = time.time()
    clf.fit(X_trs, y_tr)
    fit_ms = (time.time() - t0) * 1000.0
    
    #interence timing as average per-sample (ms)
    t0 = time.time()
    y_pred = clf.predict(X_tes)
    avg_inf_ms = (time.time() - t0) / max(len(X_tes), 1) * 1000.0
    
    # fix labels order to the training classes so layout is stable across runs
    train_labels = np.unique(y_tr)
    
    # metrics
    cm = confusion_matrix(y_te, y_pred, labels=train_labels)
    rep_dict = classification_report(y_te, y_pred, labels=train_labels, output_dict=True, zero_division=0)
    rep_text = classification_report(y_te, y_pred, labels=train_labels, zero_division=0)

    return {
        "k": int(k),
        "acc": accuracy_score(y_te, y_pred),
        "bal_acc": balanced_accuracy_score(y_te, y_pred),
        "macro_f1": f1_score(y_te, y_pred, average="macro"),
        "cm": cm,
        "labels_order": train_labels.tolist(),
        "report_dict": rep_dict,
        "report_text": rep_text,
        "fit_ms": fit_ms,
        "avg_inf_ms": avg_inf_ms,
    }

# PRINTING & PLOTTING -----------------------------------------
def print_block(res, header=None, stream=sys.stdout):
    """
    print confusion matrix, and full classification report, plot confusion-matrix 
    """
    if header:
        print(header, file=stream)
        
    #print test accuracy
    print(f"\ntest accuracy: {res['acc']:.4f}\n", file=stream)
    
    #print confusion matrix 
    print("confusion matrix:\n", file=stream, end="")
    cm = res["cm"]
    cm_to_show = cm if cm.dtype.kind != "f" else np.rint(cm).astype(int)
    print(cm_to_show, file=stream)
    
    #print classification report
    print("\n\nclassification report:", file=stream)
    print(res["report_text"], file=stream)
    
    #display confusion matrix
    plot_cm(
        cm,
        labels=["CHANGE SPEED", "GO", "HOLD", "ROTATE", "STOP", "TRANSLATE"],
        cmap="Purples",
        normalise=False
    )

def plot_cm(cm, labels=None, title=f'Confusion Matrix for \n Drone Hand Gesture Recognition \n kNN Model', cmap="coolwarm", normalise=False):
    """
    displays a confusion matrix map
    """
    #normalise data if desired
    if normalise:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        
    #plot figure
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(colors='white')
    tick_marks = np.arange(cm.shape[0])
    if labels is None:
        labels = [str(i) for i in tick_marks]
        
    # axis labels and ticks
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # overlay counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalise else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.show()


#MAIN--------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to dataset.csv")
    ap.add_argument("--ks", default="1,3,5,7,10")  # list of k to sweep
    
    args = ap.parse_args()
    
    # parse k list
    ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())

    # load dataset 
    df, X, y = load_dataset(args.csv)

    # stratified split
    # for each k print the detailed block and record a summary row
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=0.15)
    summary_rows = []

    for k in ks:
        res = train_eval_knn(X_tr, y_tr, X_te, y_te, k=k)
        print_block(res, header=f"\n k = {k}")
        summary_rows.append({
            "k": int(k),
            "acc": float(res["acc"]),
            "bal_acc": float(res["bal_acc"]),
            "macro_f1": float(res["macro_f1"]),
            "avg_inf_ms": float(res["avg_inf_ms"]),
        })

    #print summary table for k sweep
    print("\nsummary (per k):")
    df_sum = pd.DataFrame(summary_rows)
    print(df_sum.to_string(index=False))

if __name__ == "__main__":
    main()
