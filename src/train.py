"""
Train, tune and evaluate the electricity-theft detector.

Run:  python src/train.py --data data/synthetic_sgcc.csv
      python src/train.py --data data/data.csv          # real SGCC dataset

Outputs:
  models/theft_model.keras     trained network
  models/config.json           decision threshold + input shapes
  reports/metrics.json         all numbers for your README / resume
  reports/*.png                ROC curve, confusion matrices, training curves
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from model import build_model
from preprocess import build_inputs, load_dataset

tf.random.set_seed(42)
np.random.seed(42)


def metrics_at(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(thr), 3),
            "accuracy": round(accuracy_score(y, pred), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4),
            "f1": round(f1_score(y, pred), 4),
            "auc": round(roc_auc_score(y, p), 4),
            "false_positives": int(fp), "true_positives": int(tp),
            "false_negatives": int(fn), "true_negatives": int(tn)}


def best_threshold(y_val, p_val, min_recall_ratio=0.95):
    """Pick the threshold that minimises false positives while keeping
    at least 95% of the recall we would get at the default 0.5 threshold.
    Field teams care about wasted site visits (false positives), but we must
    not start missing real thieves."""
    base_recall = recall_score(y_val, (p_val >= 0.5).astype(int))
    best = 0.5
    best_fp = ((p_val >= 0.5) & (y_val == 0)).sum()
    for t in np.arange(0.30, 0.96, 0.01):
        pred = (p_val >= t).astype(int)
        if recall_score(y_val, pred) < min_recall_ratio * base_recall:
            continue
        fp = ((pred == 1) & (y_val == 0)).sum()
        if fp < best_fp or (fp == best_fp and f1_score(y_val, pred) > f1_score(y_val, (p_val >= best) * 1)):
            best, best_fp = t, fp
    return float(best)


def subset(inputs, idx):
    return {k: v[idx] for k, v in inputs.items()}


def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("1) Loading data ...")
    X_raw, y, _ = load_dataset(args.data)
    print(f"   {X_raw.shape[0]} customers, {X_raw.shape[1]} days, theft rate {y.mean():.1%}")

    print("2) Preprocessing + feature engineering ...")
    daily, weekly, stats = build_inputs(X_raw)
    inputs = {"daily": daily, "weekly": weekly, "stats": stats}

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    tr, va = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=42)
    print(f"   train {len(tr)} | val {len(va)} | test {len(te)}")

    # Class imbalance: thieves are rare -> weight them more
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
    class_weight = {0: cw[0], 1: cw[1]}

    print("3) Baseline: logistic regression on engineered features ...")
    sc = StandardScaler().fit(stats[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(stats[tr]), y[tr])
    p_lr = lr.predict_proba(sc.transform(stats[te]))[:, 1]
    baseline = metrics_at(y[te], p_lr, 0.5)
    print(f"   baseline test: {baseline}")

    print("4) Hyperparameter search (validation AUC) ...")
    grid = list(itertools.product(args.units, args.dropout, args.lr))
    results, best_auc, best_cfg = [], -1, None
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                           patience=4, restore_best_weights=True)]
    for units, dropout, lr_ in grid:
        m = build_model(daily.shape[1], weekly.shape[1], stats.shape[1],
                        units=units, dropout=dropout, lr=lr_)
        m.fit(subset(inputs, tr), y[tr], validation_data=(subset(inputs, va), y[va]),
              epochs=args.search_epochs, batch_size=128, class_weight=class_weight,
              callbacks=cb, verbose=0)
        auc = roc_auc_score(y[va], m.predict(subset(inputs, va), verbose=0).ravel())
        results.append({"units": units, "dropout": dropout, "lr": lr_, "val_auc": round(auc, 4)})
        print(f"   units={units:<4} dropout={dropout:<4} lr={lr_:<7} -> val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_cfg = auc, (units, dropout, lr_)

    print(f"5) Training final model with {best_cfg} ...")
    units, dropout, lr_ = best_cfg
    model = build_model(daily.shape[1], weekly.shape[1], stats.shape[1],
                        units=units, dropout=dropout, lr=lr_)
    cb_final = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                                 patience=6, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max",
                                                     factor=0.5, patience=3)]
    hist = model.fit(subset(inputs, tr), y[tr], validation_data=(subset(inputs, va), y[va]),
                     epochs=args.epochs, batch_size=128, class_weight=class_weight,
                     callbacks=cb_final, verbose=2)

    print("6) Threshold tuning on the validation set ...")
    p_val = model.predict(subset(inputs, va), verbose=0).ravel()
    thr = best_threshold(y[va], p_val)
    print(f"   chosen threshold = {thr:.2f}")

    print("7) Final evaluation on the held-out TEST set ...")
    p_te = model.predict(subset(inputs, te), verbose=0).ravel()
    default = metrics_at(y[te], p_te, 0.5)
    tuned = metrics_at(y[te], p_te, thr)
    fp_reduction = (1 - tuned["false_positives"] / max(default["false_positives"], 1)) * 100
    fp_vs_baseline = (1 - tuned["false_positives"] / max(baseline["false_positives"], 1)) * 100

    report = {"dataset": args.data, "n_customers": int(len(y)), "n_days": int(X_raw.shape[1]),
              "best_hyperparameters": {"units": units, "dropout": dropout, "lr": lr_},
              "search_results": results,
              "baseline_logreg": baseline, "dnn_default_threshold": default,
              "dnn_tuned_threshold": tuned,
              "fp_reduction_vs_default_threshold_pct": round(fp_reduction, 1),
              "fp_reduction_vs_baseline_pct": round(fp_vs_baseline, 1)}
    json.dump(report, open("reports/metrics.json", "w"), indent=2)

    model.save("models/theft_model.keras")
    json.dump({"threshold": thr, "n_days": int(daily.shape[1])}, open("models/config.json", "w"))

    # ---- Plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.history["loss"], label="train"); ax[0].plot(hist.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist.history["auc"], label="train"); ax[1].plot(hist.history["val_auc"], label="val")
    ax[1].set_title("AUC"); ax[1].legend()
    plt.tight_layout(); plt.savefig("reports/training_curves.png", dpi=120); plt.close()

    fpr, tpr, _ = roc_curve(y[te], p_te); fpr_b, tpr_b, _ = roc_curve(y[te], p_lr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Wide&Deep DNN (AUC {tuned['auc']:.3f})")
    plt.plot(fpr_b, tpr_b, label=f"LogReg baseline (AUC {baseline['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey"); plt.xlabel("False positive rate")
    plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC curve (test set)")
    plt.tight_layout(); plt.savefig("reports/roc_curve.png", dpi=120); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, t, name in [(ax[0], 0.5, "threshold 0.50"), (ax[1], thr, f"tuned threshold {thr:.2f}")]:
        ConfusionMatrixDisplay(confusion_matrix(y[te], (p_te >= t).astype(int)),
                               display_labels=["normal", "theft"]).plot(ax=a, colorbar=False)
        a.set_title(name)
    plt.tight_layout(); plt.savefig("reports/confusion_matrices.png", dpi=120); plt.close()

    print("\n================ RESULTS (test set) ================")
    print(f"Baseline LogReg : acc {baseline['accuracy']:.3f}  AUC {baseline['auc']:.3f}  FP {baseline['false_positives']}")
    print(f"DNN @0.50       : acc {default['accuracy']:.3f}  AUC {default['auc']:.3f}  FP {default['false_positives']}  recall {default['recall']:.3f}")
    print(f"DNN @{thr:.2f}       : acc {tuned['accuracy']:.3f}  AUC {tuned['auc']:.3f}  FP {tuned['false_positives']}  recall {tuned['recall']:.3f}")
    print(f"False positives reduced by {fp_reduction:.1f}% vs default threshold, "
          f"{fp_vs_baseline:.1f}% vs baseline")
    print("Saved: models/theft_model.keras, reports/metrics.json, reports/*.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_sgcc.csv")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--search-epochs", type=int, default=12)
    p.add_argument("--units", type=int, nargs="+", default=[64, 128])
    p.add_argument("--dropout", type=float, nargs="+", default=[0.2, 0.4])
    p.add_argument("--lr", type=float, nargs="+", default=[1e-3])
    main(p.parse_args())
