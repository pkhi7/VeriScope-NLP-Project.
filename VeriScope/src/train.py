"""Model development & evaluation.

For each mode (6-class, binary):
  * try 3 feature sets x 5 classifiers
  * select the best by macro-F1 on the VALIDATION split
  * report final numbers on the untouched TEST split
  * save pipeline, metrics, comparison table and figures
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from config import (BINARY_FAKE, BINARY_REAL, DATA_PROCESSED, FIG_DIR, LABEL_DISPLAY,
                    LABEL_ORDER, MODELS_DIR, RANDOM_STATE, REPORTS_DIR)
from src.features import get_vectorizers


def get_models():
    return {
        "Multinomial NB": MultinomialNB(alpha=0.5),
        "Complement NB": ComplementNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced"),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced"), cv=3),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=RANDOM_STATE),
    }


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_mode(df, mode):
    target = "label" if mode == "multiclass" else "binary_label"
    tr, va, te = (df[df["split"] == s] for s in ["train", "valid", "test"])
    ytr, yva, yte = tr[target], va[target], te[target]
    print(f"\n{'='*70}\nMODE: {mode}  | train={len(tr):,} valid={len(va):,} test={len(te):,}\n{'='*70}")

    rows, cache = [], {}
    for fname, vec in get_vectorizers().items():
        Xtr = vec.fit_transform(tr["clean_text"])
        Xva, Xte = vec.transform(va["clean_text"]), vec.transform(te["clean_text"])
        for mname, model in get_models().items():
            t0 = time.time()
            model.fit(Xtr, ytr)
            mv, mt = metrics(yva, model.predict(Xva)), metrics(yte, model.predict(Xte))
            rows.append({"features": fname, "model": mname,
                         **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()},
                         "train_seconds": round(time.time() - t0, 1)})
            cache[(fname, mname)] = (vec, model, Xtr)
            print(f"{fname:<20}{mname:<22} val F1={mv['f1_macro']:.3f}  test acc={mt['accuracy']:.3f}")

    res = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / f"{mode}_model_comparison.csv", index=False)

    best = res.iloc[0]
    vec, model, Xtr = cache[(best["features"], best["model"])]
    Xte = vec.transform(te["clean_text"])
    pred = model.predict(Xte)
    classes = list(model.classes_)

    # explainer: a linear model on the same features gives word-level contributions
    if isinstance(model, LogisticRegression):
        explainer = model
    else:
        explainer = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)

    labels_order = [l for l in (LABEL_ORDER if mode == "multiclass" else [BINARY_FAKE, BINARY_REAL]) if l in classes]
    disp = (lambda l: LABEL_DISPLAY.get(l, l))
    report = classification_report(yte, pred, labels=labels_order, target_names=[disp(l) for l in labels_order],
                                   zero_division=0, output_dict=True)
    (REPORTS_DIR / f"{mode}_classification_report.txt").write_text(
        classification_report(yte, pred, labels=labels_order, target_names=[disp(l) for l in labels_order], zero_division=0))

    majority = float(yte.value_counts(normalize=True).max())
    out = {
        "mode": mode, "best_features": best["features"], "best_model": best["model"],
        "test": metrics(yte, pred), "majority_baseline_accuracy": majority,
        "random_baseline_accuracy": 1 / len(classes),
        "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
        "n_features": int(Xtr.shape[1]), "classes": classes, "per_class": report,
    }
    cleaning = json.loads((REPORTS_DIR / "cleaning_log.json").read_text())
    out["dataset_source"] = cleaning.get("dataset_source", "unknown")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"{mode}_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    joblib.dump({"vectorizer": vec, "model": model, "explainer": explainer, "classes": classes, "mode": mode},
                MODELS_DIR / f"{mode}_pipeline.joblib", compress=3)

    # --- figures
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(yte, pred, labels=labels_order)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    plt.figure(figsize=(7, 5.5) if mode == "multiclass" else (5, 4.2))
    sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=[disp(l) for l in labels_order],
                yticklabels=[disp(l) for l in labels_order], cbar_kws={"label": "Row-normalised share"})
    plt.title(f"Confusion matrix - {best['model']} ({mode})"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"{mode}_confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(9, 4.8))
    piv = res.pivot(index="model", columns="features", values="val_f1_macro")
    piv.plot(kind="bar", ax=plt.gca(), width=0.8)
    plt.ylabel("Validation macro-F1"); plt.xlabel(""); plt.xticks(rotation=20)
    plt.title(f"Model & feature comparison ({mode})"); plt.legend(title="", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / f"{mode}_model_comparison.png", dpi=150); plt.close()

    print(f"\nBEST ({mode}): {best['model']} + {best['features']}")
    print({k: round(v, 4) for k, v in out["test"].items()})
    print(f"majority-class baseline acc={majority:.3f}")
    return out


def main():
    df = pd.read_csv(DATA_PROCESSED / "liar_clean.csv")
    df["clean_text"] = df["clean_text"].fillna("")
    for mode in ["binary", "multiclass"]:
        run_mode(df, mode)


if __name__ == "__main__":
    main()
