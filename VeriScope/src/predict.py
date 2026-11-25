"""Inference + explanation used by the Streamlit GUI (and usable from the command line)."""
import json
import re
import sys

import joblib
import numpy as np

from config import MODELS_DIR
from src.preprocess import preprocess_text


def models_available(mode: str) -> bool:
    return (MODELS_DIR / f"{mode}_pipeline.joblib").exists()


class ClaimClassifier:
    def __init__(self, mode: str = "binary"):
        bundle = joblib.load(MODELS_DIR / f"{mode}_pipeline.joblib")
        self.mode = mode
        self.vec = bundle["vectorizer"]
        self.model = bundle["model"]
        self.explainer = bundle["explainer"]
        self.classes = list(bundle["classes"])
        self.names = self.vec.get_feature_names_out()
        mpath = MODELS_DIR / f"{mode}_metrics.json"
        self.metrics = json.loads(mpath.read_text()) if mpath.exists() else {}

    def _class_weights(self, k: int):
        coef = self.explainer.coef_
        if coef.shape[0] == 1:  # binary: coef refers to classes_[1]
            return coef[0] if k == 1 else -coef[0]
        return coef[k]

    def predict(self, text: str, top_k: int = 6) -> dict:
        clean = preprocess_text(text)
        X = self.vec.transform([clean]).tocsr()
        proba = self.model.predict_proba(X)[0]
        k = int(np.argmax(proba))
        w = self._class_weights(k)
        idx, vals = X.indices, X.data
        contrib = vals * w[idx]
        order = np.argsort(contrib)
        supporting = [(self.names[idx[i]], float(contrib[i])) for i in order[::-1] if contrib[i] > 0][:top_k]
        opposing = [(self.names[idx[i]], float(contrib[i])) for i in order if contrib[i] < 0][:top_k]

        # per-word scores for highlighting the ORIGINAL claim
        score_by_feat = {self.names[i]: float(c) for i, c in zip(idx, contrib)}
        words = []
        for tok in re.findall(r"\S+", text):
            stem = preprocess_text(tok)
            words.append((tok, score_by_feat.get(stem, 0.0) if stem else 0.0))

        return {
            "label": self.classes[k],
            "confidence": float(proba[k]),
            "probabilities": {c: float(p) for c, p in zip(self.classes, proba)},
            "supporting_terms": supporting,
            "opposing_terms": opposing,
            "word_scores": words,
            "clean_text": clean,
            "known_terms": int(X.nnz),
        }


if __name__ == "__main__":
    claim = " ".join(sys.argv[1:]) or "Says the unemployment rate has been cut in half since 2010."
    for m in ("binary", "multiclass"):
        if models_available(m):
            r = ClaimClassifier(m).predict(claim)
            print(f"[{m}] {r['label']}  ({r['confidence']:.1%})")
