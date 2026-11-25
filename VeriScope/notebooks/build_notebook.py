"""Generates notebooks/VeriScope_Walkthrough.ipynb (no nbformat dependency)."""
import json
from pathlib import Path

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)})

md("# 🔍 VeriScope - Fake News & Claim Classification\nStep-by-step walkthrough of the full NLP pipeline (LIAR dataset).\n\n**Run from the project root or from this folder** - the first cell fixes the path.")
code("""import os, sys
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.getcwd())
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from config import *""")

md("## 1. Dataset collection\nLIAR: ~12.8k short political statements fact-checked by PolitiFact into 6 labels. Run `python -m src.download_data` once (or place `train/valid/test.tsv` in `data/raw/`).")
code("""from src.prepare_data import load_raw
raw = load_raw()
print(raw.shape)
raw[['label','statement','subject','speaker','party','split']].head()""")
code("raw['label'].value_counts().reindex(LABEL_ORDER)")

md("## 2. Data cleaning\nMissing values, invalid labels, duplicates (kept first so train claims never leak into test).")
code("""from src.prepare_data import clean
print('Missing values before cleaning:'); print(raw.isna().sum()[raw.isna().sum() > 0])
df, log = clean(raw)
log""")

md("## 3. Text preprocessing\nlowercase → remove URLs/HTML/symbols → numbers → `num` → tokenise → stop-words (negations kept) → stemming")
code("""from src.preprocess import clean_text, tokenize, remove_stopwords, stem_tokens, preprocess_text
s = "Says the U.S. unemployment rate didn't fall by 50% since 2010 - see http://x.com!"
c = clean_text(s); t = tokenize(c); n = remove_stopwords(t)
print('original :', s); print('cleaned  :', c); print('tokens   :', t)
print('no stops :', n); print('stemmed  :', stem_tokens(n)); print('final    :', preprocess_text(s))""")

md("## 4. Exploratory data analysis")
code("""fig, ax = plt.subplots(1, 3, figsize=(17, 4))
df['label'].value_counts().reindex(LABEL_ORDER).plot.bar(ax=ax[0], color='steelblue'); ax[0].set_title('Class distribution')
df['n_words_raw'].plot.hist(bins=40, ax=ax[1]); ax[1].set_title('Claim length (words)')
df['binary_label'].value_counts().plot.pie(autopct='%1.0f%%', ax=ax[2]); ax[2].set_ylabel(''); ax[2].set_title('Fake vs Credible')
plt.tight_layout(); plt.show()""")
code("""from collections import Counter
for grp in [BINARY_FAKE, BINARY_REAL]:
    words = Counter(w for t in df.loc[df.binary_label == grp, 'clean_text'] for w in t.split())
    print(grp, '->', [w for w, _ in words.most_common(15)])""")
md("More charts (speakers, subjects, parties, missing values) are produced by `python -m src.eda` into `reports/figures/`.")

md("## 5. Feature engineering\nCompare **Bag-of-Words**, **TF-IDF (unigram)** and **TF-IDF (1-2 gram)**.")
code("""from src.features import get_vectorizers
tr, va, te = (df[df.split == s] for s in ['train', 'valid', 'test'])
for name, vec in get_vectorizers().items():
    X = vec.fit_transform(tr.clean_text)
    print(f'{name:<20} matrix {X.shape}  sparsity {1 - X.nnz / np.prod(X.shape):.2%}')
vec = get_vectorizers()['TF-IDF (1-2 gram)']
Xtr = vec.fit_transform(tr.clean_text); Xva = vec.transform(va.clean_text); Xte = vec.transform(te.clean_text)""")

md("## 6. Model development & evaluation (binary: Fake vs Credible)")
code("""from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.train import get_models, metrics
rows = []
for name, m in get_models().items():
    m.fit(Xtr, tr.binary_label)
    rows.append({'model': name, **metrics(te.binary_label, m.predict(Xte))})
pd.DataFrame(rows).set_index('model').round(3).sort_values('f1_macro', ascending=False)""")
code("""best = get_models()['Logistic Regression'].fit(Xtr, tr.binary_label)
pred = best.predict(Xte)
print(classification_report(te.binary_label, pred))
ConfusionMatrixDisplay.from_predictions(te.binary_label, pred, cmap='Blues'); plt.show()""")

md("## 7. Full experiment (all models × feature sets × both modes)\n`python -m src.train` runs the full grid, picks the best by validation macro-F1 and saves the models used by the GUI.")

md("## 8. Try the saved model\n(needs `python -m src.train` to have been run)")
code("""from src.predict import ClaimClassifier, models_available
if models_available('binary'):
    clf = ClaimClassifier('binary')
    r = clf.predict('Says the unemployment rate has been cut in half since 2010.')
    print(r['label'], f"{r['confidence']:.1%}"); print('supporting:', r['supporting_terms'][:5])
else:
    print('Run: python -m src.train')""")

md("## 9. GUI\nLaunch with `streamlit run app.py`.")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "VeriScope_Walkthrough.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
