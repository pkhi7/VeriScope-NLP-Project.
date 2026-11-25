# Presentation & Viva Guide – VeriScope (Group 6)

## 0. Before the day (checklist)
- [ ] `python run_pipeline.py` done on the **real** LIAR data (no “synthetic data” banner in the GUI)
- [ ] Fill the results table in README.md; take GUI screenshots into `screenshots/`
- [ ] Push to GitHub, test the repo link works in a private window
- [ ] Test `streamlit run app.py` on the demo laptop; keep one browser tab pre-opened
- [ ] Every member can explain every stage (the guidelines require it)

## 1. Suggested split of the 5 stages (adapt to your group size)
| Member | Stage | Files they should know cold |
|---|---|---|
| 1 | Dataset & EDA | `prepare_data.py` (load), `eda.py`, EDA tab |
| 2 | Cleaning & preprocessing | `preprocess.py`, `prepare_data.py` (clean) |
| 3 | Feature engineering | `features.py`, TF-IDF maths |
| 4 | Models & evaluation | `train.py`, metrics, confusion matrix |
| 5 | GUI & integration | `app.py`, `predict.py` |
| 6 | PPT, README, GitHub, intro/conclusion | everything at overview level |

## 2. 10-minute flow
1. **Problem (1 min)** – misinformation is a real problem; we classify claims into credibility categories. *Say clearly: we predict a label from text patterns, not truth.*
2. **Dataset (1 min)** – LIAR, PolitiFact, 6 labels, show class chart.
3. **Cleaning & preprocessing (1.5 min)** – use the *“original → processed”* example in the Dataset tab.
4. **Features (1 min)** – BoW vs TF-IDF vs bigrams, why TF-IDF.
5. **Models & results (2 min)** – comparison chart, best model, confusion matrix, baseline.
6. **LIVE DEMO (3 min)** – see script below.
7. **Limitations & future scope (30 s).**

## 3. Live demo script
1. Sidebar → show chosen model & metrics.
2. Click **Example 3** → **Analyze Claim** → point at the label, confidence, probability chart.
3. Point at the **highlighted words** – “green words pushed towards this prediction, red against.”
4. Type your own claim (prepare 2 beforehand). Show the *preprocessing* expander.
5. Switch to **Binary** mode, analyze the same claim – “simpler task, higher accuracy.”
6. Show **Model Performance** and **EDA** tabs briefly.
7. Show session history → download CSV.

## 4. Likely questions & strong answers
**Why is 6-class accuracy so low?** LIAR is intentionally hard: statements are short, labels are subjective, neighbouring classes overlap. Published text-only baselines are also ~25–28 %. Random guessing is 16.7 %, so the model learns real signal. Binary mode is much easier.

**Doesn’t that mean the model is useless?** It’s a decision-support prototype, not a fact-checker. It shows how to build the pipeline; better scores need evidence retrieval and transformers (our future scope).

**Why TF-IDF over BoW?** BoW counts treat frequent words as important; TF-IDF down-weights words common across all documents (idf = log(N/df)) so distinctive words matter more. We tested both – see comparison chart.

**Why keep negations when removing stop-words?** “not”, “never”, “no” change meaning and are strong credibility cues.

**Why stemming not lemmatization?** Porter stemming needs no extra downloads and is fast; lemmatization (WordNet) is an easy extension.

**How did you avoid data leakage?** Removed duplicate claims (train copy kept), vectorizer fit on train only, model chosen on validation, final numbers on untouched test.

**Why macro-F1?** Classes are imbalanced; macro-F1 treats every class equally. Accuracy alone can hide poor minority-class performance.

**What is precision vs recall here?** Precision: of claims we flagged as fake, how many truly were. Recall: of truly fake claims, how many we caught.

**How does the explanation work?** For a linear model, score = coefficient × TF-IDF value per word; positive = supports predicted class.

**Why did you use Linear SVM + calibration?** SVMs are strong for sparse text, but have no probabilities; `CalibratedClassifierCV` converts margins to probabilities for the confidence display.

**Why not BERT?** Brief scope was classical NLP + TF-IDF; BERT is a planned extension (note Project 10 in the course uses BERT).

**Is it ethical?** We label the output a prediction, show confidence, add disclaimers, and keep a human in the loop.

**What would you improve?** Transformers, speaker/context metadata (known to help on LIAR), evidence retrieval, more recent data.

## 5. Things NOT to claim
- ❌ “Our model detects fake news with X % accuracy” without saying *which task / dataset*.
- ❌ “It tells whether a statement is true.”
- ✅ “It predicts the PolitiFact-style credibility label from the text of the claim.”
