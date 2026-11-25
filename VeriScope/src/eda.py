"""Exploratory Data Analysis -> figures saved in reports/figures/ and eda_summary.json."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import (BINARY_FAKE, BINARY_REAL, DATA_PROCESSED, FIG_DIR,
                    LABEL_DISPLAY, LABEL_ORDER, REPORTS_DIR)

sns.set_theme(style="whitegrid", context="talk", font_scale=0.7)
PAL = dict(zip(LABEL_ORDER, ["#b71c1c", "#e53935", "#fb8c00", "#fdd835", "#7cb342", "#2e7d32"]))


def _save(name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def top_words(texts, n=15):
    c = Counter(w for t in texts for w in t.split())
    return c.most_common(n)


def main():
    df = pd.read_csv(DATA_PROCESSED / "liar_clean.csv")
    log = json.loads((REPORTS_DIR / "cleaning_log.json").read_text())

    # 1. class distribution
    plt.figure(figsize=(8, 4.5))
    counts = df["label"].value_counts().reindex(LABEL_ORDER)
    ax = sns.barplot(x=[LABEL_DISPLAY[l] for l in counts.index], y=counts.values,
                     palette=[PAL[l] for l in counts.index], hue=[LABEL_DISPLAY[l] for l in counts.index], legend=False)
    for p, v in zip(ax.patches, counts.values):
        ax.annotate(f"{v:,}", (p.get_x() + p.get_width() / 2, v), ha="center", va="bottom", fontsize=9)
    plt.title("Claims per credibility category"); plt.xlabel(""); plt.ylabel("Number of claims")
    plt.xticks(rotation=15)
    _save("01_class_distribution.png")

    # 2. binary split
    plt.figure(figsize=(5, 4.5))
    b = df["binary_label"].value_counts()
    plt.pie(b.values, labels=b.index, autopct="%1.1f%%", colors=["#e53935", "#43a047"] if b.index[0] == BINARY_FAKE else ["#43a047", "#e53935"],
            startangle=90, wedgeprops=dict(edgecolor="white"))
    plt.title("Fake/Misleading vs Credible")
    _save("02_binary_split.png")

    # 3. length distribution
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x=df["label"].map(LABEL_DISPLAY), y="n_words_raw", order=[LABEL_DISPLAY[l] for l in LABEL_ORDER],
                palette=[PAL[l] for l in LABEL_ORDER], hue=df["label"].map(LABEL_DISPLAY), legend=False, showfliers=False)
    plt.title("Claim length (words) by category"); plt.xlabel(""); plt.ylabel("Words per claim"); plt.xticks(rotation=15)
    _save("03_length_by_label.png")

    # 4. raw vs clean length
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["n_words_raw"], bins=40, color="#1e88e5", label="Before cleaning", alpha=0.6)
    sns.histplot(df["n_words_clean"], bins=40, color="#fb8c00", label="After preprocessing", alpha=0.6)
    plt.legend(); plt.title("Claim length before vs after preprocessing"); plt.xlabel("Words")
    _save("04_length_before_after.png")

    # 5. top words fake vs credible
    fake = top_words(df.loc[df["binary_label"] == BINARY_FAKE, "clean_text"])
    real = top_words(df.loc[df["binary_label"] == BINARY_REAL, "clean_text"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, col in [(axes[0], fake, "Fake / Misleading", "#e53935"), (axes[1], real, "Credible / Mostly True", "#43a047")]:
        w, c = zip(*data)
        ax.barh(w[::-1], c[::-1], color=col); ax.set_title(f"Top words - {title}")
    _save("05_top_words.png")

    # 6. top subjects & speakers
    subj = df["subject"].str.split(",").explode().str.strip().value_counts().head(10)
    spk = df[df["speaker"] != "unknown"]["speaker"].value_counts().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(subj.index[::-1], subj.values[::-1], color="#5e35b1"); axes[0].set_title("Top 10 subjects")
    axes[1].barh(spk.index[::-1], spk.values[::-1], color="#00897b"); axes[1].set_title("Top 10 speakers")
    _save("06_subjects_speakers.png")

    # 7. party vs label share
    top_parties = df["party"].value_counts().head(6).index
    ct = pd.crosstab(df[df["party"].isin(top_parties)]["party"], df["binary_label"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), color=["#43a047", "#e53935"] if ct.columns[0] == BINARY_REAL else ["#e53935", "#43a047"])
    plt.ylabel("% of claims"); plt.title("Credibility mix by party label"); plt.xticks(rotation=20); plt.xlabel("")
    _save("07_party_vs_label.png")

    # 8. missing values (before cleaning)
    miss = pd.Series(log["missing_before"]); miss = miss[miss > 0].sort_values()
    if len(miss):
        plt.figure(figsize=(8, 4.5)); plt.barh(miss.index, miss.values, color="#8d6e63")
        plt.title("Missing values per column (before cleaning)"); _save("08_missing_values.png")

    summary = {
        "n_claims": int(len(df)),
        "n_speakers": int(df["speaker"].nunique()),
        "avg_words_raw": round(float(df["n_words_raw"].mean()), 2),
        "avg_words_clean": round(float(df["n_words_clean"].mean()), 2),
        "class_counts": {LABEL_DISPLAY[k]: int(v) for k, v in counts.items()},
        "binary_counts": {k: int(v) for k, v in df["binary_label"].value_counts().items()},
        "vocabulary_size": len({w for t in df["clean_text"] for w in t.split()}),
    }
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
