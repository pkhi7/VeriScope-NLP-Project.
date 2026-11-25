"""Load raw LIAR files -> clean -> preprocess -> save data/processed/liar_clean.csv.

Steps logged to reports/cleaning_log.json (used by the EDA & the GUI 'Dataset' tab).
"""
import json

import pandas as pd

from config import (BINARY_MAP, DATA_PROCESSED, DATA_RAW, LABEL_ORDER,
                    LIAR_COLUMNS, REPORTS_DIR)
from src.preprocess import preprocess_text

SPLITS = ["train", "valid", "test"]


def load_raw() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        path = DATA_RAW / f"{split}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.download_data` first "
                "(or copy train/valid/test.tsv into data/raw/)."
            )
        df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS,
                         quoting=3, on_bad_lines="skip", encoding="utf-8")
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame):
    log = {"rows_raw": int(len(df))}
    log["missing_before"] = {c: int(v) for c, v in df.isna().sum().items()}
    log["rows_per_split_raw"] = {k: int(v) for k, v in df["split"].value_counts().items()}

    # 1. rows without a claim or label are unusable
    df = df.dropna(subset=["statement", "label"]).copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["statement"].str.len() > 0]

    # 2. keep only known labels (guards against malformed rows)
    bad = ~df["label"].isin(LABEL_ORDER)
    log["invalid_label_rows_removed"] = int(bad.sum())
    df = df[~bad]

    # 3. fill optional metadata so nothing downstream breaks
    for col in ["subject", "speaker", "job_title", "state", "party", "context"]:
        df[col] = df[col].fillna("unknown").astype(str)

    # 4. duplicates: identical claims (keep first -> train wins, avoids train/test leakage)
    before = len(df)
    df = df.drop_duplicates(subset="statement", keep="first")
    log["duplicate_claims_removed"] = int(before - len(df))

    # 5. preprocessing
    df["clean_text"] = df["statement"].map(preprocess_text)
    empty = df["clean_text"].str.len() == 0
    log["empty_after_preprocessing_removed"] = int(empty.sum())
    df = df[~empty]

    df["n_words_raw"] = df["statement"].str.split().str.len()
    df["n_words_clean"] = df["clean_text"].str.split().str.len()
    df["binary_label"] = df["label"].map(BINARY_MAP)

    log["rows_final"] = int(len(df))
    log["rows_per_split_final"] = {k: int(v) for k, v in df["split"].value_counts().items()}
    return df.reset_index(drop=True), log


def main(source_note: str = "LIAR (PolitiFact)"):
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    df, log = clean(raw)
    log["dataset_source"] = source_note
    out = DATA_PROCESSED / "liar_clean.csv"
    df.to_csv(out, index=False)
    (REPORTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"\nSaved {len(df):,} cleaned rows -> {out}")


if __name__ == "__main__":
    import os
    main(os.environ.get("DATASET_NOTE", "LIAR (PolitiFact)"))
