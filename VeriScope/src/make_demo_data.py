"""Create a SMALL SYNTHETIC dataset in LIAR format - ONLY for a quick dry-run of the code.

   >>> It is NOT real data. Use `python -m src.download_data` for the real LIAR dataset. <<<

Usage: python -m src.make_demo_data
"""
import random

import pandas as pd

from config import DATA_RAW, LABEL_ORDER, LIAR_COLUMNS, RANDOM_STATE

random.seed(RANDOM_STATE)
SUBJECTS = ["economy", "health-care", "education", "taxes", "immigration", "energy", "jobs", "elections"]
SPEAKERS = ["jane-doe", "john-roe", "alex-smith", "sam-lee", "chris-park", "viral-posts"]
PARTIES = ["republican", "democrat", "none", "independent"]
SUBJ_TXT = {
    "economy": "the economy", "health-care": "health care costs", "education": "school funding",
    "taxes": "federal taxes", "immigration": "border policy", "energy": "energy prices",
    "jobs": "unemployment", "elections": "voter turnout",
}
BASE = ["Says {t} has changed by {n} percent since {y}.", "{t} is a major issue for families in this state.",
        "The bill will affect {t} for the next {n} years.", "Our plan on {t} was approved last {m}."]
FLAVOR = {  # words that correlate with credibility in this synthetic set only
    "pants-fire": ["always", "everyone knows", "total hoax", "secretly", "never", "completely destroyed"],
    "false": ["always", "never", "nobody", "disaster", "hidden", "fake"],
    "barely-true": ["some say", "almost", "mostly ignored", "claims", "reportedly"],
    "half-true": ["partly", "according to estimates", "about", "roughly", "projected"],
    "mostly-true": ["according to the report", "data shows", "official figures", "audit found", "approximately"],
    "true": ["according to the census", "official records show", "budget office confirmed", "audited figures", "data shows"],
}


def make(n: int, split: str, start: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        lab = random.choice(LABEL_ORDER)
        subj = random.choice(SUBJECTS)
        s = random.choice(BASE).format(t=SUBJ_TXT[subj], n=random.randint(2, 90),
                                       y=random.randint(1995, 2020), m=random.choice(["year", "month", "spring"]))
        if random.random() < 0.8:
            s += " " + random.choice(FLAVOR[lab]).capitalize() + "."
        if random.random() < 0.25:  # label noise, as in real fact-check data
            s = s.replace(s.split()[0], random.choice(["Says", "Claims", "Reports"]), 1)
        rows.append([f"{start + i}.json", lab, s, subj, random.choice(SPEAKERS), "analyst", "unknown",
                     random.choice(PARTIES), 0, 0, 0, 0, 0, "a speech"])
    return pd.DataFrame(rows, columns=LIAR_COLUMNS)


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for split, n, start in [("train", 3500, 0), ("valid", 450, 10000), ("test", 450, 20000)]:
        make(n, split, start).to_csv(DATA_RAW / f"{split}.tsv", sep="\t", header=False, index=False)
    print("Synthetic DEMO data written to data/raw/ (not real!).")


if __name__ == "__main__":
    main()
