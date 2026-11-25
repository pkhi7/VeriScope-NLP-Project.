"""Central configuration for VeriScope (Fake News / Claim Classification)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Column names of the original LIAR dataset (train.tsv / valid.tsv / test.tsv)
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state",
    "party", "barely_true_ct", "false_ct", "half_true_ct", "mostly_true_ct",
    "pants_fire_ct", "context",
]

# The six credibility categories used by PolitiFact, ordered false -> true
LABEL_ORDER = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]
LABEL_DISPLAY = {
    "pants-fire": "Pants on Fire",
    "false": "False",
    "barely-true": "Barely True",
    "half-true": "Half True",
    "mostly-true": "Mostly True",
    "true": "True",
}

# Collapsed two-class view
BINARY_FAKE = "Fake / Misleading"
BINARY_REAL = "Credible / Mostly True"
BINARY_MAP = {
    "pants-fire": BINARY_FAKE, "false": BINARY_FAKE, "barely-true": BINARY_FAKE,
    "half-true": BINARY_REAL, "mostly-true": BINARY_REAL, "true": BINARY_REAL,
}

MODES = {
    "multiclass": "6 credibility categories",
    "binary": "Fake vs Credible (2 classes)",
}
