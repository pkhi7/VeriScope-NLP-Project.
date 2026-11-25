"""Download the LIAR dataset (PolitiFact claims, 6 credibility labels).

Usage:  python -m src.download_data

If the automatic download fails (no internet / link moved), download manually:
  1. Open https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
     (or search "LIAR dataset" on Kaggle / HuggingFace)
  2. Extract train.tsv, valid.tsv, test.tsv into  data/raw/
"""
import io
import sys
import urllib.request
import zipfile

from config import DATA_RAW

URLS = [
    "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
    "http://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
]
NEEDED = ["train.tsv", "valid.tsv", "test.tsv"]


def already_present() -> bool:
    return all((DATA_RAW / f).exists() for f in NEEDED)


def main() -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if already_present():
        print("LIAR files already present in data/raw/ - nothing to do.")
        return 0
    for url in URLS:
        try:
            print(f"Downloading {url} ...")
            with urllib.request.urlopen(url, timeout=60) as r:
                payload = r.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                for name in z.namelist():
                    base = name.split("/")[-1]
                    if base in NEEDED:
                        (DATA_RAW / base).write_bytes(z.read(name))
                        print("  extracted", base)
            if already_present():
                print("Done. Dataset saved to data/raw/")
                return 0
        except Exception as exc:  # noqa: BLE001
            print("  failed:", exc)
    print("\nAutomatic download failed. Please download manually:\n" + __doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
