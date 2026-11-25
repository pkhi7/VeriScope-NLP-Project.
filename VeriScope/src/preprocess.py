"""Text cleaning & preprocessing: normalisation, tokenisation, stop-word removal, stemming.

The same function is used for training AND for the live GUI, so predictions
see exactly the same kind of text the model was trained on.
"""
import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations are strong credibility signals ("never", "not") - keep them.
KEEP = {"not", "no", "never", "nor", "none", "nothing", "cannot", "without", "against"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - KEEP

try:  # NLTK's Porter stemmer needs no extra data download
    from nltk.stem import PorterStemmer

    _porter = PorterStemmer()
    _stem = _porter.stem
except Exception:  # pragma: no cover - lightweight fallback if NLTK missing
    def _stem(w: str) -> str:
        for suf, rep in (("ies", "y"), ("ing", ""), ("edly", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                return w[: len(w) - len(suf)] + rep
        return w

_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML = re.compile(r"<[^>]+>")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lower-case and strip URLs, HTML, symbols; map numbers to a 'num' token."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _URL.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = t.replace("n't", " not").replace("’", "'")
    t = t.replace("%", " percent ").replace("$", " dollar ")
    t = _NUM.sub(" num ", t)
    t = re.sub(r"'s\b", "", t)
    t = _NON_ALPHA.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def tokenize(text: str):
    """Regex word tokenizer (alphabetic tokens of length >= 2)."""
    return [w for w in re.findall(r"[a-z]+", text) if len(w) > 1]


def remove_stopwords(tokens):
    return [w for w in tokens if w not in STOP_WORDS]


@lru_cache(maxsize=200_000)
def stem_word(word: str) -> str:
    return _stem(word)


def stem_tokens(tokens):
    return [stem_word(w) for w in tokens]


def preprocess_text(text: str, stem: bool = True) -> str:
    """Full pipeline: clean -> tokenize -> stop-word removal -> stemming -> string."""
    tokens = remove_stopwords(tokenize(clean_text(text)))
    if stem:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    demo = "Says the U.S. unemployment rate didn't fall by 50% since 2010 - see http://x.com!"
    print("RAW  :", demo)
    print("CLEAN:", clean_text(demo))
    print("FINAL:", preprocess_text(demo))
