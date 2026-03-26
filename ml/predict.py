from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")

# Minimal built-in stopword list (keeps the app functional without NLTK corpora).
_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def _porter_stem(word: str) -> str:
    """
    Pure-Python Porter stemmer.

    This is a compact implementation of the classic Porter stemming algorithm.
    """

    if len(word) <= 2:
        return word

    word = word.lower()

    vowels = set("aeiou")

    def is_consonant(i: int) -> bool:
        ch = word[i]
        if ch in vowels:
            return False
        if ch == "y":
            return i == 0 or not is_consonant(i - 1)
        return True

    def measure() -> int:
        m = 0
        i = 0
        n = len(word)
        while i < n:
            while i < n and is_consonant(i):
                i += 1
            i += 1
            while i < n and not is_consonant(i):
                i += 1
            m += 1
            i += 1
        return m

    def contains_vowel() -> bool:
        for i in range(len(word)):
            if not is_consonant(i):
                return True
        return False

    def ends_with(suffix: str) -> bool:
        return word.endswith(suffix)

    def replace_suffix(suffix: str, replacement: str) -> None:
        nonlocal word
        if word.endswith(suffix):
            word = word[: -len(suffix)] + replacement

    # Step 1a
    if ends_with("sses"):
        replace_suffix("sses", "ss")
    elif ends_with("ies"):
        replace_suffix("ies", "i")
    elif ends_with("ss"):
        pass
    elif ends_with("s"):
        replace_suffix("s", "")

    # Step 1b
    if ends_with("eed"):
        if measure() > 0:
            replace_suffix("eed", "ee")
    elif (ends_with("ed") and contains_vowel()) or (ends_with("ing") and contains_vowel()):
        if ends_with("ed"):
            replace_suffix("ed", "")
        else:
            replace_suffix("ing", "")
        if ends_with("at") or ends_with("bl") or ends_with("iz"):
            word = word + "e"
        elif len(word) >= 2 and word[-1] == word[-2] and (word[-1] not in vowels):
            word = word[:-1]
        elif measure() == 1 and _ends_cvc(word):
            word = word + "e"

    # Step 1c
    if ends_with("y") and contains_vowel():
        word = word[:-1] + "i"

    # Step 2
    step2_suffixes = {
        "ational": "ate",
        "tional": "tion",
        "enci": "ence",
        "anci": "ance",
        "izer": "ize",
        "abli": "able",
        "alli": "al",
        "entli": "ent",
        "eli": "e",
        "ousli": "ous",
        "ization": "ize",
        "ation": "ate",
        "ator": "ate",
        "alism": "al",
        "iveness": "ive",
        "fulness": "ful",
        "ousness": "ous",
        "aliti": "al",
        "iviti": "ive",
        "biliti": "ble",
    }
    for suf, rep in step2_suffixes.items():
        if ends_with(suf):
            if measure() > 0:
                word = word[: -len(suf)] + rep
            break

    # Step 3
    step3_suffixes = {
        "icate": "ic",
        "ative": "",
        "alize": "al",
        "iciti": "ic",
        "ical": "ic",
        "ful": "",
        "ness": "",
    }
    for suf, rep in step3_suffixes.items():
        if ends_with(suf):
            if measure() > 0:
                word = word[: -len(suf)] + rep
            break

    # Step 4
    step4_suffixes = {
        "al",
        "ance",
        "ence",
        "er",
        "ic",
        "able",
        "ible",
        "ant",
        "ement",
        "ment",
        "ent",
        "ion",
        "ou",
        "ism",
        "ate",
        "iti",
        "ous",
        "ive",
        "ize",
    }
    for suf in step4_suffixes:
        if ends_with(suf):
            if measure() > 1:
                if suf == "ion" and len(word) >= 4:
                    if word[-4] not in ("s", "t"):
                        break
                word = word[: -len(suf)]
            break

    # Step 5a
    if ends_with("e"):
        if measure() > 1:
            word = word[:-1]
        elif measure() == 1 and not _ends_cvc(word[:-1]):
            word = word[:-1]

    # Step 5b
    if measure() > 1 and word.endswith("ll"):
        word = word[:-1]

    return word


def _ends_cvc(w: str) -> bool:
    """
    True if w ends with consonant-vowel-consonant and the last consonant
    is not w/x/y.
    """
    if len(w) < 3:
        return False
    vowels = set("aeiou")

    def is_consonant(i: int) -> bool:
        ch = w[i]
        if ch in vowels:
            return False
        if ch == "y":
            return i == 0 or not is_consonant(i - 1)
        return True

    i = len(w) - 1
    c1 = is_consonant(i - 2)
    v = not is_consonant(i - 1)
    c2 = is_consonant(i)
    if c1 and v and c2:
        if w[-1] not in ("w", "x", "y"):
            return True
    return False


def preprocess_to_tokens(text: str) -> list[str]:
    text = (text or "").lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text)
    raw_tokens = [t for t in text.split() if t]

    out: list[str] = []
    for t in raw_tokens:
        if t in _STOPWORDS:
            continue
        out.append(_porter_stem(t))
    return out


@dataclass(frozen=True)
class Prediction:
    verdict: str  # "SPAM" | "HAM"
    confidence: float  # 0..1
    elapsed_ms: float
    top_tokens: list[tuple[str, float]]


class Predictor:
    def __init__(self, model_path: str | os.PathLike = "ml/model.json"):
        self.model_path = Path(model_path)
        self.model: dict | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Missing model file at {self.model_path}. Run `python -m ml.train`."
            )
        with self.model_path.open("r", encoding="utf-8") as f:
            self.model = json.load(f)

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        assert self.model is not None
        vocab = self.model["vocab"]
        idf = self.model["idf"]

        counts = Counter(tokens)
        if not counts:
            return {}

        # TF (raw count) * IDF, then optional L2 normalization for stability.
        vec: dict[str, float] = {}
        for term, c in counts.items():
            if term not in vocab:
                continue
            vec[term] = float(c) * float(idf.get(term, 0.0))

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] = vec[k] / norm
        return vec

    def predict(self, subject: str, body: str) -> Prediction:
        if not self.ready:
            self.load()
        assert self.model is not None

        full_text = f"{subject or ''} {body or ''}".strip()
        tokens = preprocess_to_tokens(full_text)

        start = time.perf_counter()
        x = self._tfidf(tokens)

        priors = self.model["priors"]  # { "SPAM": logprior, "HAM": logprior }
        log_theta = self.model["log_theta"]  # { "SPAM": {term: logtheta}, ... }

        spam_log = float(priors["SPAM"])
        ham_log = float(priors["HAM"])

        default_spam = float(self.model["default_log_theta"]["SPAM"])
        default_ham = float(self.model["default_log_theta"]["HAM"])

        for term, val in x.items():
            spam_log += val * float(log_theta["SPAM"].get(term, default_spam))
            ham_log += val * float(log_theta["HAM"].get(term, default_ham))

        # Posterior via log-sum-exp
        m = max(spam_log, ham_log)
        p_spam = math.exp(spam_log - m)
        p_ham = math.exp(ham_log - m)
        denom = p_spam + p_ham
        prob_spam = p_spam / denom if denom else 0.5
        prob_ham = 1.0 - prob_spam

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        verdict = "SPAM" if prob_spam >= 0.5 else "HAM"
        confidence = max(prob_spam, prob_ham)

        # Contribution score based on log-odds.
        top_tokens: list[tuple[str, float]] = []
        if x:
            contributions: list[tuple[str, float]] = []
            for term, val in x.items():
                a = float(log_theta["SPAM"].get(term, default_spam))
                b = float(log_theta["HAM"].get(term, default_ham))
                contributions.append((term, val * (a - b)))
            contributions.sort(key=lambda t: abs(t[1]), reverse=True)
            top_tokens = [(t, float(score)) for t, score in contributions[:8]]

        return Prediction(
            verdict=verdict,
            confidence=float(confidence),
            elapsed_ms=float(elapsed_ms),
            top_tokens=top_tokens,
        )

