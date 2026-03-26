from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from collections import Counter, defaultdict

from ml.predict import preprocess_to_tokens


def load_dataset(csv_path: Path) -> list[tuple[str, str]]:
    """
    Load dataset rows as (label, text) using the SpamAssassin/UCI format:
    - label in v1: 'spam' or 'ham'
    - text in v2
    """
    rows: list[tuple[str, str]] = []
    with csv_path.open("r", encoding="latin-1", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) >= 2 and header[0].strip().lower() in {"v1", "label"}:
            for r in reader:
                label = (r[0] or "").strip().lower()
                text = r[1] if len(r) > 1 else ""
                if label in {"spam", "ham"}:
                    rows.append((label, text))
        else:
            # No header case
            if header and len(header) >= 2:
                first_label = (header[0] or "").strip().lower()
                first_text = header[1]
                if first_label in {"spam", "ham"}:
                    rows.append((first_label, first_text))
            for r in reader:
                if not r:
                    continue
                label = (r[0] or "").strip().lower()
                text = r[1] if len(r) > 1 else ""
                if label in {"spam", "ham"}:
                    rows.append((label, text))
    return rows


def train_model(tokens_by_doc: list[tuple[str, list[str]]], alpha: float = 0.1) -> dict:
    labels = [lab for lab, _ in tokens_by_doc]
    n_docs = len(tokens_by_doc)
    if n_docs == 0:
        raise ValueError("Dataset is empty.")

    # Build vocabulary + document frequencies.
    df: Counter[str] = Counter()
    vocab_set: set[str] = set()
    for _, tokens in tokens_by_doc:
        uniq = set(tokens)
        for t in uniq:
            df[t] += 1
            vocab_set.add(t)

    vocab = sorted(vocab_set)
    V = len(vocab)
    if V == 0:
        raise ValueError("Vocabulary is empty after preprocessing.")

    idf: dict[str, float] = {}
    for term in vocab:
        # Smooth IDF
        idf[term] = math.log((n_docs + 1) / (df[term] + 1)) + 1.0

    # TF-IDF transform (must match predictor implementation).
    def tfidf(tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        vec: dict[str, float] = {}
        for term, c in counts.items():
            if term not in idf:
                continue
            vec[term] = float(c) * float(idf[term])
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            for k in list(vec.keys()):
                vec[k] = vec[k] / norm
        return vec

    # Accumulate feature totals per class.
    class_docs: dict[str, list[dict[str, float]]] = {"SPAM": [], "HAM": []}
    for label, tokens in tokens_by_doc:
        cls = "SPAM" if label == "spam" else "HAM"
        class_docs[cls].append(tfidf(tokens))

    priors = {c: math.log(len(class_docs[c]) / n_docs) for c in ["SPAM", "HAM"]}

    log_theta: dict[str, dict[str, float]] = {"SPAM": {}, "HAM": {}}
    default_log_theta: dict[str, float] = {"SPAM": 0.0, "HAM": 0.0}

    for cls in ["SPAM", "HAM"]:
        feature_totals: defaultdict[str, float] = defaultdict(float)
        total_sum = 0.0
        for vec in class_docs[cls]:
            for term, val in vec.items():
                feature_totals[term] += val
                total_sum += val

        denom = total_sum + alpha * V
        # theta_default is what we'd get for unseen terms (feature_total=0).
        default_theta = alpha / denom
        default_log_theta[cls] = math.log(default_theta)

        for term in vocab:
            theta = (feature_totals.get(term, 0.0) + alpha) / denom
            log_theta[cls][term] = math.log(theta)

    return {
        "vocab": vocab,
        "idf": idf,
        "priors": priors,
        "log_theta": log_theta,
        "default_log_theta": default_log_theta,
        "alpha": alpha,
    }


def predict_with_model(model: dict, tokens: list[str]) -> tuple[str, float]:
    vocab_set = set(model["vocab"])
    idf = model["idf"]
    counts = Counter(tokens)
    vec: dict[str, float] = {}
    for term, c in counts.items():
        if term not in vocab_set:
            continue
        vec[term] = float(c) * float(idf.get(term, 0.0))
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for k in list(vec.keys()):
            vec[k] = vec[k] / norm

    priors = model["priors"]
    log_theta = model["log_theta"]
    default_log_theta = model["default_log_theta"]

    spam_log = float(priors["SPAM"])
    ham_log = float(priors["HAM"])

    for term, val in vec.items():
        spam_log += val * float(log_theta["SPAM"].get(term, default_log_theta["SPAM"]))
        ham_log += val * float(log_theta["HAM"].get(term, default_log_theta["HAM"]))

    m = max(spam_log, ham_log)
    p_spam = math.exp(spam_log - m)
    p_ham = math.exp(ham_log - m)
    denom = p_spam + p_ham
    prob_spam = p_spam / denom if denom else 0.5

    verdict = "SPAM" if prob_spam >= 0.5 else "HAM"
    return verdict, prob_spam


def compute_metrics(y_true: list[str], y_pred: list[str]) -> tuple[float, float, float]:
    # Metrics reported in report: accuracy, precision(spam), recall(spam).
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "SPAM" and p == "SPAM")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "HAM" and p == "SPAM")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "SPAM" and p == "HAM")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return accuracy, precision, recall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to spam.csv (v1,v2).")
    parser.add_argument(
        "--out",
        default="ml/model.json",
        help="Output path for trained model (pure-Python TF-IDF + NB).",
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="Laplace smoothing alpha.")
    args = parser.parse_args()

    rows = load_dataset(Path(args.dataset))
    if not rows:
        raise ValueError("No valid rows found in dataset.")

    # Stratified split (spam/ham) without sklearn.
    random.seed(42)
    spam_rows = [r for r in rows if r[0] == "spam"]
    ham_rows = [r for r in rows if r[0] == "ham"]

    def split_class(class_rows: list[tuple[str, str]], test_frac: float = 0.2):
        random.shuffle(class_rows)
        cut = int(len(class_rows) * (1 - test_frac))
        return class_rows[:cut], class_rows[cut:]

    spam_train, spam_test = split_class(spam_rows)
    ham_train, ham_test = split_class(ham_rows)

    train_rows = spam_train + ham_train
    test_rows = spam_test + ham_test

    # Preprocess once.
    tokens_train = [(lab, preprocess_to_tokens(text)) for lab, text in train_rows]
    tokens_test = [(lab, preprocess_to_tokens(text)) for lab, text in test_rows]

    model = train_model(tokens_train, alpha=args.alpha)

    # Evaluate
    y_true = []
    y_pred = []
    for lab, tokens in tokens_test:
        true_cls = "SPAM" if lab == "spam" else "HAM"
        pred_cls, _ = predict_with_model(model, tokens)
        y_true.append(true_cls)
        y_pred.append(pred_cls)

    acc, prec, rec = compute_metrics(y_true, y_pred)
    print(f"accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_dumps(model), encoding="utf-8")
    print(f"Wrote {out_path}")


def json_dumps(obj: dict) -> str:
    # Keep JSON stable and reasonably compact.
    import json as _json

    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()

