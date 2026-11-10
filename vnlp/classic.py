"""Traditional ML: TF-IDF + Logistic Regression / linear SVM for classification, CRF for NER."""
import joblib
import numpy as np
import sklearn_crfsuite
from seqeval.metrics import f1_score as span_f1
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000)),
    ])


def make(kind: str, C: float, balanced: bool):
    cw = "balanced" if balanced else None
    if kind == "lr":
        clf = LogisticRegression(C=C, max_iter=3000, class_weight=cw)
    else:  # calibrated so the SVM also returns a usable confidence
        clf = CalibratedClassifierCV(LinearSVC(C=C, class_weight=cw), cv=5, method="sigmoid")
    return make_pipeline(tfidf(), clf)


def fit_select(kind: str, X_tr, y_tr, X_va, y_va):
    """Grid over C and class weighting, chosen on validation macro-F1."""
    best = None
    for C in ([0.5, 2, 8, 32] if kind == "lr" else [0.05, 0.2, 1]):
        for balanced in (False, True):
            m = make(kind, C, balanced).fit(X_tr, y_tr)
            score = f1_score(y_va, m.predict(X_va), average="macro")
            if best is None or score > best[0]:
                best = (score, m, {"C": C, "balanced": balanced})
    print(f"{kind}: val macro-F1={best[0]:.4f} {best[2]}")
    return best[1], best[2]


# ---------- CRF NER ----------

def _word_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    f = {"bias": 1.0, "w.lower": w.lower(), "w.istitle": w.istitle(), "w.isupper": w.isupper(),
         "w.isdigit": w.isdigit(), "w.hasdigit": any(c.isdigit() for c in w), "w.len": min(len(w), 8),
         "w.suf3": w[-3:].lower(), "w.pre3": w[:3].lower(), "w.alpha": w.isalpha()}
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            v = tokens[j]
            f.update({f"{off}.lower": v.lower(), f"{off}.istitle": v.istitle(), f"{off}.hasdigit": any(c.isdigit() for c in v)})
            if off in (-1, 1):
                f[f"bigram{off}"] = (v + "|" + w if off < 0 else w + "|" + v).lower()
        else:
            f[f"{off}.pad"] = True
    return f


def sent_features(tokens: list[str]) -> list[dict]:
    return [_word_features(tokens, i) for i in range(len(tokens))]


def fit_crf(train: list[dict], val: list[dict]):
    Xtr, ytr = [sent_features(s["tokens"]) for s in train], [s["tags"] for s in train]
    Xva, yva = [sent_features(s["tokens"]) for s in val], [s["tags"] for s in val]
    best = None
    for c1 in (0.05, 0.2):
        for c2 in (0.01, 0.1):
            crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=200, all_possible_transitions=True)
            crf.fit(Xtr, ytr)
            score = span_f1(yva, crf.predict(Xva))
            if best is None or score > best[0]:
                best = (score, crf, {"c1": c1, "c2": c2})
    print(f"crf: val span-F1={best[0]:.4f} {best[2]}")
    return best[1], best[2]


def save(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def proba(model, texts) -> np.ndarray:
    return model.predict_proba(list(texts))
