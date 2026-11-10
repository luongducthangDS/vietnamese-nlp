"""Training steps; each one saves its model and its predictions on val/test/OOD to artifacts/.

    python -m vnlp.train classic       # TF-IDF + LR / SVM (topic, sentiment) and CRF NER, minutes
    python -m vnlp.train phobert       # PhoBERT topic classifier, ~1 h on CPU
    python -m vnlp.train phobert-ner   # PhoBERT NER, ~30 min on CPU
    python -m vnlp.report              # metrics, OOD, latency -> docs/REPORT.md
"""
import json
import sys
from pathlib import Path

import numpy as np

from vnlp import classic
from vnlp.data import load_ner, load_ood, load_vsfc

ART = Path(__file__).resolve().parent.parent / "artifacts"
PRED = ART / "preds"


def save_pred(model: str, task: str, split: str, probs: np.ndarray) -> None:
    PRED.mkdir(parents=True, exist_ok=True)
    np.save(PRED / f"{model}__{task}__{split}.npy", probs.astype("float32"))


def run_classic() -> None:
    tr, va, te, ood = load_vsfc("train"), load_vsfc("validation"), load_vsfc("test"), load_ood()
    params = {}
    for task in ("topic", "sentiment"):
        for kind in ("lr", "svm"):
            m, p = classic.fit_select(kind, tr["text"], tr[task], va["text"], va[task])
            name = f"tfidf_{kind}"
            classic.save(m, ART / f"{name}_{task}.joblib")
            params[f"{name}_{task}"] = p
            for split, df in (("val", va), ("test", te), ("ood", ood)):
                save_pred(name, task, split, classic.proba(m, df["text"]))

    ner_tr, ner_va, ner_te = load_ner("train"), load_ner("validation"), load_ner("test")
    crf, p = classic.fit_crf(ner_tr, ner_va)
    classic.save(crf, ART / "crf_ner.joblib")
    params["crf_ner"] = p
    pred = crf.predict([classic.sent_features(s["tokens"]) for s in ner_te])
    (PRED / "crf__ner__test.json").write_text(json.dumps([[str(t) for t in s] for s in pred]))
    (ART / "classic_params.json").write_text(json.dumps(params, indent=2))


def run_phobert_topic() -> None:
    from vnlp.phobert import TopicModel, seed_all
    seed_all()
    tr, va, te, ood = load_vsfc("train"), load_vsfc("validation"), load_vsfc("test"), load_ood()
    m = TopicModel()
    hist = m.fit(tr["text"], tr["topic"], va["text"], va["topic"].to_numpy())
    m.save(ART / "phobert_topic")
    (ART / "phobert_topic_history.json").write_text(json.dumps(hist, indent=2))
    for split, df in (("val", va), ("test", te), ("ood", ood)):
        save_pred("phobert", "topic", split, m.predict_proba(df["text"]))


def run_phobert_ner() -> None:
    from vnlp.phobert import NerModel, seed_all
    seed_all()
    tr, va, te = load_ner("train"), load_ner("validation"), load_ner("test")
    labels = ["O"] + sorted({t for s in tr for t in s["tags"]} - {"O"})
    m = NerModel(labels)
    hist = m.fit(tr, va)
    m.save(ART / "phobert_ner")
    (ART / "phobert_ner_history.json").write_text(json.dumps(hist, indent=2))
    PRED.mkdir(parents=True, exist_ok=True)
    (PRED / "phobert__ner__test.json").write_text(json.dumps(m.predict([s["tokens"] for s in te])))


if __name__ == "__main__":
    {"classic": run_classic, "phobert": run_phobert_topic, "phobert-ner": run_phobert_ner}[sys.argv[1]]()
