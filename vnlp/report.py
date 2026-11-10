"""Evaluate every saved model the same way and write docs/REPORT.md.

Topic routing policy: auto-route when max softmax probability >= tau, else send to a human.
tau is chosen on validation as the lowest value whose auto-routed accuracy is >= TARGET_ACC.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from seqeval.metrics import classification_report as span_report
from seqeval.metrics import f1_score as span_f1
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score)

from vnlp.data import SENTIMENTS, TOPICS, load_ner, load_ood, load_vsfc

ROOT = Path(__file__).resolve().parent.parent
ART, PRED, DOCS = ROOT / "artifacts", ROOT / "artifacts" / "preds", ROOT / "docs"
TARGET_ACC = 0.95
NAMES = {"tfidf_lr": "TF-IDF + Logistic Regression", "tfidf_svm": "TF-IDF + linear SVM (calibrated)",
         "phobert": "PhoBERT-base-v2 fine-tuned"}


def load(model, task, split):
    p = PRED / f"{model}__{task}__{split}.npy"
    return np.load(p) if p.exists() else None


def pick_threshold(probs: np.ndarray, y: np.ndarray, target: float = TARGET_ACC) -> float:
    conf, correct = probs.max(1), probs.argmax(1) == y
    for tau in np.linspace(0, 1, 201):
        keep = conf >= tau
        if keep.sum() >= 20 and correct[keep].mean() >= target:
            return float(tau)
    return 1.0


def ood_stats(p_in: np.ndarray, p_out: np.ndarray, tau: float) -> dict:
    s_in, s_out = p_in.max(1), p_out.max(1)
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    t95 = np.quantile(s_in, 0.05)  # keeps 95% of in-domain texts
    return {"AUROC": float(roc_auc_score(y, np.r_[s_in, s_out])), "FPR@95TPR": float((s_out >= t95).mean()),
            "ood_auto_routed": float((s_out >= tau).mean())}


def latency(model: str, task: str, texts: list[str]) -> dict:
    if model == "phobert":
        from vnlp.phobert import TopicModel
        m = TopicModel(str(ART / "phobert_topic"))
        f = m.predict_proba
    else:
        import joblib
        f = joblib.load(ART / f"{model}_{task}.joblib").predict_proba
    f(texts[:5])  # warm-up
    single = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        f([t])
        single.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    f(texts[:1000])
    return {"p50_ms": float(np.percentile(single, 50)), "p95_ms": float(np.percentile(single, 95)),
            "batch_texts_per_s": 1000 / (time.perf_counter() - t0)}


def plot_cm(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    va, te, ood = load_vsfc("validation"), load_vsfc("test"), load_ood()
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("vietnamese-nlp")
    L, results, thresholds = [], {}, {}

    L += ["# Evaluation report", "",
          f"UIT-VSFC test set: {len(te)} feedback sentences. OOD set: {len(ood)} texts "
          f"({(ood['source'] == 'banking_news').sum()} banking-news sentences, {(ood['source'] == 'ecommerce_title').sum()} "
          "e-commerce product titles), normalised the same way as the training text.", ""]

    for task, labels in (("topic", TOPICS), ("sentiment", SENTIMENTS)):
        y_va, y_te = va[task].to_numpy(), te[task].to_numpy()
        rows = []
        for model in NAMES:
            p_te = load(model, task, "test")
            if p_te is None:
                continue
            pred = p_te.argmax(1)
            r = {"accuracy": accuracy_score(y_te, pred), "macro_f1": f1_score(y_te, pred, average="macro"),
                 "weighted_f1": f1_score(y_te, pred, average="weighted"),
                 "per_class": classification_report(y_te, pred, target_names=labels, output_dict=True, zero_division=0)}
            p_va = load(model, task, "val")
            tau = pick_threshold(p_va, y_va)
            keep = p_te.max(1) >= tau
            r.update(tau=tau, coverage=float(keep.mean()), auto_accuracy=float((pred[keep] == y_te[keep]).mean()))
            if task == "topic":
                p_ood = load(model, task, "ood")
                r["ood"] = ood_stats(p_te, p_ood, tau)
                for src in ("banking_news", "ecommerce_title"):
                    r["ood"][f"auto_routed_{src}"] = float((p_ood[(ood["source"] == src).to_numpy()].max(1) >= tau).mean())
                thresholds[model] = tau
            r["latency"] = latency(model, task, te["text"].tolist())
            r["cm"] = confusion_matrix(y_te, pred).tolist()
            results[f"{model}_{task}"] = r
            rows.append((model, r))
            with mlflow.start_run(run_name=f"{model}_{task}"):
                mlflow.log_metrics({k: v for k, v in r.items() if isinstance(v, float)})
                mlflow.log_metrics({f"latency_{k}": v for k, v in r["latency"].items()})
                if "ood" in r:
                    mlflow.log_metrics({f"ood_{k}".replace("@", "_at_"): v for k, v in r["ood"].items()})

        if not rows:
            continue
        L += [f"## Task: {task} ({len(labels)} classes)", "",
              "| Model | Accuracy | Macro-F1 | Weighted-F1 | p50 latency (1 text) | p95 | Batch throughput |",
              "|---|---|---|---|---|---|---|"]
        for model, r in sorted(rows, key=lambda x: -x[1]["macro_f1"]):
            lt = r["latency"]
            L.append(f"| {NAMES[model]} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
                     f"{lt['p50_ms']:.1f} ms | {lt['p95_ms']:.1f} ms | {lt['batch_texts_per_s']:.0f} texts/s |")
        best_model, best = max(rows, key=lambda x: x[1]["macro_f1"])
        L += ["", f"Per-class precision / recall / F1, best model ({NAMES[best_model]}):", "",
              "| Class | Support | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for c in labels:
            pc = best["per_class"][c]
            L.append(f"| {c} | {int(pc['support'])} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['f1-score']:.3f} |")
        cm_path = DOCS / f"confusion_{task}.png"
        plot_cm(np.array(best["cm"]), labels, f"{task}: {NAMES[best_model]}", cm_path)
        L += ["", f"![confusion matrix]({cm_path.name})", "",
              f"Human-review policy (threshold chosen on validation for ≥{TARGET_ACC:.0%} accuracy on auto-routed items):", "",
              "| Model | Threshold | Auto-routed share (test) | Accuracy of auto-routed | " +
              ("OOD AUROC | OOD FPR@95TPR | OOD texts auto-routed |" if task == "topic" else ""),
              "|---|---|---|---|" + ("---|---|---|" if task == "topic" else "")]
        for model, r in rows:
            extra = (f" {r['ood']['AUROC']:.3f} | {r['ood']['FPR@95TPR']:.1%} | {r['ood']['ood_auto_routed']:.1%} "
                     f"(news {r['ood']['auto_routed_banking_news']:.0%}, product titles {r['ood']['auto_routed_ecommerce_title']:.0%}) |"
                     if task == "topic" else "")
            L.append(f"| {NAMES[model]} | {r['tau']:.3f} | {r['coverage']:.1%} | {r['auto_accuracy']:.1%} |{extra}")

        if task == "topic":
            p = load(best_model, task, "test")
            pred, conf = p.argmax(1), p.max(1)
            wrong = np.where(pred != y_te)[0]
            wrong = wrong[np.argsort(-conf[wrong])][:12]
            L += ["", f"Most confident errors ({NAMES[best_model]}): these pass the threshold and reach the wrong team.", "",
                  "| Text | True | Predicted | Confidence |", "|---|---|---|---|"]
            for i in wrong:
                L.append(f"| {te['text'][i][:110]} | {labels[y_te[i]]} | {labels[pred[i]]} | {conf[i]:.2f} |")
        L.append("")

    ner_te = load_ner("test")
    gold = [s["tags"] for s in ner_te]
    ner_rows = []
    for model, label in (("crf", "CRF (hand-crafted features)"), ("phobert", "PhoBERT-base-v2 token classification")):
        p = PRED / f"{model}__ner__test.json"
        if p.exists():
            pred = json.loads(p.read_text())
            rep = span_report(gold, pred, output_dict=True, zero_division=0)
            ner_rows.append((label, span_f1(gold, pred), rep))
            results[f"{model}_ner"] = {"micro_f1": span_f1(gold, pred), "macro_f1": rep["macro avg"]["f1-score"]}
    if ner_rows:
        L += ["## Task: NER (ViEcomNER, 24 entity types, span-level, exact match)", "",
              "| Model | Micro-F1 | Macro-F1 |", "|---|---|---|"]
        for label, f1, rep in ner_rows:
            L.append(f"| {label} | {f1:.4f} | {rep['macro avg']['f1-score']:.4f} |")
        label, _, rep = max(ner_rows, key=lambda x: x[1])
        top = sorted((k for k in rep if k not in ("micro avg", "macro avg", "weighted avg")), key=lambda k: -rep[k]["support"])[:10]
        L += ["", f"Ten most frequent entity types, {label}:", "", "| Entity | Support | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for k in top:
            L.append(f"| {k} | {int(rep[k]['support'])} | {rep[k]['precision']:.3f} | {rep[k]['recall']:.3f} | {rep[k]['f1-score']:.3f} |")

    (DOCS / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ART / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (ART / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
