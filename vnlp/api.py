"""Feedback triage API: topic -> team, sentiment -> priority, entities, human review when unsure.

    uvicorn vnlp.api:app --port 8001
TOPIC_MODEL = phobert | tfidf_lr | tfidf_svm (default: phobert if trained, else tfidf_lr)
NER_MODEL   = crf | phobert                   (default: crf, ~1 ms on CPU)
"""
import json
import os
import re
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from vnlp.classic import sent_features
from vnlp.data import ROUTES, SENTIMENTS, TOPICS, normalize

ART = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
M: dict = {}
latency: deque = deque(maxlen=5000)
decisions: Counter = Counter()

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?:\+84|0)(?:[\s.-]?\d){9,10}\b")


def spans(tokens: list[str], tags: list[str]) -> list[dict]:
    out, cur = [], None
    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur["label"] != tag[2:])):
            cur = {"label": tag[2:], "text": tok}
            out.append(cur)
        elif tag.startswith("I-"):
            cur["text"] += " " + tok
        else:
            cur = None
    return out


@asynccontextmanager
async def lifespan(_: FastAPI):
    name = os.getenv("TOPIC_MODEL") or ("phobert" if (ART / "phobert_topic").exists() else "tfidf_lr")
    if name == "phobert":
        from vnlp.phobert import TopicModel
        M["topic"] = TopicModel(str(ART / "phobert_topic")).predict_proba
    else:
        M["topic"] = joblib.load(ART / f"{name}_topic.joblib").predict_proba
    M["topic_name"] = name
    M["tau"] = json.loads((ART / "thresholds.json").read_text())[name]
    M["sentiment"] = joblib.load(ART / "tfidf_lr_sentiment.joblib").predict_proba
    ner = os.getenv("NER_MODEL", "crf")
    if ner == "phobert":
        from transformers import AutoConfig

        from vnlp.phobert import NerModel
        cfg = AutoConfig.from_pretrained(ART / "phobert_ner")
        m = NerModel([cfg.id2label[i] for i in range(len(cfg.id2label))], str(ART / "phobert_ner"))
        M["ner"] = lambda toks: m.predict([toks])[0]
    else:
        crf = joblib.load(ART / "crf_ner.joblib")
        M["ner"] = lambda toks: crf.predict([sent_features(toks)])[0]
    M["ner_name"] = ner
    yield


app = FastAPI(title="Vietnamese feedback triage", lifespan=lifespan)


class Ticket(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.post("/classify")
def classify(t: Ticket):
    t0 = time.perf_counter()
    text = normalize(t.text)
    p = M["topic"]([text])[0]
    s = M["sentiment"]([text])[0]
    topic, conf = TOPICS[int(p.argmax())], float(p.max())
    tokens = t.text.split()
    entities = spans(tokens, M["ner"](tokens)) if tokens else []
    entities += [{"label": "EMAIL", "text": m} for m in EMAIL.findall(t.text)]
    entities += [{"label": "PHONE", "text": m} for m in PHONE.findall(t.text)]

    auto = conf >= M["tau"]
    decision = "auto" if auto else "human_review"
    decisions[decision] += 1
    sentiment = SENTIMENTS[int(s.argmax())]
    latency.append((time.perf_counter() - t0) * 1000)
    return {
        "topic": topic, "team": ROUTES[topic] if auto else "Nhân viên phân loại",
        "confidence": round(conf, 4), "threshold": M["tau"], "decision": decision,
        "reason": None if auto else f"confidence {conf:.2f} < {M['tau']:.2f}: unclear topic or text outside the training domain",
        "sentiment": sentiment, "priority": "high" if sentiment == "negative" else "normal",
        "scores": {k: round(float(v), 4) for k, v in zip(TOPICS, p)},
        "entities": entities, "model": {"topic": M["topic_name"], "ner": M["ner_name"]},
    }


@app.get("/health")
def health():
    return {"status": "ok", "topic_model": M["topic_name"], "ner_model": M["ner_name"], "threshold": M["tau"]}


@app.get("/monitoring")
def monitoring():
    lat = np.array(latency) if latency else np.array([0.0])
    n = sum(decisions.values())
    # A rising human-review share is the first sign of drift (new kinds of tickets the model has not seen).
    return {"requests": n, "human_review_rate": decisions["human_review"] / n if n else None,
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2)}}
