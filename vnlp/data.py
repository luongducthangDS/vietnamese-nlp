"""Datasets: UIT-VSFC (feedback topic + sentiment), ViEcomNER (entities), out-of-distribution text."""
import json
import re
import unicodedata
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HF = "https://huggingface.co/"
SOURCES = {
    **{f"vsfc_{s}.parquet": f"api/datasets/uitnlp/vietnamese_students_feedback/parquet/default/{s}/0.parquet"
       for s in ("train", "validation", "test")},
    "bank_test.parquet": "api/datasets/iaiuet/banking_sentiment_vietnamese/parquet/default/test/0.parquet",
    **{f"ner_{s}.jsonl": f"datasets/tnhyen/vietnamese-ecommerce-ner/resolve/main/{s}.jsonl"
       for s in ("train", "validation", "test")},
}
TOPICS = ["lecturer", "training_program", "facility", "others"]
SENTIMENTS = ["negative", "neutral", "positive"]
# Feedback topic -> team that handles it (the routing target).
ROUTES = {"lecturer": "Khoa / giảng viên", "training_program": "Phòng đào tạo",
          "facility": "Phòng cơ sở vật chất", "others": "Bộ phận tổng hợp"}


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, path in SOURCES.items():
        if not (RAW / name).exists():
            print("downloading", name)
            urllib.request.urlretrieve(HF + path, RAW / name)


def normalize(text: str) -> str:
    # VSFC is lower-cased with spaces around punctuation; bring every input to that form so casing or
    # punctuation spacing alone cannot give away an out-of-distribution text.
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"([.,!?;:()\"])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=50_000)
def segment(text: str) -> str:
    """Word segmentation for PhoBERT (multi-syllable words joined by '_')."""
    from underthesea import word_tokenize
    return word_tokenize(text, format="text")


def load_vsfc(split: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"vsfc_{split}.parquet")
    df["text"] = df["sentence"].map(normalize)
    return df[["text", "topic", "sentiment"]].reset_index(drop=True)


def load_ood(n_bank: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Texts the classifier was never meant to see: banking news sentences and e-commerce product titles.

    Banking paragraphs are split into sentences of 5-40 words so length alone does not separate them.
    """
    bank = pd.read_parquet(RAW / "bank_test.parquet")["text_paragraph"]
    sents = bank.str.split(r"(?<=[.!?])\s+").explode().dropna()
    sents = sents[sents.str.split().str.len().between(5, 40)]
    sents = sents.sample(min(n_bank, len(sents)), random_state=seed)
    titles = [json.loads(line)["text"] for line in open(RAW / "ner_test.jsonl", encoding="utf-8")]
    df = pd.DataFrame({"text": list(sents) + titles,
                       "source": ["banking_news"] * len(sents) + ["ecommerce_title"] * len(titles)})
    df["text"] = df["text"].map(normalize)
    return df


def load_ner(split: str) -> list[dict]:
    """Syllable tokens with BIO tags from character spans."""
    out = []
    for line in open(RAW / f"ner_{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        tokens, tags = [], []
        for m in re.finditer(r"\S+", r["text"]):
            tag = "O"
            for e in r["entities"]:
                if e["start_char"] <= m.start() < e["end_char"]:
                    tag = ("B-" if m.start() == e["start_char"] or not tags or tags[-1][2:] != e["label"]
                           else "I-") + e["label"]
                    break
            tokens.append(m.group())
            tags.append(tag)
        out.append({"id": r["id"], "tokens": tokens, "tags": tags})
    return out


if __name__ == "__main__":
    download()
