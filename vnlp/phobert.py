"""PhoBERT fine-tuning: sentence classification (topic) and token classification (NER).

Runs on CPU (≈1 h for topic on a 12-thread laptop) or GPU (Colab: minutes). Same code path.
"""
import copy
import random
import time

import numpy as np
import torch
from seqeval.metrics import f1_score as span_f1
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from vnlp.data import segment

NAME = "vinai/phobert-base-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n: int, size: int, shuffle: bool):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def _pad(seqs: list[list[int]], value: int) -> torch.Tensor:
    return pad_sequence([torch.tensor(s) for s in seqs], batch_first=True, padding_value=value)


def _fit(model, encode_batch, n_train: int, evaluate, epochs: int, lr: float, bs: int):
    """Generic loop: AdamW + linear warmup, keep the epoch with the best validation score."""
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((n_train + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, losses = time.time(), []
        for b in _batches(n_train, bs, shuffle=True):
            out = model(**{k: v.to(DEVICE) for k, v in encode_batch(b).items()})
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(out.loss.item())
        score = evaluate(model)
        history.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val": score, "sec": time.time() - t0})
        print(history[-1], flush=True)
        if score > best:
            best, best_state = score, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


# ---------- sentence classification ----------

class TopicModel:
    def __init__(self, path_or_name: str = NAME, num_labels: int = 4, max_len: int = 96):
        self.tok = AutoTokenizer.from_pretrained(path_or_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(path_or_name, num_labels=num_labels).to(DEVICE)
        self.max_len = max_len

    def _encode(self, texts: list[str]) -> dict:
        enc = self.tok([segment(t) for t in texts], truncation=True, max_length=self.max_len, padding=True,
                       return_tensors="pt")
        return dict(enc)

    def fit(self, texts, labels, val_texts, val_labels, epochs=3, lr=2e-5, bs=32):
        texts, labels = list(texts), np.asarray(labels)

        def encode_batch(b):
            enc = self._encode([texts[i] for i in b])
            enc["labels"] = torch.tensor(labels[b])
            return enc

        def evaluate(_):
            return f1_score(val_labels, self.predict_proba(val_texts).argmax(1), average="macro")

        self.model, hist = _fit(self.model, encode_batch, len(texts), evaluate, epochs, lr, bs)
        return hist

    @torch.no_grad()
    def predict_proba(self, texts, bs: int = 64) -> np.ndarray:
        self.model.eval()
        texts, out = list(texts), []
        for i in range(0, len(texts), bs):
            enc = {k: v.to(DEVICE) for k, v in self._encode(texts[i:i + bs]).items()}
            out.append(torch.softmax(self.model(**enc).logits, -1).cpu().numpy())
        return np.concatenate(out)

    def save(self, path) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


# ---------- token classification (NER) ----------

class NerModel:
    """PhoBERT has only a slow tokenizer (no word_ids), so subwords are aligned by tokenizing word by word;
    the first subword of each syllable carries the label, the rest are ignored (-100)."""

    def __init__(self, labels: list[str], path_or_name: str = NAME, max_len: int = 128):
        self.labels = labels
        self.l2i = {l: i for i, l in enumerate(labels)}
        self.tok = AutoTokenizer.from_pretrained(path_or_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            path_or_name, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=self.l2i).to(DEVICE)
        self.max_len = max_len

    def _encode_one(self, tokens: list[str], tags: list[str] | None):
        ids, lab, first = [self.tok.cls_token_id], [-100], []
        for i, w in enumerate(tokens):
            sub = self.tok.encode(w, add_special_tokens=False) or [self.tok.unk_token_id]
            if len(ids) + len(sub) >= self.max_len - 1:
                break  # ponytail: titles fit in 128 subwords; truncated tail is predicted "O"
            first.append(len(ids))
            ids += sub
            lab += [self.l2i[tags[i]] if tags else 0] + [-100] * (len(sub) - 1)
        return ids + [self.tok.sep_token_id], lab + [-100], first

    def _batch(self, sents, with_labels: bool):
        enc = [self._encode_one(s["tokens"], s["tags"] if with_labels else None) for s in sents]
        ids = _pad([e[0] for e in enc], self.tok.pad_token_id)
        batch = {"input_ids": ids, "attention_mask": (ids != self.tok.pad_token_id).long()}
        if with_labels:
            batch["labels"] = _pad([e[1] for e in enc], -100)
        return batch, [e[2] for e in enc]

    def fit(self, train, val, epochs=8, lr=5e-5, bs=16):
        def evaluate(_):
            return span_f1([s["tags"] for s in val], self.predict([s["tokens"] for s in val]))

        return _fit(self.model, lambda b: self._batch([train[i] for i in b], True)[0], len(train), evaluate,
                    epochs, lr, bs)[1]

    @torch.no_grad()
    def predict(self, token_lists: list[list[str]], bs: int = 64) -> list[list[str]]:
        self.model.eval()
        out = []
        for i in range(0, len(token_lists), bs):
            chunk = token_lists[i:i + bs]
            batch, firsts = self._batch([{"tokens": t} for t in chunk], False)
            pred = self.model(**{k: v.to(DEVICE) for k, v in batch.items()}).logits.argmax(-1).cpu().numpy()
            for toks, f, p in zip(chunk, firsts, pred):
                tags = [self.labels[p[j]] for j in f]
                out.append(tags + ["O"] * (len(toks) - len(tags)))
        return out

    def save(self, path) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
