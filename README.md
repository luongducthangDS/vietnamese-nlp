# Vietnamese feedback triage: topic routing, priority, entities, human review

Vietnamese feedback is routed to the team that handles it, prioritised by sentiment, and stripped of
key fields (product, brand, phone, email). Anything the model is unsure about, including text from
outside the training domain, goes to a person instead of the wrong team. The project compares
traditional ML (TF-IDF + Logistic Regression / SVM, CRF) with PhoBERT fine-tuning and measures
behaviour on out-of-distribution text.

## Data

| Task | Dataset | Size | Labels |
|---|---|---|---|
| Topic → team, sentiment → priority | [UIT-VSFC](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback) (student feedback) | 11,426 / 1,583 / 3,166 | 4 topics (lecturer 72%, training program 18%, facility 5%, others 5%), 3 sentiments |
| Entities | [ViEcomNER](https://huggingface.co/datasets/tnhyen/vietnamese-ecommerce-ner) (CC BY-NC-SA 4.0) | 2,397 / 300 / 300 | 24 types (PRODUCT_TYPE, BRAND, MODEL, ATTRIBUTE, …) |
| Out of distribution | 1,000 banking-news sentences ([banking_sentiment_vietnamese](https://huggingface.co/datasets/iaiuet/banking_sentiment_vietnamese)) + 300 e-commerce product titles | 1,300 | none: the model should not route them |

Routing map: lecturer → Khoa / giảng viên, training_program → Phòng đào tạo, facility → Phòng cơ sở vật chất, others → Bộ phận tổng hợp.

All text is NFC-normalised and lower-cased with punctuation spaced like VSFC. Without this, casing
alone would give away OOD text, and OOD detection would look better than it is. Banking paragraphs are
split into sentences of 5–40 words for the same reason (length).

## Results (test set)

Topic routing:

| Model | Accuracy | Macro-F1 | Weighted-F1 | Auto-routed at ≥95% accuracy | OOD AUROC | OOD texts auto-routed |
|---|---|---|---|---|---|---|
| **PhoBERT-base-v2, fine-tuned 3 epochs** | **0.896** | **0.800** | **0.893** | **85.0%** (95.1% correct) | **0.956** | 8.8% |
| TF-IDF + linear SVM (calibrated) | 0.869 | 0.750 | 0.863 | 77.4% (94.4% correct) | 0.884 | 16.5% |
| TF-IDF + Logistic Regression | 0.864 | 0.742 | 0.859 | 76.1% (94.3% correct) | 0.887 | 14.1% |

- PhoBERT adds 5 macro-F1 points, and more importantly it is better calibrated: at the same accuracy
  target it handles 85% of tickets alone versus 76%, and lets less OOD text through (8.8% vs 14–17%).
- The weak class is `others` (PhoBERT F1 0.53, recall 0.46): a small, vague catch-all class.
- **Most of PhoBERT's confident "errors" are label noise.** "cô nhiệt tình, dạy có tâm" ("the teacher
  is dedicated") is labelled `training_program` in UIT-VSFC but is plainly about the lecturer. The 12
  most confident errors in `docs/REPORT.md` are almost all like this. Cleaning labels would likely help
  more than a bigger model.
- PhoBERT stops banking news well (4% leak) but auto-routes 26% of product titles. Short noun-phrase
  titles look like feedback about facilities. A real deployment would add these to training as an
  explicit "out of scope" class or use a dedicated OOD detector.

Sentiment (priority): TF-IDF + LR accuracy 0.884, macro-F1 0.752. The neutral class is hard (F1 0.43, 5% of data).

NER (span-level exact match, 24 types): CRF micro-F1 0.501, macro-F1 0.492. PhoBERT NER is ready to
train on a GPU (`notebooks/colab_phobert.ipynb`); not yet run.

Latency (one text, CPU, measured while another training job shared the CPU): TF-IDF + LR 3 ms p50;
PhoBERT 212 ms p50 / 874 ms p95, including word segmentation. TF-IDF is around 70× faster, the
trade-off behind offering both models in the API.

Full tables, confusion matrices and error examples: [docs/REPORT.md](docs/REPORT.md).

## How the decision is made

```
text ─► normalize ─► topic model ─► max probability ≥ τ ? ── yes ─► team (auto)
                         │                                └─ no ──► "Nhân viên phân loại" (human review)
                         ├─► sentiment (TF-IDF + LR) ─► negative = high priority
                         └─► NER (CRF or PhoBERT) + regex (phone, email) ─► entities
```

τ is picked on the **validation** set as the lowest threshold whose auto-routed items are ≥95%
correct, then applied unchanged to test and OOD data. A rising human-review rate in `/monitoring` is
the first signal of drift.

## Run

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m vnlp.data                 # download datasets into data/raw/
python -m vnlp.train classic        # TF-IDF + LR / SVM, CRF (~15 min)
python -m vnlp.train phobert        # PhoBERT topic (~3 h on CPU, minutes on a GPU)
python -m vnlp.train phobert-ner    # optional, GPU recommended
python -m vnlp.report               # docs/REPORT.md, artifacts/thresholds.json, MLflow
pytest -q
uvicorn vnlp.api:app --port 8001    # TOPIC_MODEL=phobert|tfidf_lr|tfidf_svm, NER_MODEL=crf|phobert
```

```bash
curl -X POST localhost:8001/classify -H "Content-Type: application/json" \
  -d '{"text": "Phòng học nóng quá, máy chiếu hỏng suốt cả tuần."}'
# {"topic": "facility", "team": "Phòng cơ sở vật chất", "decision": "auto", "priority": "high", ...}
```

## Limits

- UIT-VSFC is student feedback, not company email. The pipeline (routing, thresholds, OOD) carries
  over, but the numbers do not; a real deployment needs labelled tickets from that company.
- NER is trained on product titles, not on ticket text.
- Max-softmax confidence is a simple OOD detector. It misses OOD text that resembles a training class
  (product titles → facility).
- UIT-VSFC's licence is not stated on its dataset page; use is limited to research and portfolio.
