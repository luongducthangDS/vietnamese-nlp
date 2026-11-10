# vietnamese-nlp: agent notes

Project 02 in the AI Engineer portfolio roadmap: Vietnamese feedback triage (topic → team, sentiment → priority, NER, human review when unsure).

## 2026-11-10: status

- Done: classic (TF-IDF + LR / SVM for topic and sentiment, CRF NER), PhoBERT-base-v2 topic (3 epochs on CPU, about 3 h, val macro-F1 0.813), report, API, tests (8 pass), README.
- Test results: PhoBERT topic macro-F1 0.800 (TF-IDF 0.742–0.750); OOD AUROC 0.956; at threshold 0.885 it auto-routes 85.0% at 95.1% accuracy, OOD leak 8.8% (product titles 26%).
- Latency, idle CPU: PhoBERT 83 ms p50 / 109 ms p95; TF-IDF + LR 3.1 ms. (While CV training shared the CPU it measured 212 ms; always measure on an idle machine.)
- Not done: PhoBERT NER (use `notebooks/colab_phobert.ipynb` on a GPU).

## Gotchas

- Use `.venv` in this folder (the shell default `python` is DocuMindAI's venv).
- `SQLAlchemy==2.0.36` is pinned: 2.1.x breaks MLflow 2.19 (`FallbackAsyncAdaptedQueuePool`).
- MLflow metric names cannot contain `@` (report.py replaces it with `_at_`).
- PhoBERT needs word-segmented input (`underthesea.word_tokenize`, cached in `data.segment`). NER uses syllables, aligned by tokenizing word by word, because PhoBERT has only a slow tokenizer (no `word_ids`).
- `artifacts/phobert_topic/` (517 MB) is gitignored; retrain or copy it manually.
