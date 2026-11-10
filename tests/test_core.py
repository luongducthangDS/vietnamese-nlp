import numpy as np

from vnlp.api import spans
from vnlp.data import normalize
from vnlp.report import ood_stats, pick_threshold


def test_normalize_matches_vsfc_style():
    assert normalize("Thầy  dạy HAY,dễ hiểu!") == "thầy dạy hay , dễ hiểu !"
    # NFC: decomposed "ệ" (e + combining marks) must equal the precomposed form
    assert normalize("Việt") == normalize("Việt")


def test_bio_to_spans():
    toks = ["Má", "Hồng", "Kem", "Maybelline", "Căng", "4.7ml"]
    tags = ["B-PRODUCT_TYPE", "I-PRODUCT_TYPE", "I-PRODUCT_TYPE", "B-BRAND", "O", "I-VOLUME_WEIGHT"]
    assert spans(toks, tags) == [{"label": "PRODUCT_TYPE", "text": "Má Hồng Kem"},
                                 {"label": "BRAND", "text": "Maybelline"},
                                 {"label": "VOLUME_WEIGHT", "text": "4.7ml"}]  # stray I- starts a span


def test_threshold_reaches_target_accuracy_on_accepted():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 1000)
    conf = rng.uniform(0.34, 1, 1000)
    pred = np.where(rng.uniform(size=1000) < conf, y, (y + 1) % 3)  # higher confidence -> more often right
    probs = np.full((1000, 3), 0.0)
    probs[np.arange(1000), pred] = conf
    probs[np.arange(1000), (pred + 1) % 3] = 1 - conf
    tau = pick_threshold(probs, y, 0.9)
    keep = probs.max(1) >= tau
    assert (probs.argmax(1)[keep] == y[keep]).mean() >= 0.9
    assert 0 < tau < 1


def test_ood_stats_separable():
    p_in = np.tile([0.97, 0.01, 0.01, 0.01], (100, 1))
    p_out = np.tile([0.3, 0.3, 0.2, 0.2], (100, 1))
    s = ood_stats(p_in, p_out, tau=0.9)
    assert s["AUROC"] == 1.0 and s["FPR@95TPR"] == 0.0 and s["ood_auto_routed"] == 0.0
