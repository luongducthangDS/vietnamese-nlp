"""API against real artifacts (train classic + report first). Uses the fast TF-IDF model and CRF NER."""
import os
from pathlib import Path

import pytest

ART = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts"))
pytestmark = pytest.mark.skipif(not (ART / "thresholds.json").exists(), reason="run vnlp.train classic + vnlp.report")


@pytest.fixture(scope="module")
def client():
    os.environ["TOPIC_MODEL"] = "tfidf_lr"
    os.environ["NER_MODEL"] = "crf"
    from fastapi.testclient import TestClient

    from vnlp.api import app
    with TestClient(app) as c:
        yield c


def test_clear_feedback_is_auto_routed(client):
    r = client.post("/classify", json={"text": "Phòng học nóng quá, máy chiếu hỏng suốt cả tuần."}).json()
    assert r["topic"] == "facility" and r["decision"] == "auto"
    assert r["team"] == "Phòng cơ sở vật chất" and r["priority"] == "high"


def test_out_of_domain_goes_to_human(client):
    r = client.post("/classify", json={"text": "Ngân hàng nhà nước điều chỉnh lãi suất tái cấp vốn từ ngày 15/10."}).json()
    assert r["decision"] == "human_review" and r["team"] == "Nhân viên phân loại"


def test_entities_and_contacts(client):
    r = client.post("/classify", json={"text": "Mua Tai Nghe Bluetooth Sony bị lỗi, liên hệ 0912345678 hoặc an.nguyen@example.com"}).json()
    labels = {e["label"] for e in r["entities"]}
    assert {"PHONE", "EMAIL"} <= labels


def test_validation_and_monitoring(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422
    m = client.get("/monitoring").json()
    assert m["requests"] >= 3 and 0 <= m["human_review_rate"] <= 1
