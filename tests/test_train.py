"""
Run with:
    pytest -v tests/test_train.py
"""

import os
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.train as train

@pytest.mark.parametrize("device", ["cpu"])
def test_train_model_smoke(monkeypatch, tmp_path, device):
    """Smoke test: verify that training function runs without crashing (mocked)."""

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images\nval: images\nnc: 3\nnames: ['RTU', 'Split', 'VRF']\n")

    class MockResults:
        results_dict = {"metrics/mAP50(B)": 0.75, "metrics/precision(B)": 0.9, "metrics/recall(B)": 0.85}
        epochs = 1

    class MockModel:
        def train(self, **kwargs):
            return MockResults()

    monkeypatch.setattr(train, "YOLO", lambda *a, **k: MockModel())

    result = train.train_model(str(data_yaml), epochs=1, device=device)
    assert hasattr(result, "epochs"), "Training result should contain epoch info."
    assert result.results_dict["metrics/mAP50(B)"] >= 0.7


def test_export_formats(monkeypatch, tmp_path):

    class MockYOLO:
        def __init__(self, model): ...
        def export(self, **kwargs): pass

    monkeypatch.setattr(train, "YOLO", MockYOLO)

    model_path = tmp_path / "dummy.pt"
    model_path.write_text("dummy content")
    exported = train.export_models(str(model_path))
    assert isinstance(exported, list)