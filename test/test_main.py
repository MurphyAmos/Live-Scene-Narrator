"""Tests for Live Scene Narrator.

These tests avoid requiring a webcam, YOLO weights, or a Gemini API key by
stubbing those external dependencies before importing ``main.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def __float__(self):
        return float(self.value)


class FakeVector:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeBoxes:
    def __init__(self, detections):
        self.cls = [d["class_id"] for d in detections]
        self.conf = [d["confidence"] for d in detections]
        self.xywhn = [FakeVector(d["xywhn"]) for d in detections]
        self.xyxyn = [FakeVector(d["xyxyn"]) for d in detections]
        self.id = [FakeScalar(d["track_id"]) for d in detections]


class FakeResult:
    def __init__(self, detections):
        self.boxes = FakeBoxes(detections)
        self.names = {0: "person", 1: "cup"}

    def plot(self):
        return "annotated-frame"


class FakeModel:
    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def track(self, frame, persist=True, verbose=False):
        self.calls.append(
            {"frame": frame, "persist": persist, "verbose": verbose}
        )
        return self.results


class ImportCamera:
    """Camera used only while main.py imports.

    main.py calls get_frame_info() at import time, so grab() immediately
    returns False to stop the loop without touching real hardware.
    """

    def get(self, prop):
        return 640 if prop == 3 else 480

    def grab(self):
        return False

    def read(self):
        return False, None

    def release(self):
        pass


class OneFrameCamera:
    def __init__(self):
        self.released = False

    def get(self, prop):
        return 640 if prop == 3 else 480

    def grab(self):
        return True

    def read(self):
        return True, "raw-frame"

    def release(self):
        self.released = True


class FakeInteractions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FakeClient:
    def __init__(self, responses):
        self.interactions = FakeInteractions(responses)


class FakeResponse:
    def __init__(self, response_id, output_text="scene summary"):
        self.id = response_id
        self.output_text = output_text


@pytest.fixture
def narrator(monkeypatch):
    """Import main.py with camera/model/API dependencies replaced by fakes."""

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.INTER_NEAREST = 0
    fake_cv2.VideoCapture = lambda _index: ImportCamera()
    fake_cv2.resize = lambda frame, _size, interpolation=None: frame
    fake_cv2.imshow = lambda *_args, **_kwargs: None
    fake_cv2.waitKey = lambda _delay: ord("q")
    fake_cv2.destroyAllWindows = lambda: None
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLOE = lambda _weights: FakeModel()
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    ultralytics_models = types.ModuleType("ultralytics.models")
    ultralytics_yolo = types.ModuleType("ultralytics.models.yolo")
    ultralytics_yolo_yoloe = types.ModuleType("ultralytics.models.yolo.yoloe")
    ultralytics_yolo_yoloe.YOLOEPESegTrainer = object
    monkeypatch.setitem(sys.modules, "ultralytics.models", ultralytics_models)
    monkeypatch.setitem(sys.modules, "ultralytics.models.yolo", ultralytics_yolo)
    monkeypatch.setitem(
        sys.modules, "ultralytics.models.yolo.yoloe", ultralytics_yolo_yoloe
    )

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = lambda api_key=None: FakeClient([])
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    module_path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("live_scene_narrator_main", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_first_summary_starts_new_interaction(narrator):
    client = FakeClient([FakeResponse("interaction-1")])
    narrator.client = client
    narrator.previous_interaction_id = None

    response = narrator.prompt_summary([{"frame_id": 2, "detections": []}])

    assert response.output_text == "scene summary"
    assert narrator.previous_interaction_id == "interaction-1"
    assert "previous_interaction_id" not in client.interactions.calls[0]


def test_later_summary_chains_previous_interaction(narrator):
    client = FakeClient([FakeResponse("interaction-2")])
    narrator.client = client
    narrator.previous_interaction_id = "interaction-1"

    narrator.prompt_summary([{"frame_id": 4, "detections": []}])

    call = client.interactions.calls[0]
    assert call["previous_interaction_id"] == "interaction-1"
    assert narrator.previous_interaction_id == "interaction-2"


def test_detection_pipeline_filters_low_confidence_and_logs_jsonl(
    narrator, monkeypatch, tmp_path
):
    detections = [
        {
            "class_id": 0,
            "confidence": 0.91,
            "track_id": 7,
            "xywhn": [0.20, 0.70, 0.10, 0.20],
            "xyxyn": [0.15, 0.60, 0.25, 0.80],
        },
        {
            "class_id": 1,
            "confidence": 0.10,
            "track_id": 8,
            "xywhn": [0.80, 0.20, 0.05, 0.05],
            "xyxyn": [0.775, 0.175, 0.825, 0.225],
        },
    ]

    narrator.camera = OneFrameCamera()
    narrator.model = FakeModel([FakeResult(detections)])
    monkeypatch.chdir(tmp_path)

    # Exit after the first processed frame.
    monkeypatch.setattr(narrator.cv2, "waitKey", lambda _delay: ord("q"))

    narrator.get_frame_info()

    log_path = tmp_path / "Video_Data.jsonl"
    assert log_path.exists()

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 1
    assert len(records[0]["detections"]) == 1

    detection = records[0]["detections"][0]
    assert detection["class"] == "person"
    assert detection["track_id"] == 7
    assert detection["confidence"] == pytest.approx(0.91)
    assert detection["spatial"]["area"] == pytest.approx(0.02)
    assert detection["spatial"]["aspect_ratio"] == pytest.approx(0.5)
    assert detection["spatial"]["horizontal_region"] == "left"
    assert detection["spatial"]["vertical_region"] == "bottom"
    assert narrator.camera.released is True


def test_tracking_uses_persistent_ids(narrator, monkeypatch, tmp_path):
    detection = {
        "class_id": 0,
        "confidence": 0.95,
        "track_id": 42,
        "xywhn": [0.50, 0.50, 0.20, 0.40],
        "xyxyn": [0.40, 0.30, 0.60, 0.70],
    }

    model = FakeModel([FakeResult([detection])])
    narrator.model = model
    narrator.camera = OneFrameCamera()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(narrator.cv2, "waitKey", lambda _delay: ord("q"))

    narrator.get_frame_info()

    assert len(model.calls) == 1
    assert model.calls[0]["persist"] is True
    assert model.calls[0]["verbose"] is False
