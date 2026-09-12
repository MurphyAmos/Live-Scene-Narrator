"""
pytest tests for main.py

These tests isolate the program from:
- the real webcam
- the real YOLOE model
- Gemini/API calls
- the on-disk vector database

Run:
    pip install pytest numpy
    pytest -q

If your program is not named main.py, change MODULE_UNDER_TEST below.
"""

import importlib
import json
import sys
import types

import numpy as np
import pytest


MODULE_UNDER_TEST = "main"


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class FakeVector:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeBoxes:
    """Two detections: one valid person and one low-confidence bicycle."""

    def __init__(self):
        self.cls = [0, 1]
        self.conf = [0.90, 0.10]
        self.id = [FakeScalar(7), FakeScalar(8)]

        self.xywhn = [
            FakeVector([0.50, 0.50, 0.20, 0.40]),
            FakeVector([0.80, 0.70, 0.10, 0.10]),
        ]

        self.xyxyn = [
            FakeVector([0.40, 0.30, 0.60, 0.70]),
            FakeVector([0.75, 0.65, 0.85, 0.75]),
        ]


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.names = {
            0: "person",
            1: "bicycle",
        }

    def plot(self):
        return np.zeros((135, 240, 3), dtype=np.uint8)


class FakeYOLOE:
    def __init__(self, model_name):
        self.model_name = model_name
        self.track_calls = []

    def track(self, frame, persist=True, verbose=False):
        self.track_calls.append(
            {
                "frame": frame,
                "persist": persist,
                "verbose": verbose,
            }
        )
        return [FakeResult()]


class FakeCamera:
    def __init__(self):
        self.released = False
        self.grab_calls = 0
        self.read_calls = 0

    def get(self, prop):
        # Fake 1280x720 camera.
        if prop == 3:
            return 1280.0
        if prop == 4:
            return 720.0
        return 0.0

    def grab(self):
        self.grab_calls += 1
        return True

    def read(self):
        self.read_calls += 1
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return True, frame

    def release(self):
        self.released = True


class FakeInteractions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            output_text=(
                "A person is visible near the center of an indoor scene "
                "and appears to be shifting position."
            )
        )


class FakeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.interactions = FakeInteractions()


class FakeMemory:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.saved = []

    def save(self, summary, observations):
        self.saved.append((summary, observations))


@pytest.fixture
def app(monkeypatch):
    """
    Import main.py with all expensive/external dependencies replaced
    before the module executes.
    """

    # -------------------------
    # Fake cv2
    # -------------------------
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.CAP_PROP_FRAME_WIDTH = 3
    fake_cv2.CAP_PROP_FRAME_HEIGHT = 4
    fake_cv2.INTER_NEAREST = 0

    import_camera = FakeCamera()

    fake_cv2.VideoCapture = lambda index: import_camera

    def resize(frame, size, interpolation=None):
        width, height = size
        return np.zeros((height, width, 3), dtype=frame.dtype)

    fake_cv2.resize = resize
    fake_cv2.imshow = lambda *args, **kwargs: None
    fake_cv2.waitKey = lambda delay: ord("q")
    fake_cv2.destroyAllWindows = lambda: None

    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    # -------------------------
    # Fake ultralytics.YOLOE
    # -------------------------
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLOE = FakeYOLOE
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    # -------------------------
    # Fake google.genai
    # -------------------------
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = FakeClient

    fake_google = types.ModuleType("google")
    fake_google.genai = fake_genai

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    # -------------------------
    # Fake vectordb.Memory
    # -------------------------
    fake_vectordb = types.ModuleType("vectordb")
    fake_vectordb.Memory = FakeMemory
    monkeypatch.setitem(sys.modules, "vectordb", fake_vectordb)

    # Force a clean import each test.
    sys.modules.pop(MODULE_UNDER_TEST, None)

    module = importlib.import_module(MODULE_UNDER_TEST)

    # Prevent cls/clear from touching the test terminal.
    monkeypatch.setattr(module.os, "system", lambda *args, **kwargs: 0)

    yield module

    sys.modules.pop(MODULE_UNDER_TEST, None)


def test_camera_resolution_is_scaled_to_target_max(app):
    """
    A fake 1280x720 camera should be resized so its longest side is 240.
    Expected result: 240x135.
    """
    assert app.width == 240
    assert app.height == 135


def test_prompt_summary_updates_previous_summary_and_vector_memory(app):
    observations = [
        {
            "frame_id": 2,
            "timestamp": 123.45,
            "detections": [
                {
                    "track_id": 7,
                    "class": "person",
                    "confidence": 0.9,
                    "position": {
                        "center_x": 0.5,
                        "center_y": 0.5,
                        "width": 0.2,
                        "height": 0.4,
                    },
                }
            ],
        }
    ]

    app.previous_scene_summary = ""

    response = app.prompt_summary(observations)

    assert response.output_text
    assert app.previous_scene_summary == response.output_text

    # The generated natural-language summary should be persisted with
    # the observations that generated it.
    assert len(app.memory.saved) == 1

    saved_summary, saved_observations = app.memory.saved[0]

    assert saved_summary == response.output_text
    assert saved_observations == observations

    # Verify Gemini was called with the expected model and current data.
    assert len(app.client.interactions.calls) == 1
    call = app.client.interactions.calls[0]

    assert call["model"] == "gemini-3.5-flash-lite"
    assert "New observations:" in call["input"]
    assert "person" in call["input"]


def test_prompt_summary_includes_previous_scene_state(app):
    app.previous_scene_summary = (
        "A person was standing on the left side of the room."
    )

    new_observations = [
        {
            "frame_id": 4,
            "timestamp": 124.00,
            "detections": [],
        }
    ]

    app.prompt_summary(new_observations)

    call = app.client.interactions.calls[-1]

    assert "Previous scene state:" in call["input"]
    assert (
        "A person was standing on the left side of the room."
        in call["input"]
    )


def test_get_frame_info_writes_expected_detection_jsonl(
    app, monkeypatch, tmp_path
):
    """
    Exercise one processed frame and quit immediately.

    This verifies:
    - frame skipping/grab behavior
    - model.track() is called
    - confidence filtering works
    - normalized position/spatial fields are written
    - Description.jsonl contains valid JSON
    - camera cleanup occurs
    """

    monkeypatch.chdir(tmp_path)

    camera = FakeCamera()
    model = FakeYOLOE("fake-model.pt")

    app.camera = camera
    app.model = model

    # Exit after the first processed frame.
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))

    app.get_frame_info()

    output_file = tmp_path / "Description.jsonl"

    assert output_file.exists()

    lines = output_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    frame = json.loads(lines[0])

    # count=1 is skipped with grab(); count=2 is processed.
    assert frame["frame_id"] == 2

    # The valid person survives; the 0.10-confidence bicycle is filtered out.
    assert len(frame["detections"]) == 1

    detection = frame["detections"][0]

    assert detection["track_id"] == 7
    assert detection["class"] == "person"
    assert detection["class_id"] == 0
    assert detection["confidence"] == pytest.approx(0.90)

    assert detection["position"]["center_x"] == pytest.approx(0.50)
    assert detection["position"]["center_y"] == pytest.approx(0.50)
    assert detection["position"]["width"] == pytest.approx(0.20)
    assert detection["position"]["height"] == pytest.approx(0.40)

    assert detection["spatial"]["area"] == pytest.approx(0.08)
    assert detection["spatial"]["aspect_ratio"] == pytest.approx(0.50)
    assert detection["spatial"]["horizontal_region"] == "center"
    assert detection["spatial"]["vertical_region"] == "middle"

    assert camera.grab_calls == 1
    assert camera.read_calls == 1
    assert camera.released is True

    assert len(model.track_calls) == 1
    assert model.track_calls[0]["persist"] is True
    assert model.track_calls[0]["verbose"] is False


def test_low_confidence_detection_is_removed(app, monkeypatch, tmp_path):
    """
    The fake model returns:
      - person: 0.90
      - bicycle: 0.10

    Your program's threshold is 0.20, so only the person should remain.
    """
    monkeypatch.chdir(tmp_path)

    app.camera = FakeCamera()
    app.model = FakeYOLOE("fake-model.pt")

    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))

    app.get_frame_info()

    frame = json.loads(
        (tmp_path / "Description.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )

    classes = [d["class"] for d in frame["detections"]]

    assert "person" in classes
    assert "bicycle" not in classes
