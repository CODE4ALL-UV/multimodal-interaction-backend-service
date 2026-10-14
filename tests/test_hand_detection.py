"""Detección de la mano con MediaPipe sustituido por dobles.

El detector real necesita descargar un modelo de 7,8 MB, así que aquí se
reemplaza. La foto sí es real: se decodifica con OpenCV, como en producción.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import multimodal_service.presentation.api.sign_routes as sign_routes
from multimodal_service.main import app

client = TestClient(app)


def _png():
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _post_image(payload, content_type="image/png"):
    return client.post(
        "/api/signs/landmarks", files={"file": ("mano.png", payload, content_type)}
    )


class FakeLandmarker:
    """Ve siempre lo mismo, sea cual sea la foto."""

    def __init__(self, hand_landmarks, handedness):
        self.result = SimpleNamespace(
            hand_landmarks=hand_landmarks, handedness=handedness
        )
        self.images = []

    def detect(self, image):
        self.images.append(image)
        return self.result


def _hand(count=21):
    return [SimpleNamespace(x=i / 100, y=0.5, z=-0.1) for i in range(count)]


@pytest.fixture
def detector(monkeypatch):
    """Sustituye el detector por uno que ve una mano derecha."""
    landmarker = FakeLandmarker(
        [_hand()], [[SimpleNamespace(category_name="Right")]]
    )
    monkeypatch.setattr(sign_routes, "_get_landmarker", lambda: landmarker)
    return landmarker


@pytest.fixture
def fresh_state(monkeypatch):
    """Como recién arrancado: sin detector creado ni error recordado."""
    monkeypatch.setattr(sign_routes, "_import_error", None)
    monkeypatch.setattr(sign_routes, "_landmarker", None)
    monkeypatch.setattr(sign_routes, "_landmarker_error", None)


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    models = tmp_path / "models"
    monkeypatch.setattr(sign_routes, "MODELS_DIR", models)
    monkeypatch.setattr(sign_routes, "MODEL_PATH", models / "hand_landmarker.task")
    return models


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def fake_urlretrieve(url, destination):
        calls.append((url, destination))
        destination.write_bytes(b"modelo")

    monkeypatch.setattr(sign_routes.urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


class FakeVision:
    """Lo justo de `mediapipe.tasks.python.vision` para crear el detector."""

    created = []
    error = None

    class RunningMode:
        IMAGE = "image"

    class HandLandmarkerOptions:
        def __init__(self, **options):
            self.options = options

    class HandLandmarker:
        @staticmethod
        def create_from_options(options):
            FakeVision.created.append(options)
            if FakeVision.error is not None:
                raise FakeVision.error
            return FakeLandmarker([], [])


@pytest.fixture
def vision(monkeypatch, fresh_state, model_dir):
    FakeVision.created = []
    FakeVision.error = None
    monkeypatch.setattr(sign_routes, "vision", FakeVision)
    monkeypatch.setattr(
        sign_routes, "BaseOptions", lambda model_asset_path: model_asset_path
    )
    model_dir.mkdir()
    sign_routes.MODEL_PATH.write_bytes(b"modelo")
    return FakeVision


# --- La foto ---------------------------------------------------------------


def test_returns_the_21_points_and_which_hand_it_is(detector):
    response = _post_image(_png())

    assert response.status_code == 200
    hands = response.json()["hands"]
    assert len(hands) == 1
    assert hands[0]["handedness"] == "Right"
    assert len(hands[0]["landmarks"]) == 21
    assert hands[0]["landmarks"][3] == {"x": 0.03, "y": 0.5, "z": -0.1}
    # Al detector le llega la foto ya decodificada, no los bytes.
    assert len(detector.images) == 1


def test_no_hand_in_the_photo_is_an_empty_list(detector):
    detector.result = SimpleNamespace(hand_landmarks=[], handedness=[])

    response = _post_image(_png())

    assert response.status_code == 200
    assert response.json() == {"hands": []}


@pytest.mark.parametrize("handedness", [[], [[]]])
def test_unknown_handedness_is_an_empty_string(detector, handedness):
    detector.result = SimpleNamespace(hand_landmarks=[_hand()], handedness=handedness)

    hands = _post_image(_png()).json()["hands"]

    assert hands[0]["handedness"] == ""


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/jpg", "image/webp"])
def test_accepts_the_formats_a_phone_camera_produces(detector, content_type):
    # El tipo declarado solo filtra; OpenCV mira el contenido real.
    assert _post_image(_png(), content_type).status_code == 200


def test_rejects_images_over_4_mb(detector):
    response = _post_image(b"\0" * (sign_routes.MAX_IMAGE_BYTES + 1))

    assert response.status_code == 413
    assert detector.images == []


def test_bytes_that_are_not_a_picture_are_a_400(detector):
    response = _post_image(b"esto no es una imagen")

    assert response.status_code == 400
    assert detector.images == []


def test_missing_mediapipe_is_a_503(monkeypatch, fresh_state):
    monkeypatch.setattr(sign_routes, "_import_error", "No module named 'mediapipe'")

    response = _post_image(_png())

    assert response.status_code == 503
    assert "mediapipe" in response.json()["detail"]


# --- /status ---------------------------------------------------------------


def test_status_reports_missing_dependencies(monkeypatch, fresh_state):
    monkeypatch.setattr(sign_routes, "_import_error", "No module named 'cv2'")

    body = client.get("/api/signs/status").json()

    assert body["available"] is False
    assert body["model_ready"] is False
    assert "cv2" in body["reason"]


def test_status_is_available_and_says_if_the_model_is_downloaded(
    fresh_state, model_dir
):
    assert client.get("/api/signs/status").json() == {
        "available": True,
        "model_ready": False,
        "reason": None,
    }

    model_dir.mkdir()
    sign_routes.MODEL_PATH.write_bytes(b"modelo")

    assert client.get("/api/signs/status").json()["model_ready"] is True


# --- Crear el detector -----------------------------------------------------


def test_detector_is_created_once_and_reused(vision):
    first = sign_routes._get_landmarker()
    second = sign_routes._get_landmarker()

    assert first is second
    assert len(vision.created) == 1
    options = vision.created[0].options
    assert options["base_options"] == str(sign_routes.MODEL_PATH)
    assert options["running_mode"] == "image"
    assert options["num_hands"] == 1


def test_failure_to_create_the_detector_is_remembered(vision):
    vision.error = OSError("libGL.so.1: cannot open shared object file")

    with pytest.raises(RuntimeError, match="libGL"):
        sign_routes._get_landmarker()
    # La segunda vez no se reintenta: falla con el mismo motivo.
    with pytest.raises(RuntimeError, match="libGL"):
        sign_routes._get_landmarker()
    assert len(vision.created) == 1

    status = client.get("/api/signs/status").json()
    assert status["available"] is False
    assert status["model_ready"] is True
    assert "libGL" in status["reason"]

    assert _post_image(_png()).status_code == 503


# --- Descargar el modelo ---------------------------------------------------


def test_model_is_downloaded_the_first_time(model_dir, downloads):
    path = sign_routes._ensure_model()

    assert path == sign_routes.MODEL_PATH
    assert path.read_bytes() == b"modelo"
    assert downloads[0][0] == sign_routes.MODEL_URL
    # Se baja a un archivo temporal y luego se renombra, para no dejar a
    # medias el bueno si la descarga se corta.
    assert downloads[0][1].name == "hand_landmarker.task.part"
    assert not downloads[0][1].exists()


def test_model_already_downloaded_is_not_downloaded_again(model_dir, downloads):
    model_dir.mkdir()
    sign_routes.MODEL_PATH.write_bytes(b"modelo")

    assert sign_routes._ensure_model() == sign_routes.MODEL_PATH
    assert downloads == []


def test_empty_model_file_is_downloaded_again(model_dir, downloads):
    # Lo que queda si una descarga anterior falló a medias.
    model_dir.mkdir()
    sign_routes.MODEL_PATH.write_bytes(b"")

    sign_routes._ensure_model()

    assert len(downloads) == 1
    assert sign_routes.MODEL_PATH.read_bytes() == b"modelo"
