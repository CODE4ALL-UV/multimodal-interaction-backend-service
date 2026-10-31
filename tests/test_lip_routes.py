"""Las rutas de la lectura de labios, con MediaPipe sustituido por dobles.

El detector real necesita descargar un modelo de 3,8 MB, así que aquí se
reemplaza. Las fotos sí son reales: se decodifican con OpenCV, como en producción.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import multimodal_service.presentation.api.lip_routes as lip_routes
from multimodal_service.domain.lip_reading import LIPS_OUTER, VOCABULARY
from multimodal_service.main import app

client = TestClient(app)


def _png():
    ok, encoded = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _post_frame(payload, content_type="image/png", data=None):
    return client.post(
        "/api/lips/landmarks",
        files={"file": ("boca.png", payload, content_type)},
        data=data,
    )


def _post_clip(payloads, content_type="image/png", data=None):
    return client.post(
        "/api/lips/read",
        files=[("files", ("foto.png", p, content_type)) for p in payloads],
        data=data,
    )


def _face(opening=0.0, width=0.6):
    """Los puntos de una cara, en una foto cuadrada."""
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def place(index, x, y):
        points[index] = SimpleNamespace(x=x, y=y, z=0.0)

    place(33, 0.4, 0.4)
    place(263, 0.6, 0.4)
    place(61, 0.5 - 0.1 * width, 0.6)
    place(291, 0.5 + 0.1 * width, 0.6)
    place(13, 0.5, 0.6 - 0.1 * opening)
    place(14, 0.5, 0.6 + 0.1 * opening)
    return points


def _result(*faces, rounding=0.0):
    return SimpleNamespace(
        face_landmarks=list(faces),
        face_blendshapes=[
            [
                SimpleNamespace(category_name="jawOpen", score=0.9),
                SimpleNamespace(category_name="mouthPucker", score=rounding),
                SimpleNamespace(category_name="mouthFunnel", score=rounding / 2),
            ]
        ]
        if faces
        else [],
    )


class FakeLandmarker:
    """Ve lo que le diga la prueba, una foto tras otra."""

    def __init__(self, *results):
        self.results = list(results)
        self.images = []

    def detect(self, image):
        self.images.append(image)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


@pytest.fixture
def detector(monkeypatch):
    """Sustituye el detector por uno que ve una boca muy abierta."""
    landmarker = FakeLandmarker(_result(_face(opening=0.26)))
    monkeypatch.setattr(lip_routes, "_get_landmarker", lambda: landmarker)
    return landmarker


@pytest.fixture
def fresh_state(monkeypatch):
    """Como recién arrancado: sin detector creado ni error recordado."""
    monkeypatch.setattr(lip_routes, "_import_error", None)
    monkeypatch.setattr(lip_routes, "_landmarker", None)
    monkeypatch.setattr(lip_routes, "_landmarker_error", None)


@pytest.fixture
def model_dir(monkeypatch, tmp_path):
    models = tmp_path / "models"
    monkeypatch.setattr(lip_routes, "MODELS_DIR", models)
    monkeypatch.setattr(lip_routes, "MODEL_PATH", models / "face_landmarker.task")
    return models


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def fake_urlretrieve(url, destination):
        calls.append((url, destination))
        destination.write_bytes(b"modelo")

    monkeypatch.setattr(lip_routes.urllib.request, "urlretrieve", fake_urlretrieve)
    return calls


class FakeVision:
    """Lo justo de `mediapipe.tasks.python.vision` para crear el detector."""

    created = []
    error = None

    class RunningMode:
        IMAGE = "image"

    class FaceLandmarkerOptions:
        def __init__(self, **options):
            self.options = options

    class FaceLandmarker:
        @staticmethod
        def create_from_options(options):
            FakeVision.created.append(options)
            if FakeVision.error is not None:
                raise FakeVision.error
            return FakeLandmarker(_result())


@pytest.fixture
def vision(monkeypatch, fresh_state, model_dir):
    FakeVision.created = []
    FakeVision.error = None
    monkeypatch.setattr(lip_routes, "vision", FakeVision)
    monkeypatch.setattr(
        lip_routes, "BaseOptions", lambda model_asset_path: model_asset_path
    )
    model_dir.mkdir()
    lip_routes.MODEL_PATH.write_bytes(b"modelo")
    return FakeVision


# --- Una foto --------------------------------------------------------------


def test_a_frame_returns_the_lip_contours_and_the_mouth_shape(detector):
    response = _post_frame(_png())

    assert response.status_code == 200
    face = response.json()["face"]
    assert len(face["lips"]["outer"]) == 20
    assert len(face["lips"]["inner"]) == 20
    assert face["lips"]["outer"][0] == {"x": _face(0.26)[LIPS_OUTER[0]].x, "y": 0.6}
    assert face["opening"] == pytest.approx(0.26)
    assert face["width"] == pytest.approx(0.6)
    assert face["shape"] == "A"
    assert face["scores"]["A"] > 0.5
    # Al detector le llega la foto ya decodificada, no los bytes.
    assert len(detector.images) == 1


def test_no_face_in_the_frame_is_not_an_error(detector):
    detector.results = [_result()]

    response = _post_frame(_png())

    assert response.status_code == 200
    assert response.json() == {"face": None}


def test_a_face_without_size_counts_as_no_face(detector):
    flat = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    detector.results = [_result(flat)]

    assert _post_frame(_png()).json() == {"face": None}


def test_rounding_comes_from_the_strongest_blendshape(detector):
    detector.results = [_result(_face(opening=0.06, width=0.4), rounding=0.8)]

    face = _post_frame(_png()).json()["face"]

    assert face["rounding"] == pytest.approx(0.8)
    assert face["shape"] == "U"


def test_rounding_without_blendshapes_is_zero(detector):
    detector.results = [
        SimpleNamespace(face_landmarks=[_face()], face_blendshapes=[])
    ]

    assert _post_frame(_png()).json()["face"]["rounding"] == 0.0


def test_the_persons_own_relaxed_mouth_changes_the_reading(detector):
    detector.results = [_result(_face(opening=0.06, width=0.42), rounding=0.3)]

    default = _post_frame(_png()).json()["face"]["shape"]
    relaxed_narrow = _post_frame(
        _png(), data={"neutral_width": "0.42"}
    ).json()["face"]["shape"]

    assert default == "U"
    assert relaxed_narrow != "U"


@pytest.mark.parametrize("neutral_width", ["0.1", "2.0", "ancho"])
def test_a_relaxed_width_that_makes_no_sense_is_rejected(detector, neutral_width):
    response = _post_frame(_png(), data={"neutral_width": neutral_width})

    assert response.status_code == 422
    assert detector.images == []


@pytest.mark.parametrize("content_type", ["image/jpeg", "image/jpg", "image/webp"])
def test_accepts_the_formats_a_camera_produces(detector, content_type):
    assert _post_frame(_png(), content_type).status_code == 200


def test_rejects_files_that_are_not_images(detector):
    assert _post_frame(b"hola", "text/plain").status_code == 415
    assert detector.images == []


def test_rejects_an_empty_image(detector):
    assert _post_frame(b"").status_code == 400


def test_rejects_images_over_1_mb(detector):
    response = _post_frame(b"\0" * (lip_routes.MAX_IMAGE_BYTES + 1))

    assert response.status_code == 413
    assert detector.images == []


def test_bytes_that_are_not_a_picture_are_a_400(detector):
    response = _post_frame(b"esto no es una imagen")

    assert response.status_code == 400
    assert detector.images == []


def test_missing_mediapipe_is_a_503(monkeypatch, fresh_state):
    monkeypatch.setattr(lip_routes, "_import_error", "No module named 'mediapipe'")

    response = _post_frame(_png())

    assert response.status_code == 503
    assert "mediapipe" in response.json()["detail"]


# --- Un clip ---------------------------------------------------------------


def _clip_of(*openings):
    return FakeLandmarker(*[_result(_face(opening=o)) for o in openings])


def test_a_clip_is_read_as_one_word(monkeypatch):
    # Reposo, una A larga y reposo: solo puede ser «mal».
    landmarker = _clip_of(0, 0, 0.13, 0.26, 0.26, 0.26, 0.26, 0.13, 0, 0)
    monkeypatch.setattr(lip_routes, "_get_landmarker", lambda: landmarker)

    response = _post_clip([_png()] * 10)

    assert response.status_code == 200
    body = response.json()
    assert body["frames"] == 10
    assert body["frames_with_face"] == 10
    assert body["reason"] is None
    assert "A" in body["shapes"]
    assert "mal" in [body["word"], *body["alternatives"]]
    assert len(landmarker.images) == 10


def test_a_still_mouth_in_a_clip_says_why_it_read_nothing(detector):
    response = _post_clip([_png()] * 6)

    body = response.json()
    assert body["word"] is None
    assert body["alternatives"] == []
    assert body["reason"] == "no_movement"


def test_a_clip_without_a_face_says_so(detector):
    detector.results = [_result()]

    body = _post_clip([_png()] * 6).json()

    assert body["frames"] == 6
    assert body["frames_with_face"] == 0
    assert body["reason"] == "no_face"


def test_a_bad_frame_does_not_ruin_the_clip(detector):
    body = _post_clip([_png(), b"esto no es una imagen", _png(), _png()]).json()

    assert body["frames"] == 4
    assert body["frames_with_face"] == 3


def test_rejects_a_clip_with_too_many_frames(detector):
    response = _post_clip([_png()] * (lip_routes.MAX_CLIP_FRAMES + 1))

    assert response.status_code == 413
    assert detector.images == []


def test_rejects_a_clip_that_weighs_too_much(monkeypatch, detector):
    monkeypatch.setattr(lip_routes, "MAX_CLIP_BYTES", 10)

    response = _post_clip([_png()] * 3)

    assert response.status_code == 413
    assert detector.images == []


def test_rejects_a_clip_with_a_file_that_is_not_an_image(detector):
    assert _post_clip([b"hola"], "text/plain").status_code == 415


def test_a_clip_without_mediapipe_is_a_503(monkeypatch, fresh_state):
    monkeypatch.setattr(lip_routes, "_import_error", "No module named 'cv2'")

    assert _post_clip([_png()]).status_code == 503


# --- /status ---------------------------------------------------------------


def test_status_lists_the_words_it_can_read(fresh_state, model_dir):
    assert client.get("/api/lips/status").json() == {
        "available": True,
        "model_ready": False,
        "reason": None,
        "words": list(VOCABULARY),
    }


def test_status_says_when_the_model_is_downloaded(fresh_state, model_dir):
    model_dir.mkdir()
    lip_routes.MODEL_PATH.write_bytes(b"modelo")

    assert client.get("/api/lips/status").json()["model_ready"] is True


def test_status_reports_missing_dependencies(monkeypatch, fresh_state):
    monkeypatch.setattr(lip_routes, "_import_error", "No module named 'cv2'")

    body = client.get("/api/lips/status").json()

    assert body["available"] is False
    assert body["model_ready"] is False
    assert "cv2" in body["reason"]


# --- Crear el detector -----------------------------------------------------


def test_detector_is_created_once_and_reused(vision):
    first = lip_routes._get_landmarker()
    second = lip_routes._get_landmarker()

    assert first is second
    assert len(vision.created) == 1
    options = vision.created[0].options
    assert options["base_options"] == str(lip_routes.MODEL_PATH)
    assert options["running_mode"] == "image"
    assert options["num_faces"] == 1
    assert options["output_face_blendshapes"] is True


def test_failure_to_create_the_detector_is_remembered(vision):
    vision.error = OSError("libGL.so.1: cannot open shared object file")

    with pytest.raises(RuntimeError, match="libGL"):
        lip_routes._get_landmarker()
    # La segunda vez no se reintenta: falla con el mismo motivo.
    with pytest.raises(RuntimeError, match="libGL"):
        lip_routes._get_landmarker()
    assert len(vision.created) == 1

    status = client.get("/api/lips/status").json()
    assert status["available"] is False
    assert status["model_ready"] is True
    assert "libGL" in status["reason"]

    assert _post_frame(_png()).status_code == 503


# --- Descargar el modelo ---------------------------------------------------


def test_model_is_downloaded_the_first_time(model_dir, downloads):
    path = lip_routes._ensure_model()

    assert path == lip_routes.MODEL_PATH
    assert path.read_bytes() == b"modelo"
    assert downloads[0][0] == lip_routes.MODEL_URL
    # Se baja a un archivo temporal y luego se renombra, para no dejar a
    # medias el bueno si la descarga se corta.
    assert downloads[0][1].name == "face_landmarker.task.part"
    assert not downloads[0][1].exists()


def test_model_already_downloaded_is_not_downloaded_again(model_dir, downloads):
    model_dir.mkdir()
    lip_routes.MODEL_PATH.write_bytes(b"modelo")

    assert lip_routes._ensure_model() == lip_routes.MODEL_PATH
    assert downloads == []


def test_empty_model_file_is_downloaded_again(model_dir, downloads):
    # Lo que queda si una descarga anterior falló a medias.
    model_dir.mkdir()
    lip_routes.MODEL_PATH.write_bytes(b"")

    lip_routes._ensure_model()

    assert len(downloads) == 1
    assert lip_routes.MODEL_PATH.read_bytes() == b"modelo"
