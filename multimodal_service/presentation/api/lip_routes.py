"""Lectura de labios: ver la boca con la cámara y leer palabras sueltas.

Hay dos maneras de mirar, porque piden cosas distintas:

- `/landmarks` recibe una foto y devuelve el contorno de los labios y la forma
  de la boca. Es lo que la app dibuja encima de la cámara, en vivo.
- `/read` recibe un clip, las fotos de una palabra entera, y dice qué palabra
  parece. Va aparte porque leer exige muchas fotos seguidas: la app las toma a
  su ritmo y las manda juntas, así la lectura no depende de lo rápido que
  responda el servidor, que en el plan gratuito de Render es poco.

Las medidas y las palabras están en `domain/lip_reading.py`. Aquí solo se
reciben fotos y se le pregunta a MediaPipe dónde está la boca.

Privacidad: las fotos se procesan en memoria y se descartan. Nunca se guardan en
disco ni se envían a ningún otro servicio.
"""

import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from multimodal_service.domain.lip_reading import (
    DEFAULT_NEUTRAL_WIDTH,
    LIPS_INNER,
    LIPS_OUTER,
    SHAPES,
    VOCABULARY,
    classify_shape,
    measure_mouth,
    read_word,
    shape_scores,
)

router = APIRouter(prefix="/api/lips", tags=["Lectura de labios"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"

# Modelo oficial de MediaPipe para detectar los puntos de la cara.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# De 1 MB para arriba el servidor web guardaría la foto en un temporal mientras
# llega, y la promesa es que las fotos no tocan el disco. Una foto de la cámara
# a resolución media pesa bastante menos.
MAX_IMAGE_BYTES = 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# Unos 4 segundos a 15 fotos por segundo: de sobra para una palabra.
MAX_CLIP_FRAMES = 60
MAX_CLIP_BYTES = 16 * 1024 * 1024

# Las blendshapes de MediaPipe que dicen si la boca se redondea.
_ROUNDING_BLENDSHAPES = {"mouthPucker", "mouthFunnel"}

# El detector es caro de crear y no es seguro usarlo desde varios hilos a la
# vez, así que se crea una sola vez y se protege con un candado.
_landmarker = None
_landmarker_lock = threading.Lock()
_import_error: Optional[str] = None

# Por qué falló al crear el detector, si es que ya se intentó. Sin recordarlo,
# /status diría que todo va bien mientras cada petición falla.
_landmarker_error: Optional[str] = None

try:
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision
except Exception as exc:  # pragma: no cover - depende del entorno
    cv2 = None
    mp = None
    np = None
    _import_error = str(exc)


def _ensure_model() -> Path:
    """Descarga el modelo la primera vez y lo deja en caché."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix(".task.part")
    urllib.request.urlretrieve(MODEL_URL, temporary)
    temporary.replace(MODEL_PATH)
    return MODEL_PATH


def _get_landmarker():
    global _landmarker, _landmarker_error

    if _import_error is not None:
        raise RuntimeError(
            "Falta instalar mediapipe y opencv en el backend: " + _import_error
        )

    if _landmarker_error is not None:
        raise RuntimeError(_landmarker_error)

    if _landmarker is None:
        model = _ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            # Apenas cuesta tiempo (menos de 1 ms por foto) y es lo único que
            # distingue bien una boca redondeada sin haberla calibrado.
            output_face_blendshapes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
        )
        try:
            _landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            _landmarker_error = (
                f"El servidor no puede crear el detector de caras: {exc}"
            )
            raise RuntimeError(_landmarker_error)

    return _landmarker


def _landmarker_or_503():
    try:
        return _get_landmarker()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado: {file.content_type}",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="La imagen llegó vacía")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="La imagen es demasiado grande")
    return payload


def _rounding_of(result) -> float:
    if not result.face_blendshapes:
        return 0.0
    return max(
        (
            category.score
            for category in result.face_blendshapes[0]
            if category.category_name in _ROUNDING_BLENDSHAPES
        ),
        default=0.0,
    )


def _detect(landmarker, payload: bytes):
    """Busca la boca en una foto. Devuelve los puntos y sus medidas, o None."""
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo leer la imagen")

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )

    with _landmarker_lock:
        result = landmarker.detect(image)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    height, width = frame.shape[:2]
    mouth = measure_mouth(landmarks, width, height, _rounding_of(result))
    return None if mouth is None else (landmarks, mouth)


def _contour(landmarks, indices):
    return [
        {"x": landmarks[index].x, "y": landmarks[index].y} for index in indices
    ]


def _analyze_frame(landmarker, payload: bytes, neutral_width: float):
    detected = _detect(landmarker, payload)
    if detected is None:
        return {"face": None}

    landmarks, mouth = detected
    scores = shape_scores(mouth, neutral_width)
    return {
        "face": {
            "lips": {
                "outer": _contour(landmarks, LIPS_OUTER),
                "inner": _contour(landmarks, LIPS_INNER),
            },
            "opening": round(mouth.opening, 4),
            "width": round(mouth.width, 4),
            "rounding": round(mouth.rounding, 4),
            "shape": classify_shape(scores),
            "scores": {
                shape: round(float(score), 4)
                for shape, score in zip(SHAPES, scores)
            },
        }
    }


def _analyze_clip(landmarker, payloads, neutral_width: float):
    mouths = []
    for payload in payloads:
        try:
            detected = _detect(landmarker, payload)
        except ValueError:
            # Una foto mala en mitad de un clip no estropea el resto: cuenta
            # como una foto en la que no se vio la cara.
            continue
        if detected is not None:
            mouths.append(detected[1])

    reading = read_word(mouths, neutral_width, total_frames=len(payloads))
    return {
        "word": reading.word,
        "alternatives": list(reading.alternatives),
        "shapes": list(reading.shapes),
        "frames": reading.frames,
        "frames_with_face": reading.frames_with_face,
        "reason": reading.reason,
    }


@router.get("/status")
def lips_status():
    """Dice si la lectura está disponible y qué palabras sabe leer.

    La app lo consulta antes de ofrecer la cámara, para no abrirla y fallar.
    """
    words = list(VOCABULARY)

    if _import_error is not None:
        return {
            "available": False,
            "model_ready": False,
            "reason": "Faltan dependencias en el backend: " + _import_error,
            "words": words,
        }

    if _landmarker_error is not None:
        return {
            "available": False,
            "model_ready": MODEL_PATH.exists(),
            "reason": _landmarker_error,
            "words": words,
        }

    return {
        "available": True,
        "model_ready": MODEL_PATH.exists(),
        "reason": None,
        "words": words,
    }


@router.post("/landmarks", status_code=status.HTTP_200_OK)
async def detect_lips(
    file: UploadFile = File(...),
    neutral_width: Optional[float] = Form(None, gt=0.2, lt=1.2),
):
    """Devuelve el contorno de los labios y la forma de la boca en una foto.

    `neutral_width` es el ancho de la boca relajada de esta persona, en
    distancias entre ojos, si la app ya lo midió.

    Si no se ve ninguna cara, `face` llega en null: es un resultado normal.
    """
    payload = await _read_image(file)
    landmarker = _landmarker_or_503()

    try:
        # En un hilo aparte: todos los servicios comparten este proceso, y
        # analizar aquí mismo dejaría esperando al resto de rutas.
        return await run_in_threadpool(
            _analyze_frame,
            landmarker,
            payload,
            neutral_width or DEFAULT_NEUTRAL_WIDTH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/read", status_code=status.HTTP_200_OK)
async def read_lips(
    files: List[UploadFile] = File(...),
    neutral_width: Optional[float] = Form(None, gt=0.2, lt=1.2),
):
    """Lee la palabra que se dijo en un clip: las fotos en orden, en `files`.

    `word` solo viene si la lectura es clara. Si no, llega en null y
    `alternatives` trae las candidatas, para que decida la persona. `reason`
    explica por qué no se pudo ni intentar: `too_short`, `no_face` o
    `no_movement`.
    """
    if len(files) > MAX_CLIP_FRAMES:
        raise HTTPException(status_code=413, detail="El clip tiene demasiadas fotos")

    payloads = []
    total = 0
    for file in files:
        payload = await _read_image(file)
        total += len(payload)
        if total > MAX_CLIP_BYTES:
            raise HTTPException(status_code=413, detail="El clip pesa demasiado")
        payloads.append(payload)

    landmarker = _landmarker_or_503()

    return await run_in_threadpool(
        _analyze_clip,
        landmarker,
        payloads,
        neutral_width or DEFAULT_NEUTRAL_WIDTH,
    )
