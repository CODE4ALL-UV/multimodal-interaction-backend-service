"""Detección de la mano para el reconocimiento del alfabeto manual.

Este servicio hace una sola cosa: recibe una foto y devuelve las 21
coordenadas de la mano que MediaPipe encuentra en ella. **No decide qué letra
es**: de eso se encarga la app, que ya tiene la tabla del alfabeto y así puede
explicar al estudiante qué dedo le falta estirar.

Privacidad: la imagen se procesa en memoria y se descarta. Nunca se guarda en
disco ni se envía a ningún otro servicio.
"""

import threading
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status

router = APIRouter(prefix="/api/signs", tags=["Señas"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# Modelo oficial de MediaPipe para detectar los puntos de la mano.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_IMAGE_BYTES = 4 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# El detector es caro de crear y no es seguro usarlo desde varios hilos a la
# vez, así que se crea una sola vez y se protege con un candado.
_landmarker = None
_landmarker_lock = threading.Lock()
_import_error: Optional[str] = None

# Por que fallo al crear el detector, si es que ya se intento.
#
# Importar mediapipe puede salir bien y aun asi fallar al construirlo: le
# faltan librerias del sistema, o memoria. Sin recordarlo, /status seguiria
# diciendo que todo va bien mientras cada peticion falla, y la aplicacion
# ofreceria la camara para romperse justo despues.
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
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
        )
        try:
            _landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            # Se recuerda para no reintentarlo en cada peticion y, sobre todo,
            # para que /status lo diga en lugar de fingir que todo va bien.
            _landmarker_error = (
                f"El servidor no puede crear el detector de manos: {exc}"
            )
            raise RuntimeError(_landmarker_error)

    return _landmarker


@router.get("/status")
def sign_status():
    """Dice si el reconocimiento está disponible, sin llegar a usarlo.

    La app lo consulta antes de ofrecer la cámara, para no abrirla y fallar.
    """
    if _import_error is not None:
        return {
            "available": False,
            "model_ready": False,
            "reason": "Faltan dependencias en el backend: " + _import_error,
        }

    if _landmarker_error is not None:
        return {
            "available": False,
            "model_ready": MODEL_PATH.exists(),
            "reason": _landmarker_error,
        }

    return {
        "available": True,
        "model_ready": MODEL_PATH.exists(),
        "reason": None,
    }


@router.post("/landmarks", status_code=status.HTTP_200_OK)
async def detect_landmarks(file: UploadFile = File(...)):
    """Devuelve los 21 puntos de la mano que aparezca en la foto.

    Respuesta:
        {"hands": [{"handedness": "Right",
                    "landmarks": [{"x": 0.5, "y": 0.4, "z": 0.0}, ...]}]}

    Una lista vacía significa que no se vio ninguna mano, que es un resultado
    normal, no un error.
    """
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

    try:
        landmarker = _get_landmarker()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    buffer = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )

    with _landmarker_lock:
        result = landmarker.detect(image)

    hands = []
    for index, landmarks in enumerate(result.hand_landmarks):
        handedness = ""
        if result.handedness and index < len(result.handedness):
            categories = result.handedness[index]
            if categories:
                handedness = categories[0].category_name

        hands.append(
            {
                "handedness": handedness,
                "landmarks": [
                    {"x": point.x, "y": point.y, "z": point.z}
                    for point in landmarks
                ],
            }
        )

    return {"hands": hands}
