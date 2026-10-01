"""Lo que se puede comprobar sin descargar el modelo de MediaPipe.

Las validaciones de la foto van antes de crear el detector, así que estas
pruebas no tocan la red ni necesitan el modelo de 7,8 MB.
"""

from fastapi.testclient import TestClient

from multimodal_service.main import app

client = TestClient(app)


def test_status_says_whether_recognition_is_available():
    response = client.get("/api/signs/status")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"available", "model_ready", "reason"}
    assert isinstance(body["available"], bool)


def test_landmarks_rejects_files_that_are_not_images():
    response = client.post(
        "/api/signs/landmarks",
        files={"file": ("notas.txt", b"hola", "text/plain")},
    )

    assert response.status_code == 415


def test_landmarks_rejects_an_empty_image():
    response = client.post(
        "/api/signs/landmarks",
        files={"file": ("vacia.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
